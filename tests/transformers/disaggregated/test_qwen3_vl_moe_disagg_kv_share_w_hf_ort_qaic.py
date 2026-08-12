# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------
"""Token-level parity test for the Qwen3-VL-MoE disaggregated prefill/decode DMA path.
pytest -m "on_qaic and multimodal" tests/transformers/disaggregated/test_qwen3_vl_moe_disagg_kv_share_w_hf_ort_qaic.py
"""
import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor

from QEfficient import QEFFAutoModelForImageTextToText
from QEfficient.generation.cloud_infer import QAICInferenceSession


#MODEL_NAME = "tiny-random/qwen3-vl-moe"
MODEL_NAME = "Qwen/Qwen3-VL-30B-A3B-Instruct"
PREFILL_SEQ_LEN = 128
CTX_LEN = 4096
BATCH_SIZE = 1
GENERATION_LEN = 40
IMAGE_SIZE = (536,354)
TEXT_PROMPT = "Describe this image."

VISION_INPUTS = {
    "pixel_values",
    "image_grid_thw",
    "image_masks",
    "image_input_idx",
    "valid_idx",
    "aspect_ratio_ids",
    "aspect_ratio_mask",
}
VISION_FP16_INPUTS = {"pixel_values", "image_masks"}
VISION_OUTPUTS = ("vision_embeds", "deepstack_features")


# ---------------------------------------------------------------------------
# Helpers shared by all test variants
# ---------------------------------------------------------------------------


def _assert_onnx_path(onnx_path, label: str) -> Path:
    assert onnx_path is not None, f"{label} compile did not set an ONNX path"
    onnx_path = Path(onnx_path)
    assert onnx_path.is_file(), f"{label} ONNX path does not exist: {onnx_path}"
    assert onnx_path.suffix == ".onnx", f"{label} path is not an ONNX file: {onnx_path}"
    return onnx_path.resolve()


def _assert_distinct_onnx_paths(onnx_paths: dict[str, Path]):
    unique_paths = {str(path) for path in onnx_paths.values()}
    assert len(unique_paths) == len(onnx_paths), f"Expected distinct ONNX paths per compile, got: {onnx_paths}"


def _load_hf_model_from_pretrained(config):
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_NAME,
            config=config,
            attn_implementation="eager",
            trust_remote_code=True,
            torch_dtype=config.torch_dtype,
        )
    except ValueError:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            config=config,
            attn_implementation="eager",
            trust_remote_code=True,
            torch_dtype=config.torch_dtype,
        )
    model.eval()
    return model


def _build_config(dtype: str = "float32"):
    config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    config.dtype = dtype
    config.torch_dtype = getattr(torch, dtype)
    config.vision_config.depth = 9
    config.text_config.num_hidden_layers = 4
    config.vision_config.deepstack_visual_indexes = [8]
    return config


def _prepare_messages(image: Image.Image) -> list:
    return [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": TEXT_PROMPT},
                ],
            }
        ]
        for _ in range(BATCH_SIZE)
    ]


def _prepare_processor_inputs(processor: AutoProcessor, messages: list) -> dict:
    process_vision_info = pytest.importorskip("qwen_vl_utils").process_vision_info
    texts = [processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True) for message in messages]
    image_inputs, video_inputs = process_vision_info(messages)
    return dict(processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"))


def _get_next_token_ids(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits)
    return logits[:, -1, :].argmax(axis=-1).astype(np.int64)


def _run_hf_torch_fp32(model, processor: AutoProcessor, messages: list) -> np.ndarray:
    model = model.to(dtype=torch.float32).eval()
    inputs = _prepare_processor_inputs(processor, messages)
    inputs = {
        name: value.to(dtype=torch.float32) if torch.is_floating_point(value) else value
        for name, value in inputs.items()
    }
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=GENERATION_LEN,
            min_new_tokens=GENERATION_LEN,
            do_sample=False,
        )
    prompt_len = inputs["input_ids"].shape[-1]
    return outputs[:, prompt_len:].detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Shared QEff model / input preparation
# ---------------------------------------------------------------------------


def _build_qeff_model(hf_model) -> QEFFAutoModelForImageTextToText:
    hf_model.config.dtype = "float32"
    hf_model.config.torch_dtype = torch.float32
    if hasattr(hf_model.config, "text_config"):
        hf_model.config.text_config.dtype = "float32"
        hf_model.config.text_config.torch_dtype = torch.float32
    return QEFFAutoModelForImageTextToText(
        hf_model,
        kv_offload=True,
        config=hf_model.config,
        torch_dtype=torch.float32,
        layerwise=False,
    )


def _prepare_qeff_inputs(qeff_model, processor, common_inputs, image, cast_vision_fp16: bool = True):
    """Return (np_inputs, vision_inputs, lang_inputs, num_chunks, padded_len).

    Args:
        cast_vision_fp16: If True (default, for QAIC), cast VISION_FP16_INPUTS to float16.
                          Set False for ORT runs where the exported graph expects float32.
    """
    inputs = {
        name: value.clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
        for name, value in common_inputs.items()
    }
    inputs = qeff_model.model.prepare_inputs_for_generation(
        inputs=inputs,
        prefill_seq_len=PREFILL_SEQ_LEN,
        batch_size=BATCH_SIZE,
    )
    pad_token_id = processor.tokenizer.pad_token_id or 1
    input_ids_length = inputs["input_ids"].shape[1]
    num_chunks = -(input_ids_length // -PREFILL_SEQ_LEN)
    padded_len = num_chunks * PREFILL_SEQ_LEN

    inputs["input_ids"] = torch.nn.functional.pad(
        inputs["input_ids"], (0, padded_len - input_ids_length), "constant", pad_token_id
    )
    inputs["attention_mask"] = torch.nn.functional.pad(
        inputs["attention_mask"], (0, padded_len - input_ids_length), "constant", 0
    )
    np_inputs = {name: np.array(value) for name, value in inputs.items()}

    vision_inputs = {name: value for name, value in np_inputs.items() if name in VISION_INPUTS}
    if cast_vision_fp16:
        # QAIC path: cast pixel_values and image_masks to float16
        vision_inputs.update(
            {name: vision_inputs[name].astype("float16") for name in VISION_FP16_INPUTS if name in vision_inputs}
        )
    else:
        # ORT fp32 path: ensure float inputs stay float32
        vision_inputs.update(
            {name: vision_inputs[name].astype("float32") for name in VISION_FP16_INPUTS if name in vision_inputs}
        )

    lang_inputs = {name: value for name, value in np_inputs.items() if name not in vision_inputs}
    if "position_ids" in np_inputs:
        lang_inputs["position_ids"] = np_inputs["position_ids"]
        lang_inputs.pop("attention_mask", None)
    else:
        lang_inputs["position_ids"] = np.where(lang_inputs.pop("attention_mask"), np.arange(padded_len), -1)
    lang_inputs["image_idx"] = np.array([[0]], dtype=np.int64)

    return np_inputs, vision_inputs, lang_inputs, num_chunks, padded_len


# ---------------------------------------------------------------------------
# ORT helpers (ported from the reference debug script)
# ---------------------------------------------------------------------------


def _patch_custom_rmsnorm_for_ort(path: Path) -> Path:
    """Patch exported ONNX so ORT can run FP32 graphs with CustomRMSNorm."""
    try:
        import onnx
        from onnx import helper
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("onnx is required for ORT parity test.") from exc

    model = onnx.load(str(path), load_external_data=False)
    changed = False

    def _set_cast_to_int64(node):
        for attr in node.attribute:
            if attr.name == "to":
                attr.i = onnx.TensorProto.INT64
                return
        node.attribute.extend([helper.make_attribute("to", onnx.TensorProto.INT64)])

    def _make_const_i64_1d(name: str, values: list):
        return helper.make_node(
            "Constant",
            [],
            [name],
            value=helper.make_tensor(name, onnx.TensorProto.INT64, [len(values)], values),
        )

    def _add_scatter_safe_position_ids(new_nodes: list):
        new_nodes.extend(
            [
                helper.make_node("Shape", ["data"], ["scatter_data_shape"]),
                _make_const_i64_1d("scatter_dim_one", [1]),
                helper.make_node("Gather", ["scatter_data_shape", "scatter_dim_one"], ["scatter_ctx_dim"]),
                _make_const_i64_1d("scatter_one", [1]),
                helper.make_node("Sub", ["scatter_ctx_dim", "scatter_one"], ["scatter_last_idx_i64"]),
                helper.make_node("CastLike", ["scatter_last_idx_i64", "position_ids"], ["scatter_last_idx"]),
                _make_const_i64_1d("scatter_invalid_i64", [2147483647]),
                helper.make_node("CastLike", ["scatter_invalid_i64", "position_ids"], ["scatter_invalid"]),
                helper.make_node("Equal", ["position_ids", "scatter_invalid"], ["scatter_invalid_mask"]),
                helper.make_node(
                    "Where",
                    ["scatter_invalid_mask", "scatter_last_idx", "position_ids"],
                    ["scatter_safe_position_ids"],
                ),
            ]
        )

    for function in model.functions:
        function_changed = False
        new_nodes = []

        if function.name == "CustomRMSNorm":
            for node in function.node:
                if node.op_type == "Cast" and list(node.input) == ["weight"] and list(node.output) == ["weight_0"]:
                    node = helper.make_node(
                        "CastLike",
                        ["weight", "hidden_states"],
                        ["weight_0"],
                        name=node.name or "CastLike_weight",
                    )
                    function_changed = True
                if node.op_type == "Expand" and list(node.output) == ["epsilon_2"]:
                    node.output[0] = "epsilon_2_before_cast"
                    new_nodes.append(node)
                    new_nodes.append(
                        helper.make_node(
                            "CastLike",
                            ["epsilon_2_before_cast", "variance"],
                            ["epsilon_2"],
                            name="CastLike_epsilon",
                        )
                    )
                    function_changed = True
                    continue
                new_nodes.append(node)

        elif function.name.startswith("CtxScatter"):
            inserted_safe_position_ids = False
            for node in function.node:
                if not inserted_safe_position_ids and "position_ids" in node.input:
                    _add_scatter_safe_position_ids(new_nodes)
                    inserted_safe_position_ids = True
                    function_changed = True
                for idx, input_name in enumerate(node.input):
                    if inserted_safe_position_ids and input_name == "position_ids":
                        node.input[idx] = "scatter_safe_position_ids"
                if node.op_type == "Cast" and list(node.output) == ["batch_idx_3"]:
                    _set_cast_to_int64(node)
                    function_changed = True
                if node.op_type == "Expand" and list(node.output) == ["ctx_idx"]:
                    node.output[0] = "ctx_idx_before_cast"
                    new_nodes.append(node)
                    new_nodes.append(
                        helper.make_node(
                            "Cast",
                            ["ctx_idx_before_cast"],
                            ["ctx_idx"],
                            name="Cast_ctx_idx_i64",
                            to=onnx.TensorProto.INT64,
                        )
                    )
                    function_changed = True
                    continue
                new_nodes.append(node)

        elif function.name.startswith("CtxGather"):
            inserted_index_cast = False
            for node in function.node:
                if (
                    not inserted_index_cast
                    and node.op_type in {"Expand", "Unsqueeze", "GatherND"}
                    and "ctx_indices" in node.input
                ):
                    new_nodes.append(
                        helper.make_node(
                            "Cast",
                            ["ctx_indices"],
                            ["ctx_indices_i64"],
                            name="Cast_ctx_indices_i64",
                            to=onnx.TensorProto.INT64,
                        )
                    )
                    inserted_index_cast = True
                    function_changed = True
                for idx, input_name in enumerate(node.input):
                    if inserted_index_cast and input_name == "ctx_indices":
                        node.input[idx] = "ctx_indices_i64"
                new_nodes.append(node)

        else:
            continue  # no changes needed for this function

        if function_changed:
            del function.node[:]
            function.node.extend(new_nodes)
            changed = True

    if not changed:
        return path
    patched_path = path.with_name(f"{path.stem}.ort.onnx")
    onnx.save(model, str(patched_path))
    return patched_path


def _make_ort_session(path: Path):
    try:
        import onnxruntime as ort
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("onnxruntime is required for ORT parity test.") from exc

    ort_path = _patch_custom_rmsnorm_for_ort(path)
    if ort_path != path:
        print(f"ORT patched ONNX: {path} -> {ort_path}")
    return ort.InferenceSession(str(ort_path), providers=["CPUExecutionProvider"])


def _session_input_names(session) -> list:
    return [item.name for item in session.get_inputs()]


def _session_output_names(session) -> list:
    return [item.name for item in session.get_outputs()]


def _session_input_rank(session, name: str):
    for item in session.get_inputs():
        if item.name == name:
            return len(item.shape)
    return None


def _qeff_rank4_image_grid_thw(grid) -> np.ndarray:
    arr = np.asarray(grid)
    if arr.ndim != 2 or arr.shape[-1] != 3:
        return arr
    if arr.shape[0] != 1:
        raise NotImplementedError(f"Rank-4 grid conversion supports one image, got {arr.shape}")
    t, h, w = (int(v) for v in arr[0])
    return np.zeros((arr.shape[0], t, h, w), dtype=arr.dtype)


def _vision_feed_for_ort(vision_inputs: dict, vision_session) -> dict:
    feed = {k: v for k, v in vision_inputs.items() if k in _session_input_names(vision_session)}
    if "image_grid_thw" in feed and _session_input_rank(vision_session, "image_grid_thw") == 4:
        original = np.asarray(feed["image_grid_thw"])
        feed["image_grid_thw"] = _qeff_rank4_image_grid_thw(original)
    return feed


def _dtype_for_ort(type_name: str) -> np.dtype:
    if "float16" in type_name:
        return np.float16
    if "float" in type_name:
        return np.float32
    if "int64" in type_name:
        return np.int64
    if "int32" in type_name:
        return np.int32
    return np.float32


def _resolve_ort_dim(dim, seq_len: int) -> int:
    if isinstance(dim, int):
        return dim
    if dim in {"batch_size", "batch", "full_batch_size", "full_batch"}:
        return BATCH_SIZE
    if dim in {"seq_len", "sequence_length"}:
        return seq_len
    if dim in {"ctx_len", "context_length", "past_sequence_length"}:
        return CTX_LEN
    if dim in {"num_logits_to_keep"}:
        return 1
    raise ValueError(f"Cannot resolve dynamic ONNX dim {dim!r}")


def _empty_input_from_meta(meta, seq_len: int) -> np.ndarray:
    shape = tuple(_resolve_ort_dim(dim, seq_len) for dim in meta.shape)
    return np.zeros(shape, dtype=_dtype_for_ort(meta.type))


def _ensure_session_inputs(session, provided: dict, state: dict, seq_len: int) -> dict:
    merged = {}
    for meta in session.get_inputs():
        if meta.name in provided:
            merged[meta.name] = provided[meta.name]
        elif meta.name in state:
            merged[meta.name] = state[meta.name]
        else:
            merged[meta.name] = _empty_input_from_meta(meta, seq_len)
    return merged


def _update_state_from_outputs(state: dict, outputs: dict) -> None:
    for name, value in outputs.items():
        if name.endswith("_RetainedState"):
            key = name[: -len("_RetainedState")]
            state[key] = value


def _run_ort_generation(
    onnx_paths: dict[str, Path],
    vision_inputs: dict,
    lang_inputs: dict,
    num_chunks: int,
    processor: AutoProcessor,
) -> np.ndarray:
    """Run the disaggregated vision -> prefill -> decode loop with ORT (fp32)."""
    vision_session = _make_ort_session(onnx_paths["vision"])
    prefill_session = _make_ort_session(onnx_paths["prefill"])
    decode_session = _make_ort_session(onnx_paths["decode"])

    print(f"[ORT] vision_inputs  : {_session_input_names(vision_session)}")
    print(f"[ORT] prefill_inputs : {_session_input_names(prefill_session)}")
    print(f"[ORT] decode_inputs  : {_session_input_names(decode_session)}")

    # --- Vision ---
    vision_feed = _vision_feed_for_ort(vision_inputs, vision_session)
    vision_outputs = dict(zip(_session_output_names(vision_session), vision_session.run(None, vision_feed)))
    persistent = {k: vision_outputs[k] for k in VISION_OUTPUTS if k in vision_outputs}

    state: dict = {}
    chunk_inputs = dict(lang_inputs)
    prefill_outputs: dict = {}

    # --- Prefill ---
    for chunk_idx in range(num_chunks):
        start = chunk_idx * PREFILL_SEQ_LEN
        end = (chunk_idx + 1) * PREFILL_SEQ_LEN
        provided = dict(persistent)
        provided["input_ids"] = lang_inputs["input_ids"][:, start:end]
        provided["position_ids"] = lang_inputs["position_ids"][..., start:end]
        provided["image_idx"] = chunk_inputs.get("image_idx", np.array([[0]], dtype=np.int64))

        feed = _ensure_session_inputs(prefill_session, provided, state, PREFILL_SEQ_LEN)
        prefill_outputs = dict(zip(_session_output_names(prefill_session), prefill_session.run(None, feed)))
        _update_state_from_outputs(state, prefill_outputs)
        if "image_idx_output" in prefill_outputs:
            chunk_inputs["image_idx"] = prefill_outputs["image_idx_output"]

    first_token = _get_next_token_ids(prefill_outputs["logits"])
    tokens = [first_token]

    num_pos_sections = lang_inputs["position_ids"].shape[0]
    phys_pos = int(lang_inputs["position_ids"][0].max()) + 1
    mrope_pos = int(lang_inputs["position_ids"][1:].max()) + 1 if num_pos_sections > 1 else phys_pos

    def _decode_position_ids(next_phys: int, next_mrope: int) -> np.ndarray:
        pos = np.empty((num_pos_sections, BATCH_SIZE, 1), dtype=np.int64)
        pos[0] = next_phys
        if num_pos_sections > 1:
            pos[1:] = next_mrope
        return pos

    # --- Decode ---
    decode_inputs: dict = {
        "input_ids": first_token.reshape(BATCH_SIZE, 1),
        "position_ids": _decode_position_ids(phys_pos, mrope_pos),
    }
    if "image_idx_output" in prefill_outputs:
        decode_inputs["image_idx"] = prefill_outputs["image_idx_output"]

    for _ in range(GENERATION_LEN - 1):
        provided = dict(persistent)
        provided.update(decode_inputs)
        feed = _ensure_session_inputs(decode_session, provided, state, 1)
        decode_outputs = dict(zip(_session_output_names(decode_session), decode_session.run(None, feed)))
        _update_state_from_outputs(state, decode_outputs)

        tok = _get_next_token_ids(decode_outputs["logits"])
        tokens.append(tok)
        phys_pos += 1
        mrope_pos += 1
        decode_inputs = {
            "input_ids": tok.reshape(BATCH_SIZE, 1),
            "position_ids": _decode_position_ids(phys_pos, mrope_pos),
        }
        if "image_idx_output" in decode_outputs:
            decode_inputs["image_idx"] = decode_outputs["image_idx_output"]

    return np.stack(tokens, axis=1).astype(np.int64)


# ---------------------------------------------------------------------------
# QAIC generation
# ---------------------------------------------------------------------------


def _run_disagg_kv_share_qaic_generation(
    qeff_model: QEFFAutoModelForImageTextToText,
    processor: AutoProcessor,
    common_inputs: dict,
    vision_session: QAICInferenceSession,
    prefill_session: QAICInferenceSession,
    decode_session: QAICInferenceSession,
) -> np.ndarray:
    inputs = {
        name: value.clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
        for name, value in common_inputs.items()
    }
    inputs = qeff_model.model.prepare_inputs_for_generation(
        inputs=inputs,
        prefill_seq_len=PREFILL_SEQ_LEN,
        batch_size=BATCH_SIZE,
    )
    pad_token_id = processor.tokenizer.pad_token_id or 1
    input_ids_length = inputs["input_ids"].shape[1]
    num_chunks = -(input_ids_length // -PREFILL_SEQ_LEN)
    padded_len = num_chunks * PREFILL_SEQ_LEN
    inputs["input_ids"] = torch.nn.functional.pad(
        inputs["input_ids"], (0, padded_len - input_ids_length), "constant", pad_token_id
    )
    inputs["attention_mask"] = torch.nn.functional.pad(
        inputs["attention_mask"], (0, padded_len - input_ids_length), "constant", 0
    )
    inputs = {name: np.array(value) for name, value in inputs.items()}

    vision_inputs = {name: value for name, value in inputs.items() if name in VISION_INPUTS}
    vision_inputs.update(
        {name: vision_inputs[name].astype("float16") for name in VISION_FP16_INPUTS if name in vision_inputs}
    )
    vision_outputs = vision_session.run(vision_inputs)
    vision_session.deactivate()

    lang_inputs = {name: value for name, value in inputs.items() if name not in vision_inputs}
    if "position_ids" in inputs:
        lang_inputs["position_ids"] = inputs["position_ids"]
        lang_inputs.pop("attention_mask", None)
    else:
        lang_inputs["position_ids"] = np.where(lang_inputs.pop("attention_mask"), np.arange(padded_len), -1)
    lang_inputs["image_idx"] = np.array([[0]])

    assert "image_idx" in prefill_session.binding_index_map, "image_idx not a compiled prefill input binding"
    decode_has_image_idx = "image_idx" in decode_session.binding_index_map

    vision_persist = {name: vision_outputs[name] for name in VISION_OUTPUTS if name in vision_outputs}
    prefill_session.set_persistent_inputs(vision_persist)
    decode_session.set_persistent_inputs(
        {name: value for name, value in vision_persist.items() if name in decode_session.binding_index_map}
    )

    kv_caches = [np.zeros(shape, dtype=dtype) for (shape, dtype) in decode_session.kv_cache_info]
    chunk_inputs = dict(lang_inputs)
    exec_idx = None

    for chunk_idx in range(num_chunks):
        chunk_inputs["input_ids"] = lang_inputs["input_ids"][
            :, chunk_idx * PREFILL_SEQ_LEN : (chunk_idx + 1) * PREFILL_SEQ_LEN
        ]
        chunk_inputs["position_ids"] = lang_inputs["position_ids"][
            ..., chunk_idx * PREFILL_SEQ_LEN : (chunk_idx + 1) * PREFILL_SEQ_LEN
        ]
        last_chunk = chunk_idx == num_chunks - 1
        exec_idx = prefill_session.np_run_pipeline(
            chunk_inputs,
            last_chunk=last_chunk,
            kv_cache_buffers=kv_caches if last_chunk else None,
        )
        prefill_session.complete_inf(exec_idx, is_prefill=True)
        chunk_inputs["image_idx"] = prefill_session.get_outputs(index=exec_idx)["image_idx_output"]

    prefill_out = prefill_session.get_outputs(index=exec_idx)
    generated_ids = [_get_next_token_ids(prefill_out["logits"])]

    decode_kv_map = decode_session.decode_buff_map + decode_session.decode_rs_kv_only_buff_map
    num_pos_sections = lang_inputs["position_ids"].shape[0]
    phys_pos = int(lang_inputs["position_ids"][0].max()) + 1
    mrope_pos = int(lang_inputs["position_ids"][1:].max()) + 1

    def _decode_position_ids(next_phys: int, next_mrope: int) -> np.ndarray:
        pos = np.empty((num_pos_sections, BATCH_SIZE, 1), dtype=np.int64)
        pos[0] = next_phys
        pos[1:] = next_mrope
        return pos

    decode_inputs = {
        "input_ids": generated_ids[-1].reshape(BATCH_SIZE, 1),
        "position_ids": _decode_position_ids(phys_pos, mrope_pos),
    }
    if decode_has_image_idx:
        decode_inputs["image_idx"] = prefill_out["image_idx_output"]

    for _ in range(GENERATION_LEN - 1):
        decode_session.set_data_for_kv_handoff(
            kv_caches + kv_caches,
            [("batch_index", 0), ("ctx_start", 0)],
            index=decode_session.decode_execObj_idx,
            buff_map=decode_kv_map,
        )
        exec_idx = decode_session.np_run(decode_inputs, is_prefill=False)
        decode_session.complete_inf(exec_idx, is_prefill=False)
        decode_outputs = decode_session.get_outputs(index=exec_idx)
        generated_ids.append(_get_next_token_ids(decode_outputs["logits"]))
        phys_pos += 1
        mrope_pos += 1
        decode_inputs = {
            "input_ids": generated_ids[-1].reshape(BATCH_SIZE, 1),
            "position_ids": _decode_position_ids(phys_pos, mrope_pos),
        }
        if decode_has_image_idx:
            decode_inputs["image_idx"] = decode_outputs["image_idx_output"]

    return np.stack(generated_ids, axis=1)


def _compile_disagg_qpcs(qeff_model, image: Image.Image, compiled_onnx_paths: dict[str, Path]) -> dict[str, str]:
    """Compile QEfficient QPCs and record the ONNX path used for each compile."""
    vision_qpc_path = qeff_model.compile(
        batch_size=BATCH_SIZE,
        prefill_seq_len=PREFILL_SEQ_LEN,
        ctx_len=CTX_LEN,
        height=image.height,
        width=image.width,
        num_cores=16,
        num_devices=1,
        mos=1,
        aic_enable_depth_first=True,
        skip_vision=False,
        split_model_io=True,
        skip_lang=True,
        use_onnx_subfunctions=True,
        layerwise=False,
        offload_pt_weights=False,
    )
    compiled_onnx_paths["vision"] = _assert_onnx_path(qeff_model.vision_model.onnx_path, "vision")

    decode_qpc_path = qeff_model.compile(
        batch_size=BATCH_SIZE,
        prefill_seq_len=1,
        ctx_len=CTX_LEN,
        height=image.height,
        width=image.width,
        num_cores=16,
        num_devices=1,
        retain_full_kv=True,
        split_retained_state_io=True,
        mos=1,
        aic_enable_depth_first=True,
        prefill_only=False,
        skip_vision=True,
        use_onnx_subfunctions=True,
        layerwise=False,
        offload_pt_weights=False,
    )
    compiled_onnx_paths["decode"] = _assert_onnx_path(qeff_model.lang_model.onnx_path, "decode")

    prefill_qpc_path = qeff_model.compile(
        batch_size=BATCH_SIZE,
        prefill_seq_len=PREFILL_SEQ_LEN,
        ctx_len=CTX_LEN,
        height=image.height,
        width=image.width,
        num_cores=16,
        num_devices=2,
        split_retained_state_io=True,
        mos=1,
        aic_enable_depth_first=True,
        mdp_num_partitions=2,
        prefill_only=True,
        enable_chunking=True,
        skip_vision=True,
        use_onnx_subfunctions=True,
        layerwise=False,
        offload_pt_weights=False,
    )
    compiled_onnx_paths["prefill"] = _assert_onnx_path(qeff_model.lang_model.onnx_path, "prefill")

    _assert_distinct_onnx_paths(compiled_onnx_paths)
    print(f"Disagg ONNX paths: {compiled_onnx_paths}")
    return {
        "vision": vision_qpc_path.get("vision_qpc_path"),
        "prefill": prefill_qpc_path.get("lang_prefill_qpc_path"),
        "decode": decode_qpc_path.get("lang_decode_qpc_path"),
    }


def _create_disagg_sessions(qpc_paths: dict[str, str], sessions: list):
    vision_session = QAICInferenceSession(qpc_paths["vision"])
    prefill_session = QAICInferenceSession(qpc_paths["prefill"], kv_dma_share=True)
    decode_session = QAICInferenceSession(qpc_paths["decode"], kv_dma_share=True)
    sessions.extend([vision_session, prefill_session, decode_session])
    return vision_session, prefill_session, decode_session


# ---------------------------------------------------------------------------
# Shared token-comparison helper
# ---------------------------------------------------------------------------


def _assert_tokens_match(label_a: str, tokens_a: np.ndarray, label_b: str, tokens_b: np.ndarray) -> None:
    assert tokens_a.shape == (BATCH_SIZE, GENERATION_LEN), f"{label_a} shape mismatch: {tokens_a.shape}"
    assert tokens_b.shape == (BATCH_SIZE, GENERATION_LEN), f"{label_b} shape mismatch: {tokens_b.shape}"
    assert np.issubdtype(tokens_a.dtype, np.integer), f"{label_a} tokens are not integer dtype"
    assert np.issubdtype(tokens_b.dtype, np.integer), f"{label_b} tokens are not integer dtype"

    matches = tokens_a == tokens_b
    num_matched = int(matches.all(axis=0).cumprod().sum())
    print(f"{label_a} tokens : {tokens_a.tolist()}")
    print(f"{label_b} tokens : {tokens_b.tolist()}")
    print(f"Matched leading tokens : {num_matched}/{GENERATION_LEN}")

    if not matches.all():
        first_mismatch = int(np.argmin(matches.all(axis=0)))
        raise AssertionError(
            f"Tokens don't match for {label_a} vs {label_b}; "
            f"first mismatch at token index {first_mismatch} "
            f"(matched {num_matched}/{GENERATION_LEN} leading tokens): "
            f"{label_a}={tokens_a[:, first_mismatch].tolist()} vs "
            f"{label_b}={tokens_b[:, first_mismatch].tolist()}"
        )


def _assert_three_way_tokens_match(hf_tokens: np.ndarray, ort_tokens: np.ndarray, qaic_tokens: np.ndarray) -> None:
    comparisons = (
        ("HF fp32", hf_tokens, "ORT fp32", ort_tokens),
        ("ORT fp32", ort_tokens, "QAIC disagg DMA", qaic_tokens),
        ("HF fp32", hf_tokens, "QAIC disagg DMA", qaic_tokens),
    )
    failures = []
    for label_a, tokens_a, label_b, tokens_b in comparisons:
        assert tokens_a.shape == (BATCH_SIZE, GENERATION_LEN), f"{label_a} shape mismatch: {tokens_a.shape}"
        assert tokens_b.shape == (BATCH_SIZE, GENERATION_LEN), f"{label_b} shape mismatch: {tokens_b.shape}"
        assert np.issubdtype(tokens_a.dtype, np.integer), f"{label_a} tokens are not integer dtype"
        assert np.issubdtype(tokens_b.dtype, np.integer), f"{label_b} tokens are not integer dtype"

        matching_steps = (tokens_a == tokens_b).all(axis=0)
        num_matched = int(matching_steps.cumprod().sum())
        print(f"{label_a} vs {label_b} matched leading tokens : {num_matched}/{GENERATION_LEN}")
        if not matching_steps.all():
            first_mismatch = int(np.flatnonzero(~matching_steps)[0])
            failures.append(
                f"{label_a} vs {label_b}: first mismatch at token index {first_mismatch} "
                f"(matched {num_matched}/{GENERATION_LEN} leading tokens): "
                f"{label_a}={tokens_a[:, first_mismatch].tolist()} vs "
                f"{label_b}={tokens_b[:, first_mismatch].tolist()}"
            )

    if failures:
        raise AssertionError("Three-way parity mismatch:\n" + "\n".join(failures))


# ---------------------------------------------------------------------------
# Test: QAIC vs ORT  (three-way parity: HF == ORT == QAIC)
# ---------------------------------------------------------------------------
@pytest.mark.on_qaic
@pytest.mark.multimodal
@pytest.mark.disagg_dma
@pytest.mark.nightly_disagg
def test_qwen3_vl_moe_disagg_kv_share_qaic_vs_ort_fp32(manual_cleanup):
    """Three-way parity: HF fp32 == ORT on QPC ONNX == QAIC disagg DMA."""
    pytest.importorskip("qwen_vl_utils")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")

    torch.manual_seed(42)
    hf_model = _load_hf_model_from_pretrained(_build_config(dtype="float32"))
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

    image = Image.new("RGB", IMAGE_SIZE, color=(127, 127, 127))
    messages = _prepare_messages(image)
    common_inputs = _prepare_processor_inputs(processor, messages)
    hf_tokens = _run_hf_torch_fp32(hf_model, processor, messages)

    qeff_model = _build_qeff_model(hf_model)
    sessions = []
    compiled_onnx_paths = {}
    try:
        qpc_paths = _compile_disagg_qpcs(qeff_model, image, compiled_onnx_paths)
        _, vision_inputs, lang_inputs, num_chunks, _ = _prepare_qeff_inputs(
            qeff_model,
            processor,
            common_inputs,
            image,
            cast_vision_fp16=False,
        )
        ort_tokens = _run_ort_generation(compiled_onnx_paths, vision_inputs, lang_inputs, num_chunks, processor)

        vision_session, prefill_session, decode_session = _create_disagg_sessions(qpc_paths, sessions)
        qaic_tokens = _run_disagg_kv_share_qaic_generation(
            qeff_model=qeff_model,
            processor=processor,
            common_inputs=common_inputs,
            vision_session=vision_session,
            prefill_session=prefill_session,
            decode_session=decode_session,
        )
    finally:
        for session in sessions:
            session.deactivate()
        cleanup_paths = list(compiled_onnx_paths.values()) or [
            getattr(qeff_model.vision_model, "onnx_path", None),
            getattr(qeff_model.lang_model, "onnx_path", None),
        ]
        manual_cleanup([path for path in cleanup_paths if path is not None])

    hf_text = processor.tokenizer.batch_decode(hf_tokens, skip_special_tokens=True)
    ort_text = processor.tokenizer.batch_decode(ort_tokens, skip_special_tokens=True)
    qaic_text = processor.tokenizer.batch_decode(qaic_tokens, skip_special_tokens=True)
    print(f"HF   tokens : {hf_tokens.tolist()}")
    print(f"ORT  tokens : {ort_tokens.tolist()}")
    print(f"QAIC tokens : {qaic_tokens.tolist()}")
    print(f"HF   text   : {hf_text}")
    print(f"ORT  text   : {ort_text}")
    print(f"QAIC text   : {qaic_text}")

    _assert_three_way_tokens_match(hf_tokens, ort_tokens, qaic_tokens)
