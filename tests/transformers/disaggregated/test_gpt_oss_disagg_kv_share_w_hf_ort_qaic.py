# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

"""Token-level parity test for the gpt-oss disaggregated prefill/decode DMA path.
pytest -m "on_qaic" tests/transformers/disaggregated/test_gpt_oss_disagg_kv_share_w_hf_ort_qaic.py
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.generation.cloud_infer import QAICInferenceSession

MODEL_NAME = "openai/gpt-oss-20b"
TOKENIZER_ID = MODEL_NAME
NUM_HIDDEN_LAYERS = 4
PREFILL_SEQ_LEN = 32
CTX_LEN = 256
BATCH_SIZE = 1
GENERATION_LEN = 50
SLIDING_WINDOW = 128
TEXT_PROMPT = "Explain quantum computing in simple terms."

NUM_CORES = 16
MOE_PREFILL_PACKED_CHUNK_SIZE = 16
STAGES = 2
PREFILL_NUM_DEVICES = 2
DECODE_NUM_DEVICES = 1


def _assert_onnx_path(onnx_path, label: str) -> Path:
    assert onnx_path is not None, f"{label} compile did not set an ONNX path"
    onnx_path = Path(onnx_path)
    assert onnx_path.is_file(), f"{label} ONNX path does not exist: {onnx_path}"
    assert onnx_path.suffix == ".onnx", f"{label} path is not an ONNX file: {onnx_path}"
    return onnx_path.resolve()


def _build_config(dtype: str = "float32"):
    """Load the real config; optionally truncate depth to a shallow sliding/full layer mix.

    When ``NUM_HIDDEN_LAYERS`` is an int, run a shallow model (cheap compile/run) and
    re-derive ``layer_types`` for the reduced depth so gpt-oss still exercises BOTH a
    sliding-window and a full-attention layer (the sliding layer is the interesting case for
    the DMA handoff: retain_full_kv promotes it to full ctx_len). When it is ``None`` the
    checkpoint's own depth, ``layer_types`` and ``sliding_window`` are kept unchanged.
    """
    config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if NUM_HIDDEN_LAYERS is not None:
        config.num_hidden_layers = NUM_HIDDEN_LAYERS
        config.sliding_window = SLIDING_WINDOW
        # Alternate sliding / full so the truncated model still has at least one of each.
        config.layer_types = ["sliding_attention" if i % 2 == 0 else "full_attention" for i in range(NUM_HIDDEN_LAYERS)]
    config.dtype = dtype
    config.torch_dtype = getattr(torch, dtype)
    return config


def _load_hf_model(config) -> AutoModelForCausalLM:
    torch.manual_seed(42)
    if NUM_HIDDEN_LAYERS is None:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            config=config,
            attn_implementation="eager",
            torch_dtype=config.torch_dtype,
            trust_remote_code=True,
        )
        return model.eval()
    model = AutoModelForCausalLM.from_config(config, attn_implementation="eager")
    # Scale weights down so fp32 activations stay small; keeps HF and QAIC numerics close.
    with torch.no_grad():
        for param in model.parameters():
            param.mul_(0.02)
    return model.eval()


def _get_next_token_ids(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits)
    return logits[:, -1, :].argmax(axis=-1).astype(np.int64)


def _assert_tokens_match(label_a: str, tokens_a: np.ndarray, label_b: str, tokens_b: np.ndarray) -> None:
    assert tokens_a.shape == (BATCH_SIZE, GENERATION_LEN), f"{label_a} shape mismatch: {tokens_a.shape}"
    assert tokens_b.shape == (BATCH_SIZE, GENERATION_LEN), f"{label_b} shape mismatch: {tokens_b.shape}"
    assert np.issubdtype(tokens_a.dtype, np.integer), f"{label_a} tokens are not integer dtype"
    assert np.issubdtype(tokens_b.dtype, np.integer), f"{label_b} tokens are not integer dtype"

    matches = tokens_a == tokens_b
    matching_steps = matches.all(axis=0)
    num_matched = int(matching_steps.cumprod().sum())
    mismatches = np.flatnonzero(~matching_steps).tolist()
    print(f"{label_a} tokens : {tokens_a.tolist()}")
    print(f"{label_b} tokens : {tokens_b.tolist()}")
    print(f"Matched leading tokens : {num_matched}/{GENERATION_LEN}")

    if not mismatches:
        print("\nAll tokens agree - no token mismatches detected.")
    else:
        print(f"\nMismatches at steps: {mismatches}")
        first_mismatch = mismatches[0]
        raise AssertionError(
            f"Tokens don't match for {label_a} and {label_b}; "
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


def _prompt_input_ids(tokenizer) -> torch.Tensor:
    return tokenizer(TEXT_PROMPT, return_tensors="pt")["input_ids"]


def _prepare_inputs(tokenizer) -> dict:
    """Tokenize the prompt, right-pad to a multiple of PREFILL_SEQ_LEN, build position_ids."""
    ids = _prompt_input_ids(tokenizer)
    input_len = ids.shape[1]
    num_chunks = -(input_len // -PREFILL_SEQ_LEN)  # ceil divide without float
    padded_len = num_chunks * PREFILL_SEQ_LEN
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    input_ids = np.full((BATCH_SIZE, padded_len), pad_id, dtype=np.int64)
    input_ids[:, :input_len] = ids.numpy()
    attention_mask = np.zeros((BATCH_SIZE, padded_len), dtype=np.int64)
    attention_mask[:, :input_len] = 1
    position_ids = np.where(attention_mask, np.arange(padded_len), -1)
    return {
        "input_ids": input_ids,
        "position_ids": position_ids.astype(np.int64),
        "attention_mask": attention_mask,
        "num_chunks": num_chunks,
        "input_len": input_len,
    }


def _run_hf_torch_fp32(model, tokenizer) -> np.ndarray:
    model = model.to(dtype=torch.float32).eval()
    input_ids = _prompt_input_ids(tokenizer)
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=GENERATION_LEN,
            min_new_tokens=GENERATION_LEN,
            do_sample=False,
        )
    prompt_len = input_ids.shape[-1]
    return outputs[:, prompt_len:].detach().cpu().numpy()


def _patch_custom_rmsnorm_for_ort(path: Path) -> Path:
    """Patch exported local functions so ORT can execute the QPC ONNX in fp32."""
    try:
        import onnx
        from onnx import helper, TensorProto
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("onnx is required for ORT parity test.") from exc

    model = onnx.load(str(path), load_external_data=False)
    changed = False
    int32_max = 2147483647

    def _make_const_i64_1d(name: str, values: list):
        return helper.make_node(
            "Constant",
            [],
            [name],
            value=helper.make_tensor(name, TensorProto.INT64, [len(values)], values),
        )

    all_graph_nodes = list(model.graph.node)

    def _formal_param_for(function, call_arg: str) -> str | None:
        for node in all_graph_nodes:
            if node.op_type == function.name:
                for idx, inp in enumerate(node.input):
                    if inp == call_arg and idx < len(function.input):
                        return function.input[idx]
        for caller in model.functions:
            for node in caller.node:
                if node.op_type == function.name:
                    for idx, inp in enumerate(node.input):
                        if inp == call_arg and idx < len(function.input):
                            return function.input[idx]
        return None

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
                    node.output[0] = "epsilon_2_pre_cast"
                    new_nodes.append(node)
                    new_nodes.append(
                        helper.make_node(
                            "CastLike",
                            ["epsilon_2_pre_cast", "variance"],
                            ["epsilon_2"],
                            name="CastLike_epsilon",
                        )
                    )
                    function_changed = True
                    continue
                new_nodes.append(node)

        elif function.name.startswith("CtxScatter"):
            pos_param = _formal_param_for(function, "position_ids") or next(
                (param for param in function.input if "pos" in param.lower()), None
            )
            if pos_param is not None:
                new_nodes.extend(
                    [
                        helper.make_node("Shape", ["data"], ["_sc_data_shape"]),
                        _make_const_i64_1d("_sc_dim1_idx", [1]),
                        helper.make_node("Gather", ["_sc_data_shape", "_sc_dim1_idx"], ["_sc_ctx_dim"]),
                        _make_const_i64_1d("_sc_one", [1]),
                        helper.make_node("Sub", ["_sc_ctx_dim", "_sc_one"], ["_sc_last_i64"]),
                        helper.make_node("CastLike", ["_sc_last_i64", pos_param], ["_sc_last"]),
                        _make_const_i64_1d("_sc_inv_i64", [int32_max]),
                        helper.make_node("CastLike", ["_sc_inv_i64", pos_param], ["_sc_inv"]),
                        helper.make_node("Equal", [pos_param, "_sc_inv"], ["_sc_inv_mask"]),
                        helper.make_node("Where", ["_sc_inv_mask", "_sc_last", pos_param], ["_sc_safe_pos"]),
                        helper.make_node(
                            "Cast",
                            ["_sc_safe_pos"],
                            ["_sc_pos_i64"],
                            name="Cast_sc_pos_i64",
                            to=TensorProto.INT64,
                        ),
                    ]
                )
                function_changed = True

            for node in function.node:
                if pos_param is not None:
                    for idx, inp in enumerate(node.input):
                        if inp == pos_param:
                            node.input[idx] = "_sc_pos_i64"
                if node.op_type == "ScatterND":
                    indices_in = node.input[1]
                    if not indices_in.endswith("_i64"):
                        cast_out = indices_in + "_i64"
                        new_nodes.append(
                            helper.make_node(
                                "Cast",
                                [indices_in],
                                [cast_out],
                                name=f"Cast_{indices_in}_i64",
                                to=TensorProto.INT64,
                            )
                        )
                        node.input[1] = cast_out
                        function_changed = True
                if node.op_type == "Cast" and list(node.output) == ["batch_idx_3"]:
                    for attr in node.attribute:
                        if attr.name == "to":
                            attr.i = TensorProto.INT64
                    function_changed = True
                if node.op_type == "Expand" and list(node.output) == ["ctx_idx"]:
                    node.output[0] = "ctx_idx_pre_i64"
                    new_nodes.append(node)
                    new_nodes.append(
                        helper.make_node(
                            "Cast",
                            ["ctx_idx_pre_i64"],
                            ["ctx_idx"],
                            name="Cast_ctx_idx_i64",
                            to=TensorProto.INT64,
                        )
                    )
                    function_changed = True
                    continue
                new_nodes.append(node)

        elif function.name.startswith("CtxGather"):
            ctx_indices_param = next((param for param in function.input if "ctx_indices" in param.lower()), None)
            pos_param = next((param for param in function.input if "pos" in param.lower()), None)
            clamp_target = pos_param or ctx_indices_param
            if clamp_target is not None:
                new_nodes.extend(
                    [
                        _make_const_i64_1d("_gc_inv_i64", [int32_max]),
                        helper.make_node("CastLike", ["_gc_inv_i64", clamp_target], ["_gc_inv"]),
                        helper.make_node("Equal", [clamp_target, "_gc_inv"], ["_gc_inv_mask"]),
                        _make_const_i64_1d("_gc_zero_i64", [0]),
                        helper.make_node("CastLike", ["_gc_zero_i64", clamp_target], ["_gc_zero"]),
                        helper.make_node("Where", ["_gc_inv_mask", "_gc_zero", clamp_target], ["_gc_safe_target"]),
                        helper.make_node(
                            "Cast",
                            ["_gc_safe_target"],
                            ["_gc_target_i64"],
                            name="Cast_gc_target_i64",
                            to=TensorProto.INT64,
                        ),
                    ]
                )
                function_changed = True

            inserted_ctx_indices_cast = False
            for node in function.node:
                if clamp_target is not None:
                    for idx, inp in enumerate(node.input):
                        if inp == clamp_target:
                            node.input[idx] = "_gc_target_i64"
                if (
                    not inserted_ctx_indices_cast
                    and ctx_indices_param is not None
                    and ctx_indices_param != clamp_target
                    and node.op_type in {"Expand", "Unsqueeze", "GatherND"}
                    and ctx_indices_param in node.input
                ):
                    cast_name = f"{ctx_indices_param}_i64"
                    new_nodes.append(
                        helper.make_node(
                            "Cast",
                            [ctx_indices_param],
                            [cast_name],
                            name=f"Cast_{ctx_indices_param}_i64",
                            to=TensorProto.INT64,
                        )
                    )
                    inserted_ctx_indices_cast = True
                    function_changed = True
                    for idx, inp in enumerate(node.input):
                        if inp == ctx_indices_param:
                            node.input[idx] = cast_name
                new_nodes.append(node)

        else:
            continue

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
    if dim in {"ctx_len", "context_length", "past_sequence_length", "sliding_window"}:
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
            state[name[: -len("_RetainedState")]] = value


def _run_ort_generation(onnx_paths: dict[str, Path], tokenizer) -> np.ndarray:
    """Run the disaggregated prefill -> decode loop with ORT using QPC ONNX graphs."""
    prefill_session = _make_ort_session(onnx_paths["prefill"])
    decode_session = _make_ort_session(onnx_paths["decode"])

    print(f"[ORT] prefill_inputs : {_session_input_names(prefill_session)}")
    print(f"[ORT] decode_inputs  : {_session_input_names(decode_session)}")

    prepared = _prepare_inputs(tokenizer)
    input_ids = prepared["input_ids"]
    position_ids = prepared["position_ids"]
    num_chunks = prepared["num_chunks"]

    state: dict = {}
    prefill_outputs: dict = {}
    for chunk_idx in range(num_chunks):
        start = chunk_idx * PREFILL_SEQ_LEN
        end = (chunk_idx + 1) * PREFILL_SEQ_LEN
        provided = {
            "input_ids": input_ids[:, start:end],
            "position_ids": position_ids[:, start:end],
        }
        feed = _ensure_session_inputs(prefill_session, provided, state, PREFILL_SEQ_LEN)
        prefill_outputs = dict(zip(_session_output_names(prefill_session), prefill_session.run(None, feed)))
        _update_state_from_outputs(state, prefill_outputs)

    first_token = _get_next_token_ids(prefill_outputs["logits"])
    generated_ids = [first_token]
    pos = np.max(position_ids, axis=-1, keepdims=True) + 1

    decode_inputs = {
        "input_ids": first_token.reshape(BATCH_SIZE, 1),
        "position_ids": pos,
    }
    for _ in range(GENERATION_LEN - 1):
        feed = _ensure_session_inputs(decode_session, decode_inputs, state, 1)
        decode_outputs = dict(zip(_session_output_names(decode_session), decode_session.run(None, feed)))
        _update_state_from_outputs(state, decode_outputs)
        token = _get_next_token_ids(decode_outputs["logits"])
        generated_ids.append(token)
        pos = pos + 1
        decode_inputs = {
            "input_ids": token.reshape(BATCH_SIZE, 1),
            "position_ids": pos,
        }

    return np.stack(generated_ids, axis=1).astype(np.int64)


def _run_disagg_kv_share_qaic_generation(
    tokenizer,
    prefill_session: QAICInferenceSession,
    decode_session: QAICInferenceSession,
) -> np.ndarray:
    prepared = _prepare_inputs(tokenizer)
    num_chunks = prepared["num_chunks"]
    input_ids = prepared["input_ids"]
    position_ids = prepared["position_ids"]
    kv_caches = [np.zeros(shape, dtype=dtype) for (shape, dtype) in decode_session.kv_cache_info]
    chunk_inputs = {}
    exec_idx = None
    for chunk_idx in range(num_chunks):
        chunk_inputs["input_ids"] = input_ids[:, chunk_idx * PREFILL_SEQ_LEN : (chunk_idx + 1) * PREFILL_SEQ_LEN]
        chunk_inputs["position_ids"] = position_ids[:, chunk_idx * PREFILL_SEQ_LEN : (chunk_idx + 1) * PREFILL_SEQ_LEN]
        last_chunk = chunk_idx == num_chunks - 1
        exec_idx = prefill_session.np_run_pipeline(
            chunk_inputs,
            last_chunk=last_chunk,
            kv_cache_buffers=kv_caches if last_chunk else None,
        )
        prefill_session.complete_inf(exec_idx, is_prefill=True)

    prefill_out = prefill_session.get_outputs(index=exec_idx)
    generated_ids = [_get_next_token_ids(prefill_out["logits"])]
    decode_kv_map = decode_session.decode_buff_map + decode_session.decode_rs_kv_only_buff_map
    pos = np.max(position_ids, axis=-1, keepdims=True) + 1
    decode_inputs = {
        "input_ids": generated_ids[-1].reshape(BATCH_SIZE, 1),
        "position_ids": pos,
    }
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
        pos = pos + 1
        decode_inputs = {
            "input_ids": generated_ids[-1].reshape(BATCH_SIZE, 1),
            "position_ids": pos,
        }

    return np.stack(generated_ids, axis=1)


def _compile_disagg_sessions(qeff_model, sessions: list, compiled_onnx_paths: dict):
    decode_qpc_path = qeff_model.compile(
        prefill_seq_len=1,
        ctx_len=CTX_LEN,
        num_cores=NUM_CORES,
        num_devices=DECODE_NUM_DEVICES,
        mos=1,
        aic_enable_depth_first=True,
        num_speculative_tokens=None,
        offload_pt_weights=False,
        split_retained_state_io=True,
        retain_full_kv=True,
        use_onnx_subfunctions=True,
    )
    compiled_onnx_paths["decode"] = _assert_onnx_path(qeff_model.onnx_path, "decode")

    prefill_qpc_path = qeff_model.compile(
        prefill_seq_len=PREFILL_SEQ_LEN,
        ctx_len=CTX_LEN,
        num_cores=NUM_CORES,
        moe_prefill_packed_chunk_size=MOE_PREFILL_PACKED_CHUNK_SIZE,
        num_devices=PREFILL_NUM_DEVICES,
        mdp_num_partitions=STAGES,
        split_retained_state_io=True,
        mos=1,
        aic_enable_depth_first=False,
        num_speculative_tokens=None,
        prefill_only=True,
        enable_chunking=True,
        retain_full_kv=True,
        use_onnx_subfunctions=True,
    )
    compiled_onnx_paths["prefill"] = _assert_onnx_path(qeff_model.onnx_path, "prefill")
    print(f"Disagg ONNX paths: {compiled_onnx_paths}")

    prefill_session = QAICInferenceSession(prefill_qpc_path, kv_dma_share=True, stages=STAGES)
    decode_session = QAICInferenceSession(decode_qpc_path, kv_dma_share=True)
    sessions.extend([prefill_session, decode_session])
    return prefill_session, decode_session


def _prepare_hf_model_tokenizer():
    torch.manual_seed(42)
    config = _build_config(dtype="float32")
    hf_model = _load_hf_model(config)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return hf_model, tokenizer



@pytest.mark.on_qaic
@pytest.mark.disagg_dma
@pytest.mark.nightly_disagg
def test_gpt_oss_disagg_kv_share_qaic_vs_ort_vs_hf_fp32(manual_cleanup):
    """Three-way parity using the same ONNX graphs that QEff compiled into QPCs."""
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")

    hf_model, tokenizer = _prepare_hf_model_tokenizer()
    hf_tokens = _run_hf_torch_fp32(hf_model, tokenizer)
    qeff_model = QEFFAutoModelForCausalLM(hf_model)

    sessions = []
    compiled_onnx_paths = {}
    try:
        prefill_session, decode_session = _compile_disagg_sessions(qeff_model, sessions, compiled_onnx_paths)

        ort_tokens = _run_ort_generation(compiled_onnx_paths, tokenizer)
        qaic_tokens = _run_disagg_kv_share_qaic_generation(
            tokenizer=tokenizer,
            prefill_session=prefill_session,
            decode_session=decode_session,
        )
    finally:
        for session in sessions:
            session.deactivate()


    hf_text = tokenizer.batch_decode(hf_tokens, skip_special_tokens=True)
    ort_text = tokenizer.batch_decode(ort_tokens, skip_special_tokens=True)
    qaic_text = tokenizer.batch_decode(qaic_tokens, skip_special_tokens=True)
    print(f"HF   text : {hf_text}")
    print(f"ORT  text : {ort_text}")
    print(f"QAIC text : {qaic_text}")

    _assert_three_way_tokens_match(hf_tokens, ort_tokens, qaic_tokens)
