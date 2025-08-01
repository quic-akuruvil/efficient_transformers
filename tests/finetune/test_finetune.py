# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os
import shutil

import numpy as np
import pytest
import requests
import torch.optim as optim
from torch.utils.data import DataLoader

import QEfficient
import QEfficient.cloud.finetune
from QEfficient.cloud.finetune import main as finetune
from QEfficient.finetune.utils.helper import Device, Task_Mode

alpaca_json_path = os.path.join(os.getcwd(), "alpaca_data.json")


def clean_up(path):
    if os.path.isdir(path) and os.path.exists(path):
        shutil.rmtree(path)
    if os.path.isfile(path):
        os.remove(path)


def download_alpaca():
    alpaca_url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/refs/heads/main/alpaca_data.json"
    response = requests.get(alpaca_url)

    with open(alpaca_json_path, "wb") as f:
        f.write(response.content)


configs = [
    pytest.param(
        "meta-llama/Llama-3.2-1B",  # model_name
        Task_Mode.GENERATION,  # task_mode
        10,  # max_eval_step
        20,  # max_train_step
        "gsm8k_dataset",  # dataset_name
        None,  # data_path
        10,  # intermediate_step_save
        None,  # context_length
        True,  # run_validation
        True,  # use_peft
        Device.QAIC,  # device
        1.5416,  # expected_train_loss
        4.6722,  # expected_train_metric
        1.4040,  # expected_eval_loss
        4.0715,  # expected_eval_metric
        id="llama_config_gsm8k",  # config name
    ),
    pytest.param(
        "meta-llama/Llama-3.2-1B",  # model_name
        Task_Mode.GENERATION,  # task_mode
        10,  # max_eval_step
        20,  # max_train_step
        "alpaca_dataset",  # dataset_name
        alpaca_json_path,  # data_path
        10,  # intermediate_step_save
        None,  # context_length
        True,  # run_validation
        True,  # use_peft
        Device.QAIC,  # device
        1.3477,  # expected_train_loss
        3.8486,  # expected_train_metric
        1.5305,  # expected_eval_loss
        4.6207,  # expected_eval_metric
        id="llama_config_alpaca",  # config name
    ),
    pytest.param(
        "google-bert/bert-base-uncased",  # model_name
        Task_Mode.SEQ_CLASSIFICATION,  # task_mode
        10,  # max_eval_step
        20,  # max_train_step
        "imdb_dataset",  # dataset_name
        None,  # data_path
        10,  # intermediate_step_save
        None,  # context_length
        True,  # run_validation
        False,  # use_peft
        Device.QAIC,  # device
        0.6566,  # expected_train_loss
        0.5055,  # expected_train_metric
        0.6188,  # expected_eval_loss
        0.6904,  # expected_eval_metric
        id="bert_config_imdb",  # config name
    ),
]


@pytest.mark.skip()  # remove when it's clear why diff val_step_loss values are observed in diff runs on existing code (even without PR #478 changes)
@pytest.mark.cli
@pytest.mark.on_qaic
@pytest.mark.finetune
@pytest.mark.parametrize(
    "model_name,task_mode,max_eval_step,max_train_step,dataset_name,data_path,intermediate_step_save,context_length,run_validation,use_peft,device,expected_train_loss,expected_train_metric,expected_eval_loss,expected_eval_metric",
    configs,
)
def test_finetune(
    model_name,
    task_mode,
    max_eval_step,
    max_train_step,
    dataset_name,
    data_path,
    intermediate_step_save,
    context_length,
    run_validation,
    use_peft,
    device,
    expected_train_loss,
    expected_train_metric,
    expected_eval_loss,
    expected_eval_metric,
    mocker,
):
    train_config_spy = mocker.spy(QEfficient.cloud.finetune, "TrainConfig")
    generate_dataset_config_spy = mocker.spy(QEfficient.cloud.finetune, "generate_dataset_config")
    generate_peft_config_spy = mocker.spy(QEfficient.cloud.finetune, "generate_peft_config")
    get_dataloader_kwargs_spy = mocker.spy(QEfficient.finetune.utils.dataset_utils, "get_dataloader_kwargs")
    update_config_spy = mocker.spy(QEfficient.cloud.finetune, "update_config")
    get_custom_data_collator_spy = mocker.spy(QEfficient.finetune.utils.dataset_utils, "get_custom_data_collator")
    get_preprocessed_dataset_spy = mocker.spy(QEfficient.finetune.utils.dataset_utils, "get_preprocessed_dataset")
    get_longest_seq_length_spy = mocker.spy(QEfficient.cloud.finetune, "get_longest_seq_length")
    print_model_size_spy = mocker.spy(QEfficient.cloud.finetune, "print_model_size")
    train_spy = mocker.spy(QEfficient.cloud.finetune, "train")

    kwargs = {
        "model_name": model_name,
        "task_mode": task_mode,
        "max_eval_step": max_eval_step,
        "max_train_step": max_train_step,
        "dataset": dataset_name,
        "data_path": data_path,
        "intermediate_step_save": intermediate_step_save,
        "context_length": context_length,
        "run_validation": run_validation,
        "use_peft": use_peft,
        "device": device,
    }

    if dataset_name == "alpaca_dataset":
        download_alpaca()

    results = finetune(**kwargs)

    assert np.allclose(results["last_epoch_train_loss"], expected_train_loss, atol=1e-3), "Train loss is not matching."
    assert np.allclose(results["last_epoch_train_metric"], expected_train_metric, atol=1e-3), (
        "Train metric is not matching."
    )
    assert np.allclose(results["last_epoch_eval_loss"], expected_eval_loss, atol=1e-3), "Eval loss is not matching."
    assert np.allclose(results["last_epoch_eval_metric"], expected_eval_metric, atol=1e-3), (
        "Eval metric is not matching."
    )
    assert results["avg_epoch_time"] < 60, "Training should complete within 60 seconds."

    train_config_spy.assert_called_once()
    generate_dataset_config_spy.assert_called_once()
    if task_mode == Task_Mode.GENERATION:
        generate_peft_config_spy.assert_called_once()
    get_longest_seq_length_spy.assert_called_once()
    print_model_size_spy.assert_called_once()
    train_spy.assert_called_once()

    assert update_config_spy.call_count == 2
    assert get_custom_data_collator_spy.call_count == 2
    assert get_dataloader_kwargs_spy.call_count == 2
    assert get_preprocessed_dataset_spy.call_count == 2

    args, kwargs = train_spy.call_args
    train_dataloader = args[2]
    eval_dataloader = args[3]
    optimizer = args[4]

    batch = next(iter(train_dataloader))
    assert "labels" in batch.keys()
    assert "input_ids" in batch.keys()
    assert "attention_mask" in batch.keys()

    assert isinstance(optimizer, optim.AdamW)

    assert isinstance(train_dataloader, DataLoader)
    if run_validation:
        assert isinstance(eval_dataloader, DataLoader)
    else:
        assert eval_dataloader is None

    args, kwargs = update_config_spy.call_args_list[0]
    train_config = args[0]
    assert max_train_step >= train_config.gradient_accumulation_steps, (
        "Total training step should be more than "
        f"{train_config.gradient_accumulation_steps} which is gradient accumulation steps."
    )

    if use_peft:
        saved_file = os.path.join(train_config.output_dir, "complete_epoch_1/adapter_model.safetensors")
    else:
        saved_file = os.path.join(train_config.output_dir, "complete_epoch_1/model.safetensors")
    assert os.path.isfile(saved_file)

    clean_up(train_config.output_dir)
    clean_up("qaic-dumps")

    if dataset_name == "alpaca_dataset":
        clean_up(alpaca_json_path)
