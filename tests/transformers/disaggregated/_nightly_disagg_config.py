# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Compatibility loader for nightly disaggregated parity tests.

Nightly and regular disaggregated test configurations share test_config.json as
the single source of truth.
"""

from tests.transformers.disaggregated._disagg_dma_config import disagg_dma_configs


def nightly_disagg_configs(model_key: str) -> list:
    return disagg_dma_configs(model_key)
