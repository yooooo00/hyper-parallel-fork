# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Save physical operands so autograd hooks do not need quantized-wrapper dispatch."""

from typing import Optional

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.components.quantization.tensor.quantized_tensor import QuantizedTensor


# Saved layout: [group_list, *input_buffers, *weight_buffers].
# Each operand reserves row_data, row_scale, col_data, col_scale in that order,
# including None placeholders, so missing operands do not shift later slots.
_GROUP_LIST_SLOTS = 1
_BUFFERS_PER_OPERAND = 4


def save_quantized_operands(
    ctx: torch.autograd.function.FunctionCtx,
    inputs: Optional[QuantizedTensor],
    weight: Optional[QuantizedTensor],
    group_list: Optional[torch.Tensor] = None,
) -> None:
    """Save only physical tensors; keep reconstruction metadata outside hooks.

    Args:
        ctx: Context for one Dense or grouped projection.
        inputs: Input operand needed for the weight gradient, or None.
        weight: Weight operand needed for the input gradient, or None.
        group_list: Group boundaries or counts for grouped projections.
    """
    metadata, tensors = pack_quantized_operands((inputs, weight))
    ctx.quantized_metadata = metadata
    ctx.save_for_backward(group_list, *tensors)


def restore_quantized_operands(
    ctx: torch.autograd.function.FunctionCtx,
) -> tuple[Optional[torch.Tensor], Optional[QuantizedTensor], Optional[QuantizedTensor]]:
    """Rebuild wrappers from tensors restored by autograd, without requantization.

    Args:
        ctx: Context populated by save_quantized_operands.

    Returns:
        Grouping tensor, input operand and weight operand, in that order.
    """
    tensors = ctx.saved_tensors
    operands = unpack_quantized_operands(ctx.quantized_metadata, tensors[_GROUP_LIST_SLOTS:])
    return tensors[0], operands[0], operands[1]


def pack_quantized_operands(operands: tuple) -> tuple[tuple, list[Optional[torch.Tensor]]]:
    """Separate any number of quantized operands into metadata and physical buffers.

    Args:
        operands: Quantized wrappers or None slots in a stable operand order.

    Returns:
        Reconstruction metadata and four physical tensor slots per operand.
    """
    metadata, tensors = [], []
    for operand in operands:
        if operand is None:
            metadata.append(None)
            tensors.extend((None,) * _BUFFERS_PER_OPERAND)
        else:
            metadata.append((type(operand), tuple(operand.shape), operand.dtype, operand.quantizer))
            tensors.extend((operand.row_data, operand.row_scale, operand.col_data, operand.col_scale))
    return tuple(metadata), tensors


def unpack_quantized_operands(metadata: tuple, tensors: tuple) -> tuple:
    """Rebuild wrappers from restored physical tensors without consuming their storage.

    Args:
        metadata: Reconstruction metadata produced by pack_quantized_operands.
        tensors: Restored row data/scale and column data/scale slots.

    Returns:
        Fresh wrappers or None, in the original operand order.
    """
    operands = []
    for index, entry in enumerate(metadata):
        if entry is None:
            operands.append(None)
            continue
        tensor_type, shape, dtype, quantizer = entry
        offset = _BUFFERS_PER_OPERAND * index
        row_data, row_scale, col_data, col_scale = tensors[offset:offset + _BUFFERS_PER_OPERAND]
        operands.append(tensor_type(
            shape, dtype, quantizer=quantizer,
            row_data=row_data, row_scale=row_scale, col_data=col_data, col_scale=col_scale,
        ))
    return tuple(operands)
