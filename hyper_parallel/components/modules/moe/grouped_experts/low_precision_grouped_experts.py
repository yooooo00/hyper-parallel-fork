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
"""Low-precision grouped-expert modules, MXFP8 fusion, and QAT family registration."""

from __future__ import annotations

import math
from collections.abc import Mapping
from functools import partial
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers.activations import ACT2FN
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Experts

from hyper_parallel.components.checkpoint.weight_conversion import Transpose, WeightConverter, WeightRenaming
from hyper_parallel.components.modules.module_family import (
    ModuleFamily,
    SelectorSpec,
    VariantRule,
    register_module_family,
)
from hyper_parallel.components.quantization.config import (
    GROUPED_NDIM,
    GROUP_LIST_COUNTS,
    LowPrecisionConfig,
    LowPrecisionDtypeScheme,
    MATRIX_NDIM,
    MX_BLOCK_SIZE_32,
)
from hyper_parallel.components.quantization.functional import build_low_precision_strategy
from hyper_parallel.components.quantization.functional.base_gmm_func import (
    GroupedLinear,
    _GroupedLinearFunction,
)
from hyper_parallel.components.quantization.functional.fake_w4a8_gmm_func import FakeW4A8GroupedLinear
from hyper_parallel.components.quantization.functional.hifloat8_gmm_func import HiFloat8GroupedLinear
from hyper_parallel.components.quantization.functional.mxfp8_fused_grouped_swiglu import (
    fused_grouped_swiglu,
)
from hyper_parallel.components.quantization.functional.mxfp8_projection import mxfp8_projection
from hyper_parallel.components.quantization.functional.strategy_factory import _validate_tile_alignment
from hyper_parallel.components.quantization.ops.npu_mxfp8 import (
    LowPrecisionCapabilityError,
    validate_npu_gmm_runtime,
    validate_npu_swiglu_runtime,
)
from hyper_parallel.components.quantization.ops.npu_w4a8 import (
    W4A8CapabilityError,
)
from hyper_parallel.components.quantization.quantizers.mxfp8 import MXFP8Quantizer
from hyper_parallel.models.replacement import module_replacement

_EXPERT_PARAMETER_NAMES = ("gate_up_proj", "down_proj")
_GATE_UP_BRANCHES = 2


def grouped_matmul(*args: Any, **kwargs: Any) -> Tensor:
    """Load the optional NPU baseline kernel only when it is executed.

    Args:
        *args: Existing grouped-matmul positional arguments.
        **kwargs: Existing grouped-matmul keyword arguments.

    Returns:
        The grouped projection produced by the NPU kernel.
    """
    # Canonical policy imports and CPU strategy tests must not require torch_npu.
    # pylint: disable-next=import-outside-toplevel
    from hyper_parallel.components.functional.grouped_matmul import grouped_matmul as operator
    return operator(*args, **kwargs)


def npu_swiglu(*args: Any, **kwargs: Any) -> Tensor:
    """Load the optional fused activation only for baseline computation.

    Args:
        *args: Existing fused SwiGLU positional arguments.
        **kwargs: Existing fused SwiGLU keyword arguments.

    Returns:
        The fused gate/up activation.
    """
    # The low-precision strategy path uses its existing native Torch activation.
    # pylint: disable-next=import-outside-toplevel
    from hyper_parallel.components.functional.swiglu import swiglu
    return swiglu(*args, **kwargs)


def moe_token_permute(tokens: Tensor, indices: Tensor) -> tuple[Tensor, Tensor]:
    """Load the NPU router only when transposed-layout Family routing is used.

    Args:
        tokens: Token-major input activations.
        indices: Selected expert indices for each token and route.

    Returns:
        Expert-major activations and the inverse permutation metadata.
    """
    # The canonical compatibility path retains its hardware-independent routing.
    # pylint: disable-next=import-outside-toplevel
    from hyper_parallel.components.functional.moe_token_permute import moe_token_permute as operator
    return operator(tokens, indices)


def moe_token_unpermute(values: Tensor, indices: Tensor, probs: Tensor | None = None) -> Tensor:
    """Load the NPU inverse router at the Family execution boundary.

    Args:
        values: Expert-major output activations.
        indices: Inverse permutation metadata from the router.
        probs: Optional per-route weights for output accumulation.

    Returns:
        Restored token-major activations.
    """
    # Keep the optional dependency out of the public quantization import path.
    # pylint: disable-next=import-outside-toplevel
    from hyper_parallel.components.functional.moe_token_unpermute import moe_token_unpermute as operator
    return operator(values, indices, probs)


def _swiglu(projected: Tensor, *, limit: float | None = None) -> Tensor:
    """Preserve projection dtype and optionally apply the native DeepSeek V4/V4.1 clamp."""
    gate, up = projected.chunk(2, dim=-1)
    if limit is not None:
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    return F.silu(gate) * up


@module_replacement
class GroupedExperts(nn.Module):
    """Transformers-compatible NPU grouped routed experts.

    Holds gate_up_proj/up_proj, down_proj (and optional bias1, bias2) as nn.Parameters.
    Computation logic is handled by _grouped_gemm_expert_forward.

    Args:
        module: Source Transformers experts module.
        module_fqn: Fully qualified name supplied by the replacement framework.
        context: Additional replacement context.
    """

    @staticmethod
    def _source_parameters(module: nn.Module, source_fc1_name: str) -> tuple[nn.Parameter, nn.Parameter]:
        """Validate and return source expert weight parameters."""
        missing = [name for name in (source_fc1_name, "down_proj") if not hasattr(module, name)]
        if missing:
            raise TypeError(f"GroupedExperts source module is missing: {missing}")
        source_gate_up = getattr(module, source_fc1_name)
        if not isinstance(source_gate_up, nn.Parameter):
            raise TypeError(f"GroupedExperts requires {source_fc1_name} to be an nn.Parameter")
        if not isinstance(module.down_proj, nn.Parameter):
            raise TypeError("GroupedExperts requires down_proj to be an nn.Parameter")
        return source_gate_up, module.down_proj

    @staticmethod
    def _source_dimensions(
        module: nn.Module,
        config: Any,
        source_gate_up: nn.Parameter,
        source_down: nn.Parameter,
        gated_linear_unit: bool,
    ) -> tuple[int, int, int]:
        """Infer expert count and projection dimensions from the source module."""
        num_experts = getattr(module, "num_local_experts", getattr(module, "num_experts", None))
        if num_experts is None and config is not None:
            num_experts = getattr(config, "n_routed_experts", getattr(config, "num_experts", None))
        hidden_size = getattr(module, "hidden_size", getattr(module, "hidden_dim", None))
        if hidden_size is None and config is not None:
            hidden_size = getattr(config, "hidden_size", None)
        intermediate_size = getattr(module, "intermediate_size", getattr(module, "intermediate_dim", None))
        if intermediate_size is None and config is not None:
            intermediate_size = getattr(config, "moe_intermediate_size", getattr(config, "intermediate_size", None))
        source_is_transposed = getattr(module, "is_transposed", None)
        if not gated_linear_unit and source_gate_up.dim() == 3:
            hidden_size = GroupedExperts._infer_ungated_hidden_size(
                source_gate_up, source_down, source_is_transposed, intermediate_size, hidden_size
            )
        if num_experts is None or hidden_size is None or intermediate_size is None:
            raise ValueError("GroupedExperts requires expert count, hidden size, and intermediate size")
        return num_experts, hidden_size, intermediate_size

    @staticmethod
    def _infer_ungated_hidden_size(
        source_gate_up: nn.Parameter,
        source_down: nn.Parameter,
        source_is_transposed: bool | None,
        intermediate_size: int | None,
        default: int | None,
    ) -> int | None:
        """Infer hidden size for ungated three-dimensional source weights."""
        if source_is_transposed is False:
            return source_gate_up.shape[-1]
        if source_is_transposed is True:
            return source_gate_up.shape[-2]
        if intermediate_size is None:
            return default
        if source_gate_up.shape[-2] == intermediate_size and source_down.shape[-1] == intermediate_size:
            return source_gate_up.shape[-1]
        if source_gate_up.shape[-1] == intermediate_size and source_down.shape[-2] == intermediate_size:
            return source_gate_up.shape[-2]
        return default

    @staticmethod
    def _activation(module: nn.Module, config: Any, gated_linear_unit: bool) -> Any:
        """Resolve the source-compatible expert activation."""
        hidden_act = getattr(config, "hidden_act", None)
        if hidden_act is None:
            hidden_act = getattr(config, "hidden_activation", None)
        if hidden_act is None:
            hidden_act = getattr(config, "mlp_hidden_act", None)
        activation_func = getattr(module, "act_fn", None)
        if activation_func is None:
            if hidden_act is None:
                hidden_act = "silu"
            activation_func = ACT2FN[hidden_act]
        source_apply_gate = getattr(module, "_apply_gate", None)
        source_apply_gate_func = getattr(source_apply_gate, "__func__", source_apply_gate)
        if (
            gated_linear_unit
            and source_apply_gate_func is not None
            and getattr(source_apply_gate_func, "__name__", "") != "_default_apply_gate"
        ):
            raise ValueError("GroupedExperts custom _apply_gate requires a dedicated replacement")
        if not gated_linear_unit:
            return activation_func
        if hidden_act == "silu" and bool(getattr(config, "use_fused_swiglu", True)):
            return partial(npu_swiglu, dim=-1)

        def glu(x: torch.Tensor) -> torch.Tensor:
            """Apply the configured gated linear unit."""
            gate, value = torch.chunk(x, 2, dim=-1)
            return activation_func(gate) * value

        return glu

    @staticmethod
    def _source_layout(
        source_shapes: tuple[tuple[int, ...], tuple[int, ...]],
        source_is_transposed: bool | None,
        expert_shapes: tuple[tuple[int, ...], tuple[int, ...]],
        transformers_shapes: tuple[tuple[int, ...], tuple[int, ...]],
        npu_shapes: tuple[tuple[int, ...], tuple[int, ...]],
    ) -> tuple[bool, bool]:
        """Return whether experts are two-dimensional and need transposition."""
        if source_shapes == expert_shapes:
            return True, False
        if source_is_transposed is False:
            if source_shapes != transformers_shapes:
                raise ValueError("GroupedExperts source weights do not match their declared layout")
            return False, True
        if source_is_transposed is True:
            if source_shapes != npu_shapes:
                raise ValueError("GroupedExperts source weights do not match their declared layout")
            return False, False
        if source_shapes == transformers_shapes:
            return False, True
        if source_shapes == npu_shapes:
            return False, False
        raise ValueError("GroupedExperts source weight shapes are incompatible with the configured dimensions")

    def _initialize_weights(
        self,
        source_gate_up: nn.Parameter,
        source_down: nn.Parameter,
        npu_gate_up_shape: tuple[int, ...],
        npu_down_shape: tuple[int, ...],
        gated_linear_unit: bool,
    ) -> None:
        """Create or reuse expert parameters in the target NPU layout."""
        if self._transpose_source_weights:
            target_gate_up = nn.Parameter(
                torch.empty(npu_gate_up_shape, device=source_gate_up.device, dtype=source_gate_up.dtype),
                requires_grad=source_gate_up.requires_grad,
            )
            self.down_proj = nn.Parameter(
                torch.empty(npu_down_shape, device=source_down.device, dtype=source_down.dtype),
                requires_grad=source_down.requires_grad,
            )
            # Meta weights are populated by the existing checkpoint/init pipeline.
            # Eager replacement must preserve values already present in the source.
            if not source_gate_up.is_meta:
                with torch.no_grad():
                    target_gate_up.copy_(source_gate_up.transpose(-2, -1))
                    self.down_proj.copy_(source_down.transpose(-2, -1))
        else:
            target_gate_up = source_gate_up
            self.down_proj = source_down
        if gated_linear_unit:
            self.gate_up_proj = target_gate_up
        else:
            self.up_proj = target_gate_up

    def _initialize_biases(self, module: nn.Module) -> None:
        """Validate and materialize optional expert bias parameters."""
        self._source_bias_names = (
            "bias1" if hasattr(module, "bias1") else f"{self._source_fc1_name}_bias",
            "bias2" if hasattr(module, "bias2") else "down_proj_bias",
        )
        source_bias1 = getattr(module, self._source_bias_names[0], None)
        source_bias2 = getattr(module, self._source_bias_names[1], None)
        if (source_bias1 is None) != (source_bias2 is None):
            raise ValueError("GroupedExperts source projections must use the same bias policy")
        self.add_bias = source_bias1 is not None
        if not self.add_bias:
            return
        if not isinstance(source_bias1, nn.Parameter) or not isinstance(source_bias2, nn.Parameter):
            raise TypeError("GroupedExperts source biases must be nn.Parameter instances")
        target_biases = []
        for source_name, target_name, source_bias in zip(
            self._source_bias_names, ("bias1", "bias2"), (source_bias1, source_bias2)
        ):
            if source_name == target_name:
                target_biases.append(source_bias)
            else:
                target_biases.append(
                    nn.Parameter(torch.empty_like(source_bias), requires_grad=source_bias.requires_grad)
                )
        # astroid cannot infer the length of a list built by appends in a
        # loop; the zip above always yields exactly two entries.
        self.bias1, self.bias2 = target_biases  # pylint: disable=W0632

    def __init__(
        self,
        *,
        module: nn.Module,
        module_fqn: str = "",
        context: Mapping[str, Any] | None = None,
    ) -> None:
        """Build the high-performance module from a Transformers experts module."""
        super().__init__()
        del module_fqn, context
        config = getattr(module, "config", None)
        self.config = config
        self.initializer_range = getattr(config, "initializer_range", None)
        if bool(getattr(module, "has_bias", False)):
            raise ValueError("GroupedExperts does not support source modules with has_bias=True")
        gated_linear_unit = bool(
            getattr(
                module,
                "has_gate",
                getattr(config, "gated_linear_unit", True),
            )
        )
        self._source_fc1_name = "gate_up_proj" if gated_linear_unit else "up_proj"
        source_gate_up, source_down = self._source_parameters(module, self._source_fc1_name)
        num_experts, hidden_size, intermediate_size = self._source_dimensions(
            module, config, source_gate_up, source_down, gated_linear_unit
        )
        source_down = module.down_proj
        source_is_transposed = getattr(module, "is_transposed", None)

        self.num_local_experts = num_experts
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.router_gating_in_fp32 = bool(
            getattr(module, "router_gating_in_fp32", True)
        )
        self.activation_func = self._activation(module, config, gated_linear_unit)

        # GroupGemm state (NPU-only)
        self._group_list = None
        self._tokens_per_expert_gmm = None

        self._initialize_projection_layout(
            source_gate_up, source_down, source_is_transposed, gated_linear_unit
        )
        self._initialize_biases(module)

        self.has_gate = gated_linear_unit
        self.has_bias = self.add_bias
        self.is_concatenated = bool(getattr(module, "is_concatenated", True))
        if self.has_gate and not self.is_concatenated:
            raise ValueError("GroupedExperts requires concatenated gate/up expert weights")
        self.is_transposed = True
        # Keep HF native initialization order stable; this replacement module
        # resets its packed expert parameters after materialization instead.
        self._hp_reset_after_materialization = True
        self.train(module.training)

    def _initialize_projection_layout(
        self, source_gate_up, source_down, source_is_transposed, gated_linear_unit,
    ) -> None:
        """Resolve expert weight layouts and initialize their target parameters."""
        hidden_size = self.hidden_size
        intermediate_size = self.intermediate_size
        fc1_output_size = intermediate_size
        if gated_linear_unit:
            fc1_output_size *= 2

        npu_gate_up_shape = (
            self.num_local_experts,
            hidden_size,
            fc1_output_size,
        )
        npu_down_shape = (
            self.num_local_experts,
            intermediate_size,
            hidden_size,
        )
        transformers_gate_up_shape = (
            self.num_local_experts,
            fc1_output_size,
            hidden_size,
        )
        transformers_down_shape = (
            self.num_local_experts,
            hidden_size,
            intermediate_size,
        )
        experts_2d_gate_up_shape = (
            self.num_local_experts * hidden_size,
            fc1_output_size,
        )
        experts_2d_down_shape = (
            self.num_local_experts * intermediate_size,
            hidden_size,
        )
        source_shapes = (tuple(source_gate_up.shape), tuple(source_down.shape))
        self.use_2d_experts, self._transpose_source_weights = self._source_layout(
            source_shapes,
            source_is_transposed,
            (experts_2d_gate_up_shape, experts_2d_down_shape),
            (transformers_gate_up_shape, transformers_down_shape),
            (npu_gate_up_shape, npu_down_shape),
        )
        self._initialize_weights(
            source_gate_up, source_down, npu_gate_up_shape, npu_down_shape, gated_linear_unit
        )

    def reset_parameters(self) -> None:
        """Initialize grouped weights with the source model's configured standard deviation."""
        if self.initializer_range is None:
            raise ValueError(
                "GroupedExperts random initialization requires config.initializer_range"
            )
        target_fc1 = self.gate_up_proj if self.has_gate else self.up_proj
        nn.init.normal_(target_fc1, mean=0.0, std=self.initializer_range)
        nn.init.normal_(self.down_proj, mean=0.0, std=self.initializer_range)
        if self.add_bias:
            nn.init.zeros_(self.bias1)
            nn.init.zeros_(self.bias2)

    def make_transforms(self) -> list[WeightRenaming | WeightConverter]:
        """Describe reversible source-checkpoint to high-performance conversion."""
        transforms: list[WeightRenaming | WeightConverter] = []
        if self._transpose_source_weights:
            transforms.extend(
                [
                    WeightConverter(
                        source_patterns=self._source_fc1_name,
                        target_patterns=self._source_fc1_name,
                        operations=[Transpose(dim0=-2, dim1=-1)],
                    ),
                    WeightConverter(
                        source_patterns="down_proj",
                        target_patterns="down_proj",
                        operations=[Transpose(dim0=-2, dim1=-1)],
                    ),
                ]
            )
        if self.add_bias:
            for source_name, target_name in zip(
                self._source_bias_names,
                ("bias1", "bias2"),
            ):
                if source_name == target_name:
                    continue
                transforms.append(
                    WeightRenaming(
                        source_patterns=source_name,
                        target_patterns=target_name,
                    )
                )
        return transforms

    def _grouped_gemm_expert_forward(
        self, gate_up_proj, down_proj, permuted, tokens_per_expert, permuted_probs
    ):
        """GroupGemm expert computation (NPU-only)."""
        self._tokens_per_expert_gmm = tokens_per_expert.to(device=permuted.device)
        self._group_list = self._tokens_per_expert_gmm.cumsum(dim=0)

        if self.use_2d_experts:
            gate_up_proj = gate_up_proj.view(self.num_local_experts, self.hidden_size, -1)

        # up
        numel = permuted.nelement()
        if torch.compiler.is_compiling() or not isinstance(numel, int) or numel != 0:
            fc1_output = grouped_matmul(  # pylint: disable=not-callable
                permuted, gate_up_proj, bias=None, group_list=self._group_list, group_type=0, group_list_type=0,
            )
            if self.add_bias:
                b1 = self.bias1.view(self.num_local_experts, -1)
                fc1_output = fc1_output + torch.repeat_interleave(b1, self._tokens_per_expert_gmm, dim=0)
        else:
            gate_up_proj_2d = gate_up_proj.view(self.hidden_size, -1)
            fc1_output = torch.matmul(permuted, gate_up_proj_2d)

        # down
        if not self.router_gating_in_fp32 and permuted_probs is not None:
            fc1_output = (
                self.activation_func(fc1_output)
                * permuted_probs.reshape(*fc1_output.shape[:-1], 1)
            )
        else:
            fc1_output = self.activation_func(fc1_output)

        if self.use_2d_experts:
            down_proj = down_proj.view(self.num_local_experts, -1, self.hidden_size)

        numel = fc1_output.nelement()
        if torch.compiler.is_compiling() or not isinstance(numel, int) or numel != 0:
            fc2_output = grouped_matmul(  # pylint: disable=not-callable
                fc1_output, down_proj, bias=None, group_list=self._group_list, group_type=0, group_list_type=0,
            )
            if self.add_bias:
                b2 = self.bias2.view(self.num_local_experts, -1)
                fc2_output = fc2_output + torch.repeat_interleave(
                    b2, self._tokens_per_expert_gmm, dim=0,
                )
        else:
            down_proj_2d = down_proj.view(-1, self.hidden_size)
            fc2_output = torch.matmul(fc1_output, down_proj_2d)

        return fc2_output

    def forward_expert_major(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
        scores: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run grouped experts on expert-major routed tokens.

        Args:
            x: Expert-major input with shape ``[tokens, hidden_size]``.
            num_tokens_per_expert: Token count for every local expert.
            scores: Optional routing weight for every routed token.

        Returns:
            Expert-major output with shape ``[tokens, hidden_size]``.
        """
        gate_up_proj = self.gate_up_proj if self.has_gate else self.up_proj
        down_proj = self.down_proj
        self.num_local_experts = num_tokens_per_expert.shape[0]

        return self._grouped_gemm_expert_forward(
            gate_up_proj, down_proj, x, num_tokens_per_expert, scores
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Run grouped experts with the Transformers Experts interface."""
        hidden_shape = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_states.shape[-1])
        permuted_tokens, sorted_indices = moe_token_permute(  # pylint: disable=not-callable
            hidden_states_flat,
            top_k_index,
        )
        flatten_probs = top_k_weights.view(-1)
        permuted_probs = flatten_probs.index_select(0, sorted_indices)
        tokens_per_expert = torch.histc(
            top_k_index.view(-1).float(),
            bins=self.num_experts,
            min=0,
            max=self.num_experts,
        ).long()
        expert_outputs = self.forward_expert_major(
            permuted_tokens,
            tokens_per_expert,
            permuted_probs,
        )
        output = moe_token_unpermute(  # pylint: disable=not-callable
            expert_outputs,
            sorted_indices,
            top_k_weights,
        )
        return output.view(hidden_shape)


class LowPrecisionGroupedExperts(GroupedExperts):
    """One strategy-backed implementation for Family and canonical policy entry points.

    Family construction retains the baseline's transposed weights and checkpoint
    transforms. ``from_module`` and dimension-based construction retain canonical
    ``[E, O, K]`` weights and the original policy routing/registration contract.
    Both layouts use the same grouped-linear computation and autograd strategy.
    """

    def __init__(
        self, num_experts: int | None = None, hidden_dim: int | None = None,
        intermediate_dim: int | None = None, *, grouped_linear: GroupedLinear | None = None,
        fqn: str = "", module: nn.Module | None = None, strategy: GroupedLinear | None = None,
        module_fqn: str = "", context: Mapping[str, Any] | None = None,
    ) -> None:
        """Construct from a Family source or the compatibility projection dimensions.

        Args:
            num_experts: Expert count for canonical dimension-based construction.
            hidden_dim: Model width for canonical construction.
            intermediate_dim: Per-expert intermediate width for canonical construction.
            grouped_linear: Strategy for canonical construction.
            fqn: Target path for canonical construction diagnostics.
            module: Existing source experts for Family construction.
            strategy: Resolved grouped-linear strategy for Family construction.
            module_fqn: Target path for Family construction diagnostics.
            context: Replacement topology and policy facts for Family construction.
        """
        if module is None:
            if strategy is not None or any(size is None for size in (num_experts, hidden_dim, intermediate_dim)):
                raise TypeError("Provide a Family module and strategy or all three dimensions and grouped_linear")
            self._validate_grouped_linear(grouped_linear)
            # Canonical construction bypasses the Family parent's source transposition.
            nn.Module.__init__(self)  # pylint: disable=non-parent-init-called
            self.gate_up_proj = nn.Parameter(torch.empty(num_experts, 2 * intermediate_dim, hidden_dim))
            self.down_proj = nn.Parameter(torch.empty(num_experts, hidden_dim, intermediate_dim))
            self.act_fn = nn.SiLU()
            self.activation_limit = None
            self._bind_source_layout(grouped_linear, fqn)
            return
        if grouped_linear is not None or any(size is not None for size in (num_experts, hidden_dim, intermediate_dim)):
            raise TypeError("Family construction does not accept canonical projection dimensions or grouped_linear")
        self._validate_grouped_linear(strategy)
        context = context or {}
        prefix = f"{module_fqn or '<root>'}: LowPrecisionGroupedExperts"
        if any(context.get(axis) for axis in ("tp", "cp", "ep", "pp")) or context.get("ep_size", 1) != 1:
            raise NotImplementedError(f"{prefix} requires TP=CP=EP=PP=1")
        super().__init__(module=module, module_fqn=module_fqn, context=context)
        if (not self.has_gate or self.add_bias or self.use_2d_experts
                or not self.router_gating_in_fp32):
            raise ValueError(f"{prefix} requires 3D bias-free SwiGLU with routing scores after GMM2")
        self._preserve_source_layout = False
        self.grouped_linear = strategy
        self.fqn = module_fqn
        if not isinstance(strategy, HiFloat8GroupedLinear):
            _validate_tile_alignment(
                "fake_w4a8" if isinstance(strategy, FakeW4A8GroupedLinear) else strategy.FORMAT_NAME,
                getattr(context.get("low_precision"), "block_size", MX_BLOCK_SIZE_32),
                (tuple(self.gate_up_proj.shape), tuple(self.down_proj.shape)),
            )

    def _bind_source_layout(self, grouped_linear: GroupedLinear, fqn: str) -> None:
        """Record canonical layout metadata without touching source registrations."""
        self._preserve_source_layout = True
        self.num_experts, _, self.hidden_dim = self.gate_up_proj.shape
        self.intermediate_dim = self.down_proj.shape[-1]
        self.num_local_experts = self.num_experts
        self.hidden_size = self.hidden_dim
        self.intermediate_size = self.intermediate_dim
        self.has_gate = True
        self.is_transposed = False
        self.fqn = fqn
        self.grouped_linear = grouped_linear

    @staticmethod
    def _validate_grouped_linear(grouped_linear: GroupedLinear) -> None:
        """Require a plain strategy so module registration stays unchanged."""
        if isinstance(grouped_linear, nn.Module) or not isinstance(
            grouped_linear,
            GroupedLinear,
        ):
            raise TypeError(
                "grouped_linear must be a non-Module GroupedLinear instance"
            )

    @classmethod
    def from_module(
        cls,
        source: nn.Module,
        *,
        fqn: str,
        grouped_linear: GroupedLinear,
    ) -> "LowPrecisionGroupedExperts":
        """Create a no-allocation shell retaining the source registrations."""
        cls._validate_grouped_linear(grouped_linear)
        parameter_names = tuple(source._parameters)  # pylint: disable=protected-access
        if parameter_names != _EXPERT_PARAMETER_NAMES:
            raise TypeError(
                f"{fqn!r} must register packed expert parameters "
                f"{_EXPERT_PARAMETER_NAMES}, but got {parameter_names}."
            )
        gate_up_proj = source._parameters["gate_up_proj"]  # pylint: disable=protected-access
        down_proj = source._parameters["down_proj"]  # pylint: disable=protected-access
        if gate_up_proj is None or down_proj is None:
            raise ValueError(f"{fqn!r} packed expert parameters cannot be None.")
        if gate_up_proj.ndim != GROUPED_NDIM or down_proj.ndim != GROUPED_NDIM:
            raise ValueError(
                f"{fqn!r} packed expert weights must be three-dimensional."
            )
        expected_down_shape = (
            gate_up_proj.shape[0],
            gate_up_proj.shape[2],
            gate_up_proj.shape[1] // _GATE_UP_BRANCHES,
        )
        if (
            gate_up_proj.shape[1] % _GATE_UP_BRANCHES
            or tuple(down_proj.shape) != expected_down_shape
        ):
            raise ValueError(
                f"{fqn!r} has incompatible gate/up and down expert shapes: "
                f"gate_up={tuple(gate_up_proj.shape)}, down={tuple(down_proj.shape)}."
            )

        converted = cls.__new__(cls)  # pylint: disable=no-value-for-parameter
        nn.Module.__init__(converted)  # pylint: disable=unnecessary-dunder-call
        for name, parameter in source._parameters.items():  # pylint: disable=protected-access
            converted.register_parameter(name, parameter)
        for name, buffer in source._buffers.items():  # pylint: disable=protected-access
            converted.register_buffer(
                name,
                buffer,
                persistent=name not in source._non_persistent_buffers_set,  # pylint: disable=protected-access
            )
        for name, module in source._modules.items():  # pylint: disable=protected-access
            converted.add_module(name, module)
        converted.act_fn = getattr(source, "act_fn", nn.SiLU())

        converted._bind_source_layout(grouped_linear, fqn)
        converted.activation_limit = getattr(source, "limit", None)
        if hasattr(source, "config"):
            converted.config = source.config
        converted.training = source.training
        return converted

    @staticmethod
    def _activation(module: nn.Module, config: Any, gated_linear_unit: bool = True) -> Any:
        """Adapt V4.1's native activation without retaining its source parameters."""
        hidden_act = getattr(config, "hidden_act", None)
        if hidden_act is None:
            hidden_act = getattr(config, "hidden_activation", None)
        if hidden_act is None:
            hidden_act = getattr(config, "mlp_hidden_act", None)
        source_activation = getattr(module, "act_fn", None)
        if (hidden_act not in (None, "silu", "swiglu")
                or (source_activation is not None and source_activation is not F.silu
                    and not isinstance(source_activation, (nn.SiLU, type(ACT2FN["silu"]))))):
            raise ValueError("LowPrecisionGroupedExperts requires SwiGLU source activation")
        if isinstance(module, DeepseekV4Experts):
            return partial(_swiglu, limit=module.limit)
        GroupedExperts._activation(module, config, gated_linear_unit)
        return _swiglu

    def _apply_activation(self, projected: Tensor) -> Tensor:
        """Preserve the source callable for policy conversion and Family's SwiGLU contract."""
        if not self._preserve_source_layout:
            return self.activation_func(projected)
        gate, up = projected.chunk(_GATE_UP_BRANCHES, dim=-1)
        if self.activation_limit is not None:
            gate = gate.clamp(max=self.activation_limit)
            up = up.clamp(min=-self.activation_limit, max=self.activation_limit)
        return self.act_fn(gate) * up

    def _grouped_gemm_expert_forward(self, gate_up_proj, down_proj, permuted,
                                     tokens_per_expert, permuted_probs):
        """Use one grouped-linear chain for both persistent weight layouts."""
        del permuted_probs
        if not self._preserve_source_layout:
            gate_up_proj = gate_up_proj.transpose(-2, -1)
            down_proj = down_proj.transpose(-2, -1)
        try:
            projected = _GroupedLinearFunction.apply(
                permuted, gate_up_proj, tokens_per_expert, self.grouped_linear, GROUP_LIST_COUNTS)
            return _GroupedLinearFunction.apply(
                self._apply_activation(projected), down_proj, tokens_per_expert,
                self.grouped_linear, GROUP_LIST_COUNTS)
        except (LowPrecisionCapabilityError, W4A8CapabilityError) as exc:
            if not self._preserve_source_layout:
                raise
            target = self.fqn or "<unknown>"
            raise LowPrecisionCapabilityError(
                f"Low-precision grouped-expert target {target!r} cannot run: {exc}"
            ) from exc

    def _grouped_forward(self, sorted_inputs: Tensor, tokens_per_expert: Tensor) -> Tensor:
        """Keep canonical routing at the same shared expert-major computation boundary."""
        return self.forward_expert_major(sorted_inputs, tokens_per_expert)

    def make_transforms(self) -> list[WeightRenaming | WeightConverter]:
        """Preserve the checkpoint conversion contract of the chosen construction path.

        Returns:
            Family weight transforms, or no transforms for canonical policy weights.
        """
        return [] if self._preserve_source_layout else super().make_transforms()

    def reset_parameters(self) -> None:
        """Only Family construction opts into the baseline's materialization initialization."""
        if not self._preserve_source_layout:
            super().reset_parameters()

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Sort routed tokens, execute grouped experts, and restore token order."""
        if not self._preserve_source_layout:
            return super().forward(hidden_states, top_k_index, top_k_weights)

        if hidden_states.ndim != MATRIX_NDIM:
            raise ValueError(
                "Grouped experts require two-dimensional hidden_states, "
                f"but got shape {tuple(hidden_states.shape)}."
            )
        if (
            top_k_index.shape != top_k_weights.shape
            or top_k_index.ndim != MATRIX_NDIM
        ):
            raise ValueError(
                "top_k_index and top_k_weights must have the same two-dimensional shape."
            )
        if top_k_index.shape[0] != hidden_states.shape[0]:
            raise ValueError(
                "The routed token count must match hidden_states: "
                f"routes={top_k_index.shape[0]}, tokens={hidden_states.shape[0]}."
            )

        input_dtype = hidden_states.dtype
        token_count = hidden_states.shape[0]
        top_k = top_k_index.shape[1]
        source_token_indices = torch.arange(
            token_count,
            device=hidden_states.device,
        ).unsqueeze(1).expand(-1, top_k).reshape(-1)
        flattened_expert_indices = top_k_index.reshape(-1)
        expert_order = flattened_expert_indices.argsort()
        sorted_inputs = hidden_states[source_token_indices[expert_order]]
        tokens_per_expert = torch.bincount(
            flattened_expert_indices,
            minlength=self.num_experts,
        )
        if tokens_per_expert.numel() != self.num_experts:
            raise ValueError(
                f"Expert indices must be in [0, {self.num_experts})."
            )
        sorted_outputs = self._grouped_forward(sorted_inputs, tokens_per_expert)
        sorted_weights = top_k_weights.reshape(-1)[expert_order]
        sorted_outputs = sorted_outputs * sorted_weights.unsqueeze(-1)

        inverse_order = torch.empty_like(expert_order)
        inverse_order[expert_order] = torch.arange(
            expert_order.shape[0],
            device=expert_order.device,
        )
        # Preserve FP32 route accumulation and round only after the top-k sum.
        return sorted_outputs[inverse_order].view(
            token_count,
            top_k,
            self.hidden_dim,
        ).sum(dim=1).to(dtype=input_dtype)


def replace_grouped_experts_with_policy(
    *,
    module: nn.Module,
    module_fqn: str,
    context: Mapping[str, Any],
    scheme: LowPrecisionDtypeScheme | LowPrecisionConfig | None,
) -> LowPrecisionGroupedExperts:
    """Replace canonical packed experts with a policy-selected grouped-linear strategy.

    Args:
        module: Source experts with canonical packed parameters.
        module_fqn: Target path used in capability diagnostics.
        context: Existing replacement topology context.
        scheme: Resolved dtype scheme or legacy low-precision policy.

    Returns:
        Unified low-precision experts retaining the source registrations and layout.
    """
    active_model_parallel_axes = [
        axis.upper()
        for axis in ("tp", "cp", "ep", "pp")
        if context.get(axis)
    ]
    if active_model_parallel_axes:
        raise NotImplementedError(
            "Low-precision grouped experts currently require TP=CP=EP=PP=1; "
            f"active axes: {active_model_parallel_axes}."
        )
    parameters = tuple(module.parameters(recurse=False))
    strategy = build_low_precision_strategy(
        scheme,
        tile_shapes=tuple(tuple(parameter.shape) for parameter in parameters),
    )
    return LowPrecisionGroupedExperts.from_module(
        module,
        fqn=module_fqn,
        grouped_linear=strategy,
    )


@module_replacement
def replace_low_precision_grouped_experts(
    *, module: nn.Module, module_fqn: str, context: Mapping[str, Any],
) -> LowPrecisionGroupedExperts:
    """Resolve every native/fake dtype scheme with the shared strategy factory.

    Args:
        module: Source packed expert module accepted by the Family baseline.
        module_fqn: Target path used in capability diagnostics.
        context: Replacement facts containing the resolved low_precision scheme.

    Returns:
        Unified low-precision experts using Family routing and checkpoint transforms.
    """
    strategy = build_low_precision_strategy(
        context.get("low_precision"),
        tile_shapes=(tuple(module.gate_up_proj.shape), tuple(module.down_proj.shape)),
    )
    return LowPrecisionGroupedExperts(
        module=module, module_fqn=module_fqn, context=context, strategy=strategy,
    )


@module_replacement
def replace_mxfp8_grouped_experts(
    *, module: nn.Module, module_fqn: str, context: Mapping[str, Any],
) -> LowPrecisionGroupedExperts:
    """Keep the explicit MXFP8 entry bound to native W8A8.

    Args:
        module: Source packed expert module.
        module_fqn: Target path used in capability diagnostics.
        context: Existing replacement topology and policy context.

    Returns:
        Unified low-precision experts bound to the native MXFP8 strategy.
    """
    return replace_low_precision_grouped_experts(
        module=module, module_fqn=module_fqn,
        context={**context, "low_precision": LowPrecisionDtypeScheme()},
    )


@module_replacement
def replace_w4a8_grouped_experts(
    *, module: nn.Module, module_fqn: str, context: Mapping[str, Any],
) -> LowPrecisionGroupedExperts:
    """Keep the explicit W4A8 entry native and retain the configured block size.

    Args:
        module: Source packed expert module.
        module_fqn: Target path used in capability diagnostics.
        context: Replacement facts containing the low_precision block size.

    Returns:
        Unified low-precision experts bound to the native W4A8 strategy.
    """
    scheme = LowPrecisionDtypeScheme(
        weight_format="mxfp4", act_format="mxfp8", block_size=context["low_precision"].block_size,
    )
    return replace_low_precision_grouped_experts(
        module=module, module_fqn=module_fqn, context={**context, "low_precision": scheme},
    )


@module_replacement
def replace_fake_w4a8_grouped_experts(
    *, module: nn.Module, module_fqn: str, context: Mapping[str, Any],
) -> LowPrecisionGroupedExperts:
    """Keep the explicit fake W4A8 entry distinct from native quantization.

    Args:
        module: Source packed expert module.
        module_fqn: Target path used in capability diagnostics.
        context: Replacement facts containing the low_precision block size.

    Returns:
        Unified low-precision experts bound to the fake W4A8 strategy.
    """
    scheme = LowPrecisionDtypeScheme(
        weight_format="mxfp4", act_format="mxfp8", is_fake_quantize=True,
        block_size=context["low_precision"].block_size,
    )
    return replace_low_precision_grouped_experts(
        module=module, module_fqn=module_fqn, context={**context, "low_precision": scheme},
    )


@module_replacement
class MXFP8GroupedSwiGLU(GroupedExperts):
    """MXFP8 local/EP expert computation with optional dual-axis SwiGLU fusion."""

    requires_ep_grouped_gemm = True

    def __init__(self, *, module: nn.Module, module_fqn: str = "",
                 context: Mapping[str, Any] | None = None) -> None:
        """Validate the supported structure and initialize a non-fused variant."""
        context = context or {}
        self.module_fqn = module_fqn
        prefix = f"{module_fqn or '<root>'}: MXFP8GroupedSwiGLU"
        if any(context.get(axis) for axis in ("tp", "cp", "pp")):
            raise NotImplementedError(f"{prefix} requires TP=CP=PP=1")
        try:
            super().__init__(module=module, module_fqn=module_fqn, context=context)
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError(f"{prefix}: {exc}") from exc
        self._validate_source_contract(module, prefix)
        try:
            validate_npu_gmm_runtime()
        except RuntimeError as exc:
            raise RuntimeError(f"{prefix}: {exc}") from exc
        self.quantizer = MXFP8Quantizer()

    def _validate_source_contract(self, module: nn.Module, prefix: str) -> None:
        """Check the initial layout and activation before any device computation."""
        hidden_act = (getattr(self.config, "hidden_act", None)
                      or getattr(self.config, "hidden_activation", None)
                      or getattr(self.config, "mlp_hidden_act", "silu"))
        source_activation = getattr(module, "act_fn", None)
        if (not self.has_gate or self.add_bias or hidden_act not in ("silu", "swiglu")
                or (source_activation is not None
                    and not isinstance(source_activation, (nn.SiLU, type(ACT2FN["silu"]))))):
            raise ValueError(f"{prefix} requires standard bias-free SwiGLU")
        if self.use_2d_experts or not self.router_gating_in_fp32:
            raise ValueError(f"{prefix} requires 3D packed weights and routing weights after GMM2")
        if self.hidden_size % 32 or self.intermediate_size % 32:
            raise ValueError(f"{prefix} requires H and I divisible by 32")

    def _grouped_gemm_expert_forward(self, gate_up_proj, down_proj, permuted,
                                     tokens_per_expert, permuted_probs):
        """Validate expert groups and dispatch local math without applying scores."""
        del permuted_probs
        counts = tokens_per_expert.to(device=permuted.device)
        if counts.ndim != 1 or counts.shape[0] != gate_up_proj.shape[0]:
            raise ValueError(f"{self.module_fqn}: counts must contain one count per local expert")
        if counts.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"{self.module_fqn}: counts must be integers")
        if tokens_per_expert.device.type == "cpu":
            if bool((tokens_per_expert < 0).any()) or int(tokens_per_expert.sum()) != permuted.shape[0]:
                raise ValueError(f"{self.module_fqn}: counts must be non-negative and sum to token count")
        return self._local_compute(permuted, gate_up_proj, down_proj, counts)

    def _local_compute(self, permuted, gate_up_proj, down_proj, counts):
        """Run the non-fused local chain after common protocol validation."""
        projected = mxfp8_projection(permuted, gate_up_proj, counts, self.quantizer)
        gate, up = projected.chunk(2, dim=-1)
        activated = F.silu(gate) * up
        return mxfp8_projection(activated, down_proj, counts, self.quantizer)


@module_replacement
class FusedMXFP8GroupedSwiGLU(MXFP8GroupedSwiGLU):
    """Use fused activation/quantization while retaining the same family contract."""

    def __init__(self, *, module: nn.Module, module_fqn: str = "",
                 context: Mapping[str, Any] | None = None) -> None:
        """Validate both fused kernels during construction, with module diagnostics."""
        self.clamp_limit = float(module.limit) if isinstance(module, DeepseekV4Experts) else None
        if self.clamp_limit is not None and (not math.isfinite(self.clamp_limit) or self.clamp_limit <= 0):
            raise ValueError(f"{module_fqn or '<root>'}: clipped MXFP8 fusion requires a positive finite limit")
        super().__init__(module=module, module_fqn=module_fqn, context=context)
        try:
            validate_npu_swiglu_runtime(clamp_limit=self.clamp_limit)
        except RuntimeError as exc:
            raise RuntimeError(f"{module_fqn or '<root>'}: {exc}") from exc

    @classmethod
    def _activation(cls, module: nn.Module, config: Any, gated_linear_unit: bool = True) -> Any:
        """Accept DeepSeek V4/V4.1 clipped experts while keeping the custom-gate guard for others."""
        return (None if isinstance(module, DeepseekV4Experts)
                else super()._activation(module, config, gated_linear_unit))

    def _local_compute(self, permuted, gate_up_proj, down_proj, counts):
        """Use one autograd boundary without passing quantized carriers between nodes."""
        return fused_grouped_swiglu(
            permuted, gate_up_proj, down_proj, counts, self.quantizer, clamp_limit=self.clamp_limit)


_THIS_MODULE = "hyper_parallel.components.modules.moe.grouped_experts.low_precision_grouped_experts"

register_module_family(ModuleFamily(
    name="grouped_experts",
    baseline=GroupedExperts,
    variants={
        "w8a8": f"{_THIS_MODULE}.replace_mxfp8_grouped_experts",
        "w4a8": f"{_THIS_MODULE}.replace_w4a8_grouped_experts",
        "fake_w4a8": f"{_THIS_MODULE}.replace_fake_w4a8_grouped_experts",
        "hif8": f"{_THIS_MODULE}.replace_low_precision_grouped_experts",
        "mxfp8": f"{_THIS_MODULE}.MXFP8GroupedSwiGLU",
        "fused_mxfp8": f"{_THIS_MODULE}.FusedMXFP8GroupedSwiGLU",
    },
    selector_specs={
        "use_mxfp8": SelectorSpec(bool, False),
        "fused": SelectorSpec(bool, False, aliases=("use_fused_mxfp8",)),
        "weight_format": SelectorSpec(str, ""),
        "act_format": SelectorSpec(str, ""),
        "is_fake_quantize": SelectorSpec(bool, False),
    },
    variant_rules=(
        VariantRule({"use_mxfp8": True, "fused": False,
                     "weight_format": "", "act_format": "", "is_fake_quantize": False}, "mxfp8"),
        VariantRule({"use_mxfp8": True, "fused": True,
                     "weight_format": "", "act_format": "", "is_fake_quantize": False}, "fused_mxfp8"),
        VariantRule({"use_mxfp8": False, "fused": False,
                     "weight_format": "mxfp8", "act_format": "mxfp8", "is_fake_quantize": False}, "w8a8"),
        VariantRule({"use_mxfp8": False, "fused": True,
                     "weight_format": "mxfp8", "act_format": "mxfp8", "is_fake_quantize": False}, "fused_mxfp8"),
        VariantRule({"use_mxfp8": False, "fused": False,
                     "weight_format": "mxfp4", "act_format": "mxfp8", "is_fake_quantize": False}, "w4a8"),
        VariantRule({"use_mxfp8": False, "fused": False,
                     "weight_format": "mxfp4", "act_format": "mxfp8", "is_fake_quantize": True}, "fake_w4a8"),
        VariantRule({"use_mxfp8": False, "fused": False,
                     "weight_format": "hif8", "act_format": "hif8", "is_fake_quantize": False}, "hif8"),
    ),
))

__all__ = [
    "FusedMXFP8GroupedSwiGLU",
    "GroupedExperts",
    "LowPrecisionGroupedExperts",
    "MXFP8GroupedSwiGLU",
    "replace_fake_w4a8_grouped_experts",
    "replace_grouped_experts_with_policy",
    "replace_low_precision_grouped_experts",
    "replace_mxfp8_grouped_experts",
    "replace_w4a8_grouped_experts",
]
