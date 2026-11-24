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
from collections.abc import Callable, Mapping
from functools import partial
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers.activations import ACT2FN
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Experts

from hyper_parallel.components.checkpoint.weight_conversion import Transpose, WeightConverter
from hyper_parallel.components.functional import grouped_matmul, swiglu as npu_swiglu
from hyper_parallel.components.functional.moe_token_permute import moe_token_permute
from hyper_parallel.components.functional.moe_token_unpermute import moe_token_unpermute
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
)
from hyper_parallel.components.quantization.functional import build_low_precision_strategy
from hyper_parallel.components.quantization.functional.base_gmm_func import (
    GroupedLinear,
    _GroupedLinearFunction,
)
from hyper_parallel.components.quantization.functional.fake_w4a8_gmm_func import FakeW4A8GroupedLinear
from hyper_parallel.components.quantization.functional.mxfp8_fused_grouped_swiglu import (
    fused_grouped_swiglu,
)
from hyper_parallel.components.quantization.functional.mxfp8_gmm_func import MXFP8GroupedLinear
from hyper_parallel.components.quantization.functional.mxfp8_projection import mxfp8_projection
from hyper_parallel.components.quantization.functional.strategy_factory import _validate_tile_alignment
from hyper_parallel.components.quantization.functional.w4a8_gmm_func import W4A8GroupedLinear
from hyper_parallel.components.quantization.ops.npu_fake_w4a8 import validate_fake_w4a8_gmm_runtime
from hyper_parallel.components.quantization.ops.npu_mxfp8 import (
    LowPrecisionCapabilityError,
    validate_npu_gmm_runtime,
    validate_npu_swiglu_runtime,
)
from hyper_parallel.components.quantization.ops.npu_w4a8 import (
    W4A8CapabilityError,
    validate_w4a8_gmm_runtime,
)
from hyper_parallel.components.quantization.quantizers.mxfp8 import MXFP8Quantizer
from hyper_parallel.models.replacement import module_replacement

_EXPERT_PARAMETER_NAMES = ("gate_up_proj", "down_proj")
_GATE_UP_BRANCHES = 2


def _swiglu(projected: Tensor, *, limit: float | None = None) -> Tensor:
    """Preserve projection dtype and optionally apply the native DeepSeek V4/V4.1 clamp."""
    gate, up = projected.chunk(2, dim=-1)
    if limit is not None:
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    return F.silu(gate) * up


class QuantizedGroupedExperts(nn.Module):
    """Canonical ``[E, 2I, H]`` low-precision shell for packed grouped experts.

    The bound grouped-linear compute may select MXFP8, HiFloat8, native W4A8,
    or fake W4A8. The shell owns no quantizer; it holds only the
    ``grouped_linear`` instance selected by ``strategy_factory``.
    """

    def __init__(
        self,
        num_experts: int,
        hidden_dim: int,
        intermediate_dim: int,
        *,
        grouped_linear: GroupedLinear,
        fqn: str = "",
    ) -> None:
        """Create an unbound packed-expert module."""
        self._validate_grouped_linear(grouped_linear)
        super().__init__()
        self.gate_up_proj = nn.Parameter(
            torch.empty(
                num_experts,
                _GATE_UP_BRANCHES * intermediate_dim,
                hidden_dim,
            )
        )
        self.down_proj = nn.Parameter(
            torch.empty(num_experts, hidden_dim, intermediate_dim)
        )
        self.act_fn = nn.SiLU()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.fqn = fqn
        self.grouped_linear = grouped_linear
        self.activation_limit = None

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
    ) -> "QuantizedGroupedExperts":
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

        converted.num_experts = gate_up_proj.shape[0]
        converted.hidden_dim = gate_up_proj.shape[2]
        converted.intermediate_dim = gate_up_proj.shape[1] // _GATE_UP_BRANCHES
        converted.fqn = fqn
        converted.grouped_linear = grouped_linear
        converted.activation_limit = getattr(source, "limit", None)
        if hasattr(source, "config"):
            converted.config = source.config
        converted.training = source.training
        return converted

    def _grouped_forward(
        self,
        sorted_inputs: torch.Tensor,
        tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        """Run already expert-major tokens through the two low-precision GMMs."""
        try:
            gate_up = _GroupedLinearFunction.apply(
                sorted_inputs,
                self.gate_up_proj,
                tokens_per_expert,
                self.grouped_linear,
                GROUP_LIST_COUNTS,
            )
            gate, up = gate_up.chunk(_GATE_UP_BRANCHES, dim=-1)
            if self.activation_limit is None:
                intermediate = self.act_fn(gate) * up
            else:
                gate = gate.clamp(max=self.activation_limit)
                up = up.clamp(min=-self.activation_limit, max=self.activation_limit)
                intermediate = self.act_fn(gate) * up
            return _GroupedLinearFunction.apply(
                intermediate,
                self.down_proj,
                tokens_per_expert,
                self.grouped_linear,
                GROUP_LIST_COUNTS,
            )
        except (LowPrecisionCapabilityError, W4A8CapabilityError) as exc:
            target = self.fqn or "<unknown>"
            raise LowPrecisionCapabilityError(
                f"Low-precision grouped-expert target {target!r} cannot run: {exc}"
            ) from exc

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Sort routed tokens, execute grouped experts, and restore token order."""
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
    strategy_builder: Callable[..., GroupedLinear] = build_low_precision_strategy,
) -> QuantizedGroupedExperts:
    """Replace canonical packed experts with a policy-selected grouped-linear strategy."""
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
    strategy = strategy_builder(
        scheme,
        tile_shapes=tuple(tuple(parameter.shape) for parameter in parameters),
    )
    return QuantizedGroupedExperts.from_module(
        module,
        fqn=module_fqn,
        grouped_linear=strategy,
    )


@module_replacement
class GroupedExperts(nn.Module):
    """Transposed ``[E, H, 2I]`` / ``[E, I, H]`` baseline shell for the low-precision expert family."""

    @staticmethod
    def _source_parameters(module: nn.Module) -> tuple[nn.Parameter, nn.Parameter]:
        """Validate and return bias-free 3D packed expert weight parameters."""
        missing = [name for name in ("gate_up_proj", "down_proj") if not hasattr(module, name)]
        if missing:
            raise TypeError(f"GroupedExperts source module is missing: {missing}")
        source_gate_up = module.gate_up_proj
        source_down = module.down_proj
        if not isinstance(source_gate_up, nn.Parameter):
            raise TypeError("GroupedExperts requires gate_up_proj to be an nn.Parameter")
        if not isinstance(source_down, nn.Parameter):
            raise TypeError("GroupedExperts requires down_proj to be an nn.Parameter")
        return source_gate_up, source_down

    @staticmethod
    def _source_dimensions(module: nn.Module, config: Any) -> tuple[int, int, int]:
        """Infer expert count and projection dimensions from the source module."""
        num_experts = getattr(module, "num_local_experts", getattr(module, "num_experts", None))
        if num_experts is None and config is not None:
            num_experts = getattr(config, "n_routed_experts", getattr(config, "num_experts", None))
        hidden_size = getattr(module, "hidden_size", getattr(module, "hidden_dim", None))
        if hidden_size is None and config is not None:
            hidden_size = getattr(config, "hidden_size", None)
        intermediate_size = getattr(module, "intermediate_size", getattr(module, "intermediate_dim", None))
        if intermediate_size is None and config is not None:
            intermediate_size = getattr(
                config, "moe_intermediate_size", getattr(config, "intermediate_size", None),
            )
        if num_experts is None or hidden_size is None or intermediate_size is None:
            raise ValueError("GroupedExperts requires expert count, hidden size, and intermediate size")
        return num_experts, hidden_size, intermediate_size

    @staticmethod
    def _activation(module: nn.Module, config: Any, gated_linear_unit: bool = True) -> Any:
        """Resolve the source-compatible gated expert activation."""
        del gated_linear_unit
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
            source_apply_gate_func is not None
            and getattr(source_apply_gate_func, "__name__", "") != "_default_apply_gate"
        ):
            raise ValueError("GroupedExperts custom _apply_gate requires a dedicated replacement")
        if hidden_act == "silu" and bool(getattr(config, "use_fused_swiglu", True)):
            return partial(npu_swiglu, dim=-1)

        def glu(x: torch.Tensor) -> torch.Tensor:
            gate, value = torch.chunk(x, 2, dim=-1)
            return activation_func(gate) * value

        return glu

    @staticmethod
    def _needs_transpose(
        source_shapes: tuple[tuple[int, ...], tuple[int, ...]],
        source_is_transposed: bool | None,
        transformers_shapes: tuple[tuple[int, ...], tuple[int, ...]],
        npu_shapes: tuple[tuple[int, ...], tuple[int, ...]],
    ) -> bool:
        """Return whether 3D source weights need transposition into ``[E, H, 2I]`` / ``[E, I, H]``."""
        if source_is_transposed is False:
            if source_shapes != transformers_shapes:
                raise ValueError("GroupedExperts source weights do not match their declared layout")
            return True
        if source_is_transposed is True:
            if source_shapes != npu_shapes:
                raise ValueError("GroupedExperts source weights do not match their declared layout")
            return False
        if source_shapes == transformers_shapes:
            return True
        if source_shapes == npu_shapes:
            return False
        raise ValueError("GroupedExperts source weight shapes are incompatible with the configured dimensions")

    def __init__(
        self,
        *,
        module: nn.Module,
        module_fqn: str = "",
        context: Mapping[str, Any] | None = None,
    ) -> None:
        """Build the transposed 3D SwiGLU shell from a packed Transformers experts module."""
        super().__init__()
        del module_fqn, context
        config = getattr(module, "config", None)
        self.config = config
        self.initializer_range = getattr(config, "initializer_range", None)
        if bool(getattr(module, "has_bias", False)) or any(
            getattr(module, name, None) is not None
            for name in ("bias1", "bias2", "gate_up_proj_bias", "down_proj_bias")
        ):
            raise ValueError("GroupedExperts does not support source modules with bias")
        if not bool(getattr(module, "has_gate", getattr(config, "gated_linear_unit", True))):
            raise ValueError("GroupedExperts requires gated SwiGLU expert weights")
        if not bool(getattr(module, "is_concatenated", True)):
            raise ValueError("GroupedExperts requires concatenated gate/up expert weights")
        if not bool(getattr(module, "router_gating_in_fp32", True)):
            raise ValueError("GroupedExperts requires routing scores applied after GMM2")

        source_gate_up, source_down = self._source_parameters(module)
        num_experts, hidden_size, intermediate_size = self._source_dimensions(module, config)
        self.num_local_experts = num_experts
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.activation_func = self._activation(module, config, True)
        self._group_list = None
        self._tokens_per_expert_gmm = None

        npu_gate_up_shape = (num_experts, hidden_size, 2 * intermediate_size)
        npu_down_shape = (num_experts, intermediate_size, hidden_size)
        transformers_shapes = (
            (num_experts, 2 * intermediate_size, hidden_size),
            (num_experts, hidden_size, intermediate_size),
        )
        self._transpose_source_weights = self._needs_transpose(
            (tuple(source_gate_up.shape), tuple(source_down.shape)),
            getattr(module, "is_transposed", None),
            transformers_shapes,
            (npu_gate_up_shape, npu_down_shape),
        )
        if self._transpose_source_weights:
            target_gate_up = nn.Parameter(
                torch.empty(npu_gate_up_shape, device=source_gate_up.device, dtype=source_gate_up.dtype),
                requires_grad=source_gate_up.requires_grad,
            )
            self.down_proj = nn.Parameter(
                torch.empty(npu_down_shape, device=source_down.device, dtype=source_down.dtype),
                requires_grad=source_down.requires_grad,
            )
            if not source_gate_up.is_meta:
                with torch.no_grad():
                    target_gate_up.copy_(source_gate_up.transpose(-2, -1))
                    self.down_proj.copy_(source_down.transpose(-2, -1))
        else:
            target_gate_up = source_gate_up
            self.down_proj = source_down
        self.gate_up_proj = target_gate_up

        self.is_transposed = True
        self._hp_reset_after_materialization = True
        self.train(module.training)

    def reset_parameters(self) -> None:
        """Initialize grouped weights with the source model's configured standard deviation."""
        if self.initializer_range is None:
            raise ValueError("GroupedExperts random initialization requires config.initializer_range")
        nn.init.normal_(self.gate_up_proj, mean=0.0, std=self.initializer_range)
        nn.init.normal_(self.down_proj, mean=0.0, std=self.initializer_range)

    def make_transforms(self) -> list[WeightConverter]:
        """Describe reversible source-checkpoint to transposed-layout conversion."""
        if not self._transpose_source_weights:
            return []
        return [
            WeightConverter(
                source_patterns=name,
                target_patterns=name,
                operations=[Transpose(dim0=-2, dim1=-1)],
            )
            for name in ("gate_up_proj", "down_proj")
        ]

    def _grouped_gemm_expert_forward(
        self, gate_up_proj, down_proj, permuted, tokens_per_expert, permuted_probs,
    ):
        """Baseline GroupGemm expert computation (NPU-only)."""
        del permuted_probs
        self._tokens_per_expert_gmm = tokens_per_expert.to(device=permuted.device)
        self._group_list = self._tokens_per_expert_gmm.cumsum(dim=0)

        numel = permuted.nelement()
        if torch.compiler.is_compiling() or not isinstance(numel, int) or numel != 0:
            fc1_output = grouped_matmul(  # pylint: disable=not-callable
                permuted, gate_up_proj, bias=None, group_list=self._group_list, group_type=0, group_list_type=0,
            )
        else:
            fc1_output = torch.matmul(permuted, gate_up_proj.view(self.hidden_size, -1))

        fc1_output = self.activation_func(fc1_output)
        numel = fc1_output.nelement()
        if torch.compiler.is_compiling() or not isinstance(numel, int) or numel != 0:
            return grouped_matmul(  # pylint: disable=not-callable
                fc1_output, down_proj, bias=None, group_list=self._group_list, group_type=0, group_list_type=0,
            )
        return torch.matmul(fc1_output, down_proj.view(-1, self.hidden_size))

    def forward_expert_major(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
        scores: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run grouped experts on expert-major routed tokens."""
        self.num_local_experts = num_tokens_per_expert.shape[0]
        return self._grouped_gemm_expert_forward(
            self.gate_up_proj, self.down_proj, x, num_tokens_per_expert, scores,
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
        permuted_tokens, sorted_indices = moe_token_permute(
            hidden_states_flat,
            top_k_index,
        )
        permuted_probs = top_k_weights.view(-1).index_select(0, sorted_indices)
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
        output = moe_token_unpermute(
            expert_outputs,
            sorted_indices,
            top_k_weights,
        )
        return output.view(hidden_shape)


class LowPrecisionGroupedExperts(GroupedExperts):
    """Preserve transposed expert layout/routing and bind one QAT ``GroupedLinear`` strategy."""

    def __init__(self, *, module: nn.Module, strategy: GroupedLinear, module_fqn: str = "",
                 context: Mapping[str, Any]) -> None:
        """Build local expert computation from the already resolved dtype scheme."""
        prefix = f"{module_fqn or '<root>'}: LowPrecisionGroupedExperts"
        if any(context.get(axis) for axis in ("tp", "cp", "ep", "pp")) or context.get("ep_size", 1) != 1:
            raise NotImplementedError(f"{prefix} requires TP=CP=EP=PP=1")
        super().__init__(module=module, module_fqn=module_fqn, context=context)
        self.grouped_linear = strategy
        _validate_tile_alignment(
            "fake_w4a8" if isinstance(strategy, FakeW4A8GroupedLinear) else strategy.FORMAT_NAME,
            context["low_precision"].block_size,
            (tuple(self.gate_up_proj.shape), tuple(self.down_proj.shape)),
        )

    @staticmethod
    def _activation(module: nn.Module, config: Any, gated_linear_unit: bool = True) -> Any:
        """Adapt V4.1's native expert activation without retaining its source parameters."""
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

    def _grouped_gemm_expert_forward(self, gate_up_proj, down_proj, permuted,
                                     tokens_per_expert, permuted_probs):
        """Adapt layout at the shared strategy boundary; outer routing applies scores."""
        del permuted_probs
        projected = _GroupedLinearFunction.apply(
            permuted, gate_up_proj.transpose(-2, -1), tokens_per_expert, self.grouped_linear, 1)
        return _GroupedLinearFunction.apply(
            self.activation_func(projected), down_proj.transpose(-2, -1),
            tokens_per_expert, self.grouped_linear, 1)


@module_replacement
def replace_mxfp8_grouped_experts(
    *, module: nn.Module, module_fqn: str, context: Mapping[str, Any],
) -> LowPrecisionGroupedExperts:
    """Bind the MXFP8GroupedLinear strategy selected by the Family rule."""
    validate_npu_gmm_runtime()
    return LowPrecisionGroupedExperts(
        module=module, module_fqn=module_fqn, context=context,
        strategy=MXFP8GroupedLinear(),
    )


@module_replacement
def replace_w4a8_grouped_experts(
    *, module: nn.Module, module_fqn: str, context: Mapping[str, Any],
) -> LowPrecisionGroupedExperts:
    """Bind the W4A8GroupedLinear strategy selected by the Family rule."""
    validate_w4a8_gmm_runtime()
    return LowPrecisionGroupedExperts(
        module=module, module_fqn=module_fqn, context=context,
        strategy=W4A8GroupedLinear(context["low_precision"].block_size),
    )


@module_replacement
def replace_fake_w4a8_grouped_experts(
    *, module: nn.Module, module_fqn: str, context: Mapping[str, Any],
) -> LowPrecisionGroupedExperts:
    """Bind the FakeW4A8GroupedLinear strategy selected by the Family rule."""
    validate_fake_w4a8_gmm_runtime(context["low_precision"].block_size)
    return LowPrecisionGroupedExperts(
        module=module, module_fqn=module_fqn, context=context,
        strategy=FakeW4A8GroupedLinear(context["low_precision"].block_size),
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
        if (hidden_act not in ("silu", "swiglu")
                or (source_activation is not None
                    and not isinstance(source_activation, (nn.SiLU, type(ACT2FN["silu"]))))):
            raise ValueError(f"{prefix} requires standard bias-free SwiGLU")
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
    ),
))

__all__ = [
    "FusedMXFP8GroupedSwiGLU",
    "GroupedExperts",
    "LowPrecisionGroupedExperts",
    "MXFP8GroupedSwiGLU",
    "QuantizedGroupedExperts",
    "replace_fake_w4a8_grouped_experts",
    "replace_grouped_experts_with_policy",
    "replace_mxfp8_grouped_experts",
    "replace_w4a8_grouped_experts",
]
