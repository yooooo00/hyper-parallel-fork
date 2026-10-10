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
"""Observe real mixed Family training, checkpoints and kernels without replacing arithmetic.

This diagnostic runner intentionally synchronizes device tensors for snapshots and
assertions. Its timings are not representative of production training throughput.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import faulthandler
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch
import torch_npu
from safetensors.torch import load_file
from transformers import PretrainedConfig

from hyper_parallel.components.modules.family import get_module_family_selection
from hyper_parallel.components.modules.grouped_experts import GroupedExperts
from hyper_parallel.components.modules.mxfp8_grouped_swiglu import FusedMXFP8GroupedSwiGLU
from hyper_parallel.components.quantization.modules.mxfp8_linear import MXFP8Linear
from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.distributed._builder.planner import ShardingPlanner
from hyper_parallel.distributed.plan import ShardingPlan
from hyper_parallel.trainer.config.parser import parse_training_args
from hyper_parallel.trainer.config.trainer import TrainerConfig
from hyper_parallel.trainer.text_trainer import TextTrainer
from tests.torch.expert_parallel._test_grouped_experts_mxfp8 import CountingMXOps
from tests.torch.expert_parallel.family_trainer_validation import _local, _rng_digest


class CountingProjectionOps(CountingMXOps):
    """Count genuine FP8 projection matmuls as well as expert kernels."""

    def __init__(self, backend: Any) -> None:
        """Attach the real NPU backend and initialize projection call counters.

        Args:
            backend: Installed torch_npu operator provider.
        """
        super().__init__(backend)
        self.linear_calls = 0

    def quant_matmul(self, x1: torch.Tensor, x2: torch.Tensor, scale: torch.Tensor,
                     **kwargs: Any) -> torch.Tensor:
        """Check FP8 operands and execute the existing projection operator.

        Args:
            x1: Quantized left operand.
            x2: Quantized right operand.
            scale: Right operand scales.
            **kwargs: Unmodified left scales and output dtype.

        Returns:
            The real device operator's output tensor.
        """
        assert x1.dtype == torch.float8_e4m3fn and x2.dtype == torch.float8_e4m3fn, (
            f"Expected two float8_e4m3fn operands, got {x1.dtype} and {x2.dtype}"
        )
        self.linear_calls += 1
        return super().quant_matmul(x1, x2, scale, **kwargs)


def _kernel_counts(ops: CountingProjectionOps | None) -> dict[str, int]:
    """Return cumulative calls using the same keys for every observed module."""
    return {
        "gmm_calls": len(ops.calls) if ops else 0,
        "linear_calls": ops.linear_calls if ops else 0,
        "fused_forward": ops.fused_forward if ops else 0,
        "fused_backward": ops.fused_backward if ops else 0,
    }


@dataclass
class _ModuleObservation:
    """Keep gradient and kernel evidence for one projection or packed expert."""

    kind: str
    fused: bool
    ops: CountingProjectionOps | None
    trainable_parameters: set[str] = field(default_factory=set)
    forwards: int = 0
    nonempty_forwards: int = 0
    previous_counts: dict[str, int] = field(default_factory=lambda: _kernel_counts(None))

    def finish_step(self, name: str, observed_gradients: set[str]) -> dict[str, Any]:
        """Validate this module's evidence and reset per-step observations.

        Args:
            name: Full module name, used for failures and gradient matching.
            observed_gradients: Parameter names with finite gradients in this step.

        Returns:
            Cumulative and per-step operator counts, input activity and gradient names.
        """
        assert self.forwards > 0, f"{name}: module was not called in this optimizer step"
        missing = self.trainable_parameters - observed_gradients
        assert not missing, f"{name}: missing gradients for {sorted(missing)}"
        counts = _kernel_counts(self.ops)
        delta = {key: value - self.previous_counts[key] for key, value in counts.items()}
        if self.ops is not None and self.nonempty_forwards:
            required = ["linear_calls" if self.kind == "linear" else "gmm_calls"]
            if self.fused:
                required.extend(("fused_forward", "fused_backward"))
            assert all(delta[key] > 0 for key in required), f"{name}: missing real kernels: {delta}"
        record = {
            "kind": self.kind, "fused": self.fused,
            "forwards": self.forwards, "nonempty_forwards": self.nonempty_forwards,
            "trainable_parameters": sorted(self.trainable_parameters),
            "counts": counts, "step_counts": delta,
        }
        self.previous_counts = counts
        self.trainable_parameters.clear()
        self.forwards = 0
        self.nonempty_forwards = 0
        return record


def _mesh_record(mesh: DeviceMesh | None) -> dict[str, Any] | None:
    """Serialize mesh dimensions and physical ranks without collective operations."""
    if mesh is None:
        return None
    return {"shape": list(mesh.mesh_shape), "axes": list(mesh.mesh_dim_names), "ranks": list(mesh.rank_list)}


def _source_parameter(name: str, native: Mapping[str, torch.Tensor], config: PretrainedConfig) -> torch.Tensor:
    """Convert native checkpoint weights to the Family's persistent packed layout."""
    if name.endswith("linear_qkv.weight"):
        prefix = name.rsplit(".", 2)[0]
        pieces = [native[f"{prefix}.{part}_proj.weight"] for part in ("q", "k", "v")]
        heads = config.num_key_value_heads
        return torch.cat([value.reshape(heads, -1, value.shape[-1]) for value in pieces], 1).flatten(0, 1)
    if ".experts." in name:
        prefix, projection = name.rsplit(".", 1)
        weights = []
        for index in range(config.num_experts):
            if projection == "gate_up_proj":
                weight = torch.cat([native[f"{prefix}.{index}.{part}_proj.weight"] for part in ("gate", "up")])
            else:
                weight = native[f"{prefix}.{index}.down_proj.weight"]
            weights.append(weight.transpose(-2, -1))
        return torch.stack(weights)
    return native[name]


def _state_description(value: Any) -> Any:
    """Describe state deterministically, including local tensor bytes and scalar types."""
    if isinstance(value, torch.Tensor):
        tensor = _local(value).detach().cpu().contiguous()
        payload = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        return {"type": "tensor", "dtype": str(tensor.dtype), "shape": list(tensor.shape),
                "sha256": hashlib.sha256(payload).hexdigest()}
    if isinstance(value, Mapping):
        entries = [(_state_description(key), _state_description(item)) for key, item in value.items()]
        entries.sort(key=lambda entry: json.dumps(entry[0], sort_keys=True))
        return {"type": "mapping", "entries": entries}
    if isinstance(value, (tuple, list)):
        return {"type": type(value).__name__, "items": [_state_description(item) for item in value]}
    if value is None or isinstance(value, (bool, int, float, str)):
        return {"type": type(value).__name__, "value": value}
    if isinstance(value, (torch.dtype, torch.device)):
        return {"type": type(value).__name__, "value": str(value)}
    raise TypeError(f"Unsupported optimizer/scheduler state type: {type(value).__qualname__}")


def _optimizer_state(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    """Read existing slots and groups without checkpoint hooks or lazy state initialization."""
    parameter_names = {}
    groups = []
    for group in optimizer.param_groups:
        names = []
        for parameter in group["params"]:
            parameter_names.setdefault(id(parameter), getattr(parameter, "model_name", len(parameter_names)))
            names.append(parameter_names[id(parameter)])
        metadata = {key: value for key, value in group.items() if key != "params"}
        # ChainedOptimizer restores this inactive group-level placeholder even
        # for AdamW, whose actual step is stored separately for each parameter.
        metadata.setdefault("step", None)
        groups.append({**metadata, "params": names})
    state = {parameter_names[id(parameter)] if isinstance(parameter, torch.Tensor) else parameter: value
             for parameter, value in optimizer.state.items()}
    return {"state": state, "param_groups": groups}


def _component_state(component: Any) -> Any:
    """Read optimizer slots directly; use only scheduler state dictionaries.

    ChainedOptimizer.state_dict delegates to DCP, which can initialize empty
    optimizer state by executing a step. Observations must never invoke it.
    """
    if component is None:
        return None
    if isinstance(component, (tuple, list)):
        return [_component_state(child) for child in component]
    optimizers = getattr(component, "optimizers_dict", None)
    if optimizers is not None:
        result = {"optimizers": {name: _component_state(child) for name, child in optimizers.items()}}
        main_parameters = getattr(component, "_main_param_by_fqn", None)
        if main_parameters is not None:
            result["main_parameters"] = main_parameters
        return result
    if isinstance(component, torch.optim.Optimizer):
        return _optimizer_state(component)
    return component.state_dict()


def _state_digest(component: Any) -> str:
    """Hash production state independently of dictionary order and object addresses."""
    description = _state_description(_component_state(component))
    encoded = json.dumps(description, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class FamilyTextTrainer(TextTrainer):
    """Observe production TP/CP/EP construction, gradients and completed updates."""

    def __init__(self, config: TrainerConfig) -> None:
        """Build the real Trainer and attach diagnostics after its sharding hooks.

        Args:
            config: Fully resolved production Trainer configuration.
        """
        self.plans: list[ShardingPlan] = []
        original = ShardingPlanner.plan

        def observe_plan(planner: ShardingPlanner, *args: Any, **kwargs: Any) -> ShardingPlan:
            """Record the plan returned by the unmodified planner.

            Args:
                planner: Production planner instance.
                *args: Unmodified planner positional arguments.
                **kwargs: Unmodified planner keyword arguments.

            Returns:
                The actual plan computed by the production implementation.
            """
            plan = original(planner, *args, **kwargs)
            self.plans.append(plan)
            return plan

        with patch.object(ShardingPlanner, "plan", observe_plan):
            super().__init__(config)
        self.trace_dir = Path(os.environ["FAMILY_TRACE_DIR"])
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.rank = int(os.environ.get("RANK", "0"))
        if (self.trace_dir / f"rank{self.rank}.jsonl").exists():
            raise FileExistsError(f"Use a fresh FAMILY_TRACE_DIR: {self.trace_dir}")
        self.batches: list[dict[str, str]] = []
        self.gradient_events: list[dict[str, Any]] = []
        self.gradient_handles: dict[str, list[Any]] = {}
        self.observations: dict[str, _ModuleObservation] = {}
        self._attach_module_observers()
        if config.fsdp_config.dp_shard_size == config.fsdp_config.edp_shard_size == 1:
            self._assert_source_weights(config)
        self._record_build()

    def _attach_module_observers(self) -> None:
        """Keep NPU arithmetic intact and observe each local computation independently."""
        for name, module in self.base.model.named_modules():
            if not isinstance(module, (MXFP8Linear, GroupedExperts)):
                continue
            ops = None
            if hasattr(module, "quantizer"):
                ops = CountingProjectionOps(torch_npu)
                module.quantizer._npu_ops = ops
            self.observations[name] = _ModuleObservation(
                kind="linear" if isinstance(module, MXFP8Linear) else "experts",
                fused=isinstance(module, FusedMXFP8GroupedSwiGLU), ops=ops,
            )
            module.register_forward_pre_hook(self._observe_module_parameters(name), with_kwargs=True)

    def _record_build(self) -> None:
        """Persist selected Families, actual meshes and planner-owned local parameter shapes."""
        families = []
        for name, module in self.base.model.named_modules():
            selection = get_module_family_selection(module)
            if selection is not None:
                families.append({"fqn": name, "family": selection.family.name,
                                 "variant": selection.variant, "class": selection.implementation})
        topology = {
            "dense": _mesh_record(self.base.mesh.device_mesh),
            "dense_fsdp": _mesh_record(self.base.mesh.fsdp_non_moe_mesh),
            "expert_fsdp": _mesh_record(self.base.mesh.fsdp_moe_mesh),
            "families": families,
            "local_parameters": {name: list(_local(param).shape)
                                 for name, param in self.base.model.named_parameters()},
        }
        (self.trace_dir / f"rank{self.rank}-build.json").write_text(json.dumps(topology, indent=2), encoding="utf-8")
        (self.trace_dir / f"rank{self.rank}-plan.txt").write_text(
            "\n".join(plan.explain() for plan in self.plans), encoding="utf-8")

    def _assert_source_weights(self, config: TrainerConfig) -> None:
        """Assert exact conversion and TP/EP slices before checkpoint restoration."""
        native = load_file(str(Path(config.model.pretrained_model_name_or_path) / "model.safetensors"))
        entries = {f"{boundary}.{name}" if boundary else name: placements
                   for plan in self.plans for boundary, spec in plan.modules.items()
                   for name, placements in spec.params.items()}
        for name, parameter in self.base.model.named_parameters():
            expected = _source_parameter(name, native, self.base.model.config)
            for axis, placement in entries.get(name, {}).items():
                if not placement.is_shard():
                    continue
                mesh = (self.base.mesh.fsdp_moe_mesh if axis == "ep" else self.base.mesh.device_mesh)[axis]
                expected = expected.chunk(mesh.size(), dim=placement.dim)[mesh.get_local_rank()]
            torch.testing.assert_close(_local(parameter).detach().cpu(), expected, rtol=0, atol=0, msg=name)

    def _gradient_hook(self, name: str) -> Callable[[torch.Tensor], None]:
        """Capture finite gradients without modifying the values returned to autograd."""
        def observe(gradient: torch.Tensor) -> None:
            """Record one live parameter's local gradient.

            Args:
                gradient: Gradient produced by the actual backward computation.
            """
            values = _local(gradient).detach().float()
            assert bool(torch.isfinite(values).all()), f"{name}: non-finite parameter gradient"
            self.gradient_events.append({"name": name, "norm": float(values.norm())})
        return observe

    def _observe_module_parameters(self, name: str) -> Callable:
        """Attach leaf hooks after HSDP materializes this forward's computation parameters."""
        def observe(module: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            """Track this invocation's input activity and trainable parameter coverage.

            Args:
                module: Local projection or packed expert module.
                args: Actual positional forward inputs.
                kwargs: Actual keyword forward inputs.
            """
            observation = self.observations[name]
            tensors = [value for value in (*args, *kwargs.values()) if isinstance(value, torch.Tensor)]
            assert tensors, f"{name}: expected a tensor input"
            observation.forwards += 1
            observation.nonempty_forwards += int(_local(tensors[0]).numel() > 0)
            for handle in self.gradient_handles.get(name, []):
                handle.remove()
            handles = []
            for key, parameter in module.named_parameters(recurse=False):
                if parameter.requires_grad:
                    parameter_name = f"{name}.{key}"
                    observation.trainable_parameters.add(parameter_name)
                    handles.append(parameter.register_hook(self._gradient_hook(parameter_name)))
            self.gradient_handles[name] = handles
        return observe

    def on_train_begin(self) -> None:
        """Record RNG and optimizer/scheduler states after production checkpoint restoration."""
        super().on_train_begin()
        record = {"step": self.base.state.global_step, "rng": _rng_digest(),
                  "optimizer": _state_digest(self.base.optimizer), "scheduler": _state_digest(self.base.lr_scheduler)}
        (self.trace_dir / f"rank{self.rank}-begin.json").write_text(json.dumps(record), encoding="utf-8")

    def on_micro_step_begin(self, micro_batch: dict[str, Any]) -> None:
        """Hash actual batches and retain first-step tensors for mask/position diagnostics.

        Args:
            micro_batch: Actual model inputs and loss metadata from the production loader.
        """
        tensors = {name: _local(value).detach().cpu() for name, value in micro_batch.items()
                   if isinstance(value, torch.Tensor)}
        if self.base.state.global_step == 0:
            snapshot = {name: tensors.get(name, value) for name, value in micro_batch.items()}
            torch.save(snapshot, self.trace_dir / f"rank{self.rank}-first-batch{len(self.batches)}.pt")
        self.batches.append({name: hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()
                             for name, value in tensors.items()})
        super().on_micro_step_begin(micro_batch)

    def on_step_end(self, loss: Any = None, loss_dict: Any = None, grad_norm: Any = None) -> None:
        """Assert local computation coverage and record updates after production callbacks.

        Args:
            loss: Completed optimizer step's aggregate loss.
            loss_dict: Named losses forwarded to production callbacks.
            grad_norm: Gradient norm measured before the optimizer update.
        """
        super().on_step_end(loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)
        assert math.isfinite(loss) and math.isfinite(grad_norm), f"Non-finite loss={loss} or grad_norm={grad_norm}"
        assert all(math.isfinite(event["norm"]) for event in self.gradient_events), (
            f"Non-finite observed gradient norms: {self.gradient_events}"
        )
        gradient_names = {event["name"] for event in self.gradient_events}
        module_records = {name: observation.finish_step(name, gradient_names)
                          for name, observation in self.observations.items()}
        step = self.base.state.global_step
        torch.save({name: _local(parameter).detach().cpu().clone()
                    for name, parameter in self.base.model.named_parameters()},
                   self.trace_dir / f"rank{self.rank}-step{step}.pt")
        counts = {key: sum(record["counts"][key] for record in module_records.values())
                  for key in _kernel_counts(None)}
        record = {
            "rank": self.rank, "step": step, "loss": loss, "grad_norm": grad_norm,
            "batches": self.batches, "expert_gradients": self.gradient_events,
            **counts, "modules": module_records, "rng": _rng_digest(),
            "optimizer": _state_digest(self.base.optimizer), "scheduler": _state_digest(self.base.lr_scheduler),
        }
        with (self.trace_dir / f"rank{self.rank}.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        self.batches = []
        self.gradient_events = []


if __name__ == "__main__":
    faulthandler.dump_traceback_later(120)
    try:
        FamilyTextTrainer(parse_training_args()).train()
    finally:
        faulthandler.cancel_dump_traceback_later()
