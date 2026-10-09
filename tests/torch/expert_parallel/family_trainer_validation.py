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
"""Run the actual TextTrainer with observational family acceptance checks.

Usage: FAMILY_TRACE_DIR=/path/to/trace python -m torch.distributed.run
--standalone --nproc-per-node=2 -m tests.torch.expert_parallel.family_trainer_validation config.yaml
"""

import faulthandler
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any

import torch
import torch_npu
from safetensors.torch import load_file

from hyper_parallel.components.modules.grouped_experts import GroupedExperts
from hyper_parallel.components.modules.mxfp8_grouped_swiglu import FusedMXFP8GroupedSwiGLU
from hyper_parallel.trainer.config.parser import parse_training_args
from hyper_parallel.trainer.text_trainer import TextTrainer
from tests.torch.expert_parallel._test_grouped_experts_mxfp8 import CountingMXOps


def _local(tensor: torch.Tensor) -> torch.Tensor:
    """Read a local shard for diagnostics without initiating communication."""
    return tensor.to_local() if hasattr(tensor, "to_local") else tensor


def _rng_digest():
    return {
        "cpu": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
        "npu": hashlib.sha256(torch.npu.get_rng_state().cpu().numpy().tobytes()).hexdigest(),
        "python": hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
    }


class ObservedTextTrainer(TextTrainer):
    """Add observations while retaining production Trainer and planner execution."""

    def __init__(self, config: Any) -> None:
        """Build production components and attach real operator counters."""
        super().__init__(config)
        self.trace_dir = Path(os.environ["FAMILY_TRACE_DIR"])
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.rank = int(os.environ.get("RANK", "0"))
        if (self.trace_dir / f"rank{self.rank}.jsonl").exists():
            raise FileExistsError(f"Use a fresh FAMILY_TRACE_DIR: {self.trace_dir}")
        self.experts = [(name, module) for name, module in self.base.model.named_modules()
                        if isinstance(module, GroupedExperts)]
        assert len(self.experts) == 2
        self.ops = []
        self.batches = []
        self.gradient_events = []
        self.gradient_handles = {}
        native = load_file(str(Path(config.model.pretrained_model_name_or_path) / "model.safetensors"))
        for name, parameter in self.base.model.named_parameters():
            if ".experts." in name:
                prefix, projection = name.rsplit(".", 1)
                if projection == "gate_up_proj":
                    expected = torch.stack([torch.cat([native[f"{prefix}.{index}.gate_proj.weight"],
                                                       native[f"{prefix}.{index}.up_proj.weight"]], dim=0)
                                            for index in range(4)]).transpose(-2, -1)
                else:
                    expected = torch.stack([native[f"{prefix}.{index}.down_proj.weight"]
                                            for index in range(4)]).transpose(-2, -1)
                if config.accelerator.ep_size == 2:
                    expected = expected.chunk(2, dim=0)[self.rank % 2]
            else:
                expected = native[name]
            torch.testing.assert_close(_local(parameter).detach().cpu(), expected, rtol=0, atol=0, msg=name)
        for name, expert in self.experts:
            assert expert.gate_up_proj.shape[-2:] == (128, 512), name
            assert expert.down_proj.shape[-2:] == (256, 128), name
            assert _local(expert.gate_up_proj).shape[0] == 4 // config.accelerator.ep_size, name
            if hasattr(expert, "quantizer"):
                ops = CountingMXOps(torch_npu)
                expert.quantizer._npu_ops = ops
                self.ops.append(ops)
            expert.register_forward_pre_hook(self._observe_live_parameters(name))
        print(json.dumps({"rank": self.rank, "family_modules": [
            {"fqn": name, "class": type(expert).__name__,
             "local_gate_up_shape": list(_local(expert.gate_up_proj).shape)}
            for name, expert in self.experts]}), flush=True)

    def _gradient_hook(self, name):
        def observe(gradient: torch.Tensor) -> None:
            """Check and record the gradient without changing it.

            Args:
                gradient: Gradient of one live expert parameter.
            """
            values = _local(gradient).detach().float()
            assert bool(torch.isfinite(values).all()), name
            self.gradient_events.append({"name": name, "norm": float(values.norm())})
        return observe

    def _observe_live_parameters(self, name):
        def observe(module: torch.nn.Module, args: tuple[Any, ...]) -> None:
            """Attach observations after HSDP has installed live parameters.

            Args:
                module: Expert module with materialized computation parameters.
                args: Forward inputs, unused by this observer.
            """
            del args
            # HSDP installs its pre-hook first and replaces sharded parameters
            # with live unsharded parameters. Observe those actual graph leaves.
            for handle in self.gradient_handles.get(name, []):
                handle.remove()
            self.gradient_handles[name] = [
                getattr(module, weight).register_hook(self._gradient_hook(f"{name}.{weight}"))
                for weight in ("gate_up_proj", "down_proj")]
        return observe

    def on_micro_step_begin(self, micro_batch: dict[str, Any]) -> None:
        """Hash real input/label tensors to verify resumed data progression.

        Args:
            micro_batch: Actual inputs and loss metadata produced by the data pipeline.
        """
        tensors = {name: hashlib.sha256(_local(value).detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                   for name, value in micro_batch.items() if isinstance(value, torch.Tensor)}
        self.batches.append(tensors)
        super().on_micro_step_begin(micro_batch)

    def on_train_begin(self) -> None:
        """Record RNG and step after production checkpoint restoration."""
        super().on_train_begin()
        record = {"step": self.base.state.global_step, "rng": _rng_digest()}
        (self.trace_dir / f"rank{self.rank}-begin.json").write_text(json.dumps(record), encoding="utf-8")

    def on_step_end(self, loss: Any = None, loss_dict: Any = None, grad_norm: Any = None) -> None:
        """Record completed updates after the normal checkpoint callbacks run.

        Args:
            loss: Completed optimizer step's aggregate loss.
            loss_dict: Named losses passed through to production callbacks.
            grad_norm: Gradient norm measured before the optimizer update.
        """
        super().on_step_end(loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)
        assert math.isfinite(loss) and math.isfinite(grad_norm)
        assert self.gradient_events and any(event["norm"] > 0 for event in self.gradient_events)
        step = self.base.state.global_step
        state = {name: _local(parameter).detach().cpu().clone()
                 for name, parameter in self.base.model.named_parameters()}
        torch.save(state, self.trace_dir / f"rank{self.rank}-step{step}.pt")
        record = {"rank": self.rank, "step": step, "loss": loss, "grad_norm": grad_norm,
                  "batches": self.batches, "expert_gradients": self.gradient_events,
                  "gmm_calls": sum(len(ops.calls) for ops in self.ops),
                  "fused_forward": sum(ops.fused_forward for ops in self.ops),
                  "fused_backward": sum(ops.fused_backward for ops in self.ops), "rng": _rng_digest()}
        if self.ops:
            assert record["gmm_calls"] > 0
        if any(isinstance(expert, FusedMXFP8GroupedSwiGLU) for _, expert in self.experts):
            assert record["fused_forward"] > 0 and record["fused_backward"] > 0
        with (self.trace_dir / f"rank{self.rank}.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        self.batches = []
        self.gradient_events = []


if __name__ == "__main__":
    faulthandler.dump_traceback_later(90)
    try:
        ObservedTextTrainer(parse_training_args()).train()
    finally:
        faulthandler.cancel_dump_traceback_later()
