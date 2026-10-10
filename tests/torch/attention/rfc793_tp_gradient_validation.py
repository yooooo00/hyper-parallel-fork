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
"""Capture and compare the first real Trainer step's BF16 TP gradients.

Collect with torchrun and FAMILY_TRACE_DIR, using the ordinary Trainer YAML::

    python -m tests.torch.attention.rfc793_tp_gradient_validation collect train.yaml

Compare a TP=1 trace with TP=N after both runs have finished::

    python -m tests.torch.attention.rfc793_tp_gradient_validation compare \
        --reference-trace single --parallel-trace parallel --tp-size 2 --output errors.json

Collection observes gradients immediately before the production clip function.
It does not change selection, loading, backward, normalization, clipping or
optimizer updates. Fixed BF16 limits are relative L2 <= .03 and maximum absolute
error <= .003 for every parameter shard, with all errors saved before failure.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch

from hyper_parallel.core.utils.clip_grad import _get_grad_obj
from hyper_parallel.trainer import text_trainer
from hyper_parallel.trainer.config.parser import parse_training_args
from tests.torch.attention.rfc793_trainer_validation import FamilyTextTrainer


_RELATIVE_LIMIT = .03
_ABSOLUTE_LIMIT = .003


def _collect(config_file: str, overrides: list[str]) -> None:
    """Run the ordinary Trainer while recording its first pre-clip gradient set."""
    trainer = FamilyTextTrainer(parse_training_args([config_file, *overrides]))
    original_clip = text_trainer.clip_grad_norm_
    recorded = False

    def observed_clip(model: torch.nn.Module, *args: Any, **kwargs: Any) -> Any:
        """Save the completed distributed gradients and call the real clip operation.

        Args:
            model: Distributed model containing the completed parameter gradients.
            args: Positional arguments forwarded unchanged to the observed operation.
            kwargs: Keyword arguments forwarded unchanged to the observed operation.
        """
        nonlocal recorded
        if not recorded:
            gradients = {}
            for name, parameter in model.named_parameters():
                gradient = _get_grad_obj(parameter)
                if gradient is not None:
                    local = gradient.to_local() if hasattr(gradient, "to_local") else gradient
                    gradients[name] = local.detach().cpu().clone()
            if not gradients:
                raise AssertionError("The first optimizer step produced no gradients")
            torch.save(gradients, trainer.trace_dir / f"rank{trainer.rank}-preclip.pt")
            recorded = True
        return original_clip(model, *args, **kwargs)

    with patch.object(text_trainer, "clip_grad_norm_", observed_clip):
        trainer.train()
    if not recorded:
        raise AssertionError("Gradient capture requires at least one step with max_grad_norm > 0")


def _first_record(directory: Path, rank: int) -> dict:
    return json.loads((directory / f"rank{rank}.jsonl").read_text(encoding="utf-8").splitlines()[0])


def _expected_shard(expected: torch.Tensor, actual: torch.Tensor, rank: int, tp_size: int) -> torch.Tensor:
    """Infer the single ordinary TP axis from logical and local parameter shapes."""
    if actual.shape == expected.shape:
        return expected
    axes = [axis for axis, (local, global_) in enumerate(zip(actual.shape, expected.shape)) if local != global_]
    if actual.ndim != expected.ndim or len(axes) != 1:
        raise AssertionError(f"Unsupported TP shape pair: {tuple(actual.shape)} vs {tuple(expected.shape)}")
    axis = axes[0]
    if expected.shape[axis] != tp_size * actual.shape[axis]:
        raise AssertionError(f"Expected TP={tp_size} partition along dimension {axis}")
    return expected.chunk(tp_size, dim=axis)[rank]


def _parameter_error(name: str, actual: torch.Tensor, expected: torch.Tensor, rank: int, tp_size: int) -> dict:
    """Measure the local shard against the corresponding full-reference slice."""
    expected = _expected_shard(expected, actual, rank, tp_size).float()
    actual = actual.float()
    error = actual - expected
    relative = float(error.norm()) / max(float(expected.norm()), 1e-12)
    maximum = float(error.abs().max())
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    return {"rank": rank, "name": name, "relative_l2": relative, "max_abs": maximum,
            "actual_norm": float(actual.norm()), "reference_norm": float(expected.norm()),
            "passed": finite and relative <= _RELATIVE_LIMIT and maximum <= _ABSOLUTE_LIMIT}


def _record_failures(reference: dict, current: dict, rank: int) -> list[str]:
    """Check data identity, finite loss and the completed distributed gradient norm."""
    failures = []
    for field in ("step", "batches"):
        if current[field] != reference[field]:
            failures.append(f"rank {rank}: different {field}")
    if not math.isfinite(current["loss"]) or abs(current["loss"] - reference["loss"]) > _ABSOLUTE_LIMIT:
        failures.append(f"rank {rank}: loss differs by more than {_ABSOLUTE_LIMIT}")
    norm_limit = _RELATIVE_LIMIT * abs(reference["grad_norm"]) + _ABSOLUTE_LIMIT
    norm_error = abs(current["grad_norm"] - reference["grad_norm"])
    if not math.isfinite(current["grad_norm"]) or norm_error > norm_limit:
        failures.append(f"rank {rank}: gradient norm differs from reference")
    return failures


def _compare(reference_trace: Path, parallel_trace: Path, tp_size: int, output: Path) -> None:
    """Check all parameter gradients, data identity, loss and distributed clip norms."""
    if tp_size < 1:
        raise ValueError("tp_size must be positive")
    reference = torch.load(reference_trace / "rank0-preclip.pt", weights_only=True, map_location="cpu")
    reference_record = _first_record(reference_trace, 0)
    records, failures, norms = [], [], []
    if not all(math.isfinite(reference_record[key]) for key in ("loss", "grad_norm")):
        failures.append("The reference loss or gradient norm is not finite")
    for rank in range(tp_size):
        actual = torch.load(parallel_trace / f"rank{rank}-preclip.pt", weights_only=True, map_location="cpu")
        current = _first_record(parallel_trace, rank)
        if set(actual) != set(reference):
            raise AssertionError(f"Rank {rank}: gradient parameter names differ from the full reference")
        failures.extend(_record_failures(reference_record, current, rank))
        norms.append(current["grad_norm"])
        records.extend(_parameter_error(name, value, reference[name], rank, tp_size)
                       for name, value in sorted(actual.items()))
    if len(set(norms)) != 1:
        failures.append("TP ranks used different gradient clipping norms")
    failures.extend(f"rank {item['rank']}: {item['name']}" for item in records if not item["passed"])
    report = {"relative_l2_limit": _RELATIVE_LIMIT, "max_abs_limit": _ABSOLUTE_LIMIT,
              "reference_loss": reference_record["loss"], "reference_grad_norm": reference_record["grad_norm"],
              "parallel_grad_norms": norms, "failures": failures, "parameters": records}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if failures:
        raise AssertionError(f"TP gradient comparison failed; complete results: {output}")
    print(f"Validated {len(records)} parameter shards; results: {output}")


def main() -> None:
    """Dispatch capture or comparison without embedding local asset paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("collect")
    collect.add_argument("config_file")
    compare = commands.add_parser("compare")
    compare.add_argument("--reference-trace", type=Path, required=True)
    compare.add_argument("--parallel-trace", type=Path, required=True)
    compare.add_argument("--tp-size", type=int, default=2)
    compare.add_argument("--output", type=Path, required=True)
    args, overrides = parser.parse_known_args()
    if args.command == "collect":
        _collect(args.config_file, overrides)
    else:
        if overrides:
            parser.error(f"Unknown comparison arguments: {overrides}")
        _compare(args.reference_trace, args.parallel_trace, args.tp_size, args.output)


if __name__ == "__main__":
    main()
