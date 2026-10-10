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
"""Compare real Family Trainer runs without inventing a training-quality tolerance."""

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml


def _records(directory: Path, rank: int) -> dict[int, dict]:
    items = [json.loads(line) for line in (directory / f"rank{rank}.jsonl").read_text().splitlines()]
    records = {item["step"]: item for item in items}
    assert len(items) == len(records), (directory, rank, "duplicate steps")
    return records


def _compare_parameters(actual: Path, expected: Path) -> None:
    actual_state = torch.load(actual, map_location="cpu", weights_only=True)
    expected_state = torch.load(expected, map_location="cpu", weights_only=True)
    assert actual_state.keys() == expected_state.keys(), actual
    for name, value in actual_state.items():
        torch.testing.assert_close(value, expected_state[name], rtol=0, atol=0,
                                   msg=f"{actual}: {name}")


def compare_continuation(root: Path, case: str, kind: str, rank: int) -> dict:
    """Require exact same-topology resume or recomputation, including training state.

    Args:
        root: Generated acceptance directory.
        case: Configuration stem, such as tp1-cp2-ep2-fused.
        kind: Either resume (checkpoint step 5) or ac (full recomputation).
        rank: Rank whose local state is compared.

    Returns:
        Verified steps and the observed extra forward calls for recomputation.
    """
    reference_dir, actual_dir = root / f"trace-{case}", root / f"{kind}-{case}"
    reference, actual = _records(reference_dir, rank), _records(actual_dir, rank)
    first_step = 6 if kind == "resume" else 1
    assert set(actual) == set(range(first_step, max(reference) + 1)), (case, kind, rank)
    state_fields = ("rng", "optimizer", "scheduler")
    if kind == "resume":
        begin = json.loads((actual_dir / f"rank{rank}-begin.json").read_text())
        assert begin["step"] == 5
        for field in state_fields:
            assert begin[field] == reference[5][field], (case, rank, "restored", field)
    for step, record in actual.items():
        for field in ("loss", "grad_norm", "batches", *state_fields):
            assert record[field] == reference[step][field], (case, kind, rank, step, field)
        filename = f"rank{rank}-step{step}.pt"
        _compare_parameters(actual_dir / filename, reference_dir / filename)
    last_step = max(actual)
    extra_forward = actual[last_step]["fused_forward"] - reference[last_step]["fused_forward"]
    if kind == "ac":
        assert extra_forward > 0, (case, rank, "full recomputation did not execute fused forward again")
    return {"steps": sorted(actual), "exact": True, "extra_fused_forward": extra_forward}


def compare_equivalent(root: Path, actual_case: str, reference_case: str) -> dict:
    """Require exact training equivalence for default and explicit adapter runs.

    Args:
        root: Generated configuration directory containing full trace folders.
        actual_case: Explicit adapter configuration stem.
        reference_case: Corresponding default adapter configuration stem.

    Returns:
        Number of ranks and steps whose state, parameters and kernel counts agree.
    """
    actual_dir, reference_dir = root / f"trace-{actual_case}", root / f"trace-{reference_case}"
    rank_files = sorted(reference_dir.glob("rank*-build.json"))
    assert rank_files and len(list(actual_dir.glob("rank*-build.json"))) == len(rank_files)
    fields = ("loss", "grad_norm", "batches", "rng", "optimizer", "scheduler",
              "gmm_calls", "linear_calls", "fused_forward", "fused_backward")
    for rank in range(len(rank_files)):
        actual, reference = _records(actual_dir, rank), _records(reference_dir, rank)
        assert actual.keys() == reference.keys(), (actual_case, reference_case, rank)
        for step, record in actual.items():
            for field in fields:
                assert record[field] == reference[step][field], (actual_case, rank, step, field)
            filename = f"rank{rank}-step{step}.pt"
            _compare_parameters(actual_dir / filename, reference_dir / filename)
    return {"actual": actual_case, "reference": reference_case, "ranks": len(rank_files), "exact": True}


def _loss_comparisons(traces: dict) -> dict:
    comparisons = {}
    for case, ranks in traces.items():
        topology, mode = case.rsplit("-", 1)
        for reference_mode in ("bf16", "mxfp8"):
            reference_case = f"{topology}-{reference_mode}"
            if mode == reference_mode or reference_case not in traces or mode == "bf16":
                continue
            references = traces[reference_case]
            assert len(ranks) == len(references)
            deltas = []
            relative = []
            for rank, (actual, expected) in enumerate(zip(ranks, references)):
                assert actual.keys() == expected.keys(), (case, reference_case, "step counts")
                for step, record in actual.items():
                    assert record["batches"] == expected[step]["batches"], (case, rank, step, "input mismatch")
                    delta = abs(record["loss"] - expected[step]["loss"])
                    deltas.append(delta)
                    relative.append(delta / max(abs(expected[step]["loss"]), 1e-30))
            comparisons[f"{case}-vs-{reference_mode}"] = {
                "max_absolute_loss": max(deltas), "max_relative_loss": max(relative),
                "criterion": "descriptive only; no convergence-quality threshold",
            }
    return comparisons


def compare(root: Path, cases: list[str], resume_cases: list[str], recompute_cases: list[str]) -> dict[str, Any]:
    """Validate complete runs, report CE differences and check requested continuations.

    Args:
        root: Generated configuration directory containing independent trace folders.
        cases: Configuration stems whose full declared step counts are required.
        resume_cases: Cases to compare from a fresh process restored at step five.
        recompute_cases: Cases to compare with full activation recomputation enabled.

    Returns:
        Loss trajectories, descriptive differences and exact continuation checks.
    """
    traces, report = {}, {"runs": {}, "resume": {}, "recompute": {}}
    for case in cases:
        config = yaml.safe_load((root / f"{case}.yaml").read_text())
        directory = root / f"trace-{case}"
        rank_files = sorted(directory.glob("rank*-build.json"))
        assert rank_files, (case, "no build records")
        ranks = [_records(directory, rank) for rank in range(len(rank_files))]
        steps = config["training"]["train_iters"]
        for rank, records in enumerate(ranks):
            assert set(records) == set(range(1, steps + 1)), (case, rank, "incomplete training")
            if not case.endswith("bf16"):
                assert records[steps]["linear_calls"] > 0 and records[steps]["gmm_calls"] > 0
            if case.endswith("fused"):
                assert records[steps]["fused_forward"] > 0 and records[steps]["fused_backward"] > 0
        traces[case] = ranks
        report["runs"][case] = {"ranks": len(ranks), "steps": steps,
                                "rank_losses": [[item[step]["loss"] for step in range(1, steps + 1)]
                                                for item in ranks]}
    report["loss_comparisons"] = _loss_comparisons(traces)
    for kind, selected in (("resume", resume_cases), ("ac", recompute_cases)):
        for case in selected:
            assert case in traces, (case, "continuation requires the original run")
            destination = "resume" if kind == "resume" else "recompute"
            report[destination][case] = [compare_continuation(root, case, kind, rank)
                                         for rank in range(len(traces[case]))]
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--resume-cases", nargs="*", default=[])
    parser.add_argument("--recompute-cases", nargs="*", default=[])
    parser.add_argument("--equivalent-cases", nargs=2, metavar=("EXPLICIT", "DEFAULT"))
    arguments = parser.parse_args()
    comparison = compare(arguments.root, arguments.cases, arguments.resume_cases, arguments.recompute_cases)
    if arguments.equivalent_cases:
        comparison["equivalent"] = compare_equivalent(arguments.root, *arguments.equivalent_cases)
    print(json.dumps(comparison, indent=2))
