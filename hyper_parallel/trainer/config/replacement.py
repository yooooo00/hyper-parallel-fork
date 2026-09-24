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
"""Replacement-only Target retaining family selectors for Python/CLI overrides."""

from typing import Any, Callable

from hyper_parallel.components.modules.family import resolve_module_family_target
from hyper_parallel.trainer.config.target import Target


class ReplacementTarget(Target):
    """Resolve the implementation without discarding the requested baseline."""

    def __init__(self, _target_: Callable, *, target_path: str, **kwargs: Any) -> None:
        """Select a final callable and strictly validate constructor keywords."""
        self.requested_target = _target_
        self.selectors = {name: value for name, value in kwargs.items()
                          if name in ("use_mxfp8", "use_fused_mxfp8")}
        resolved, arguments = resolve_module_family_target(_target_, kwargs)
        super().__init__(resolved, target_path=target_path, **arguments)

    def replace(self, **changes: Any) -> "ReplacementTarget":
        """Re-resolve selectors after an override, without mutating the original."""
        return type(self)(self.requested_target, target_path=self._target_path,
                          **{**self._kwargs, **self.selectors, **changes})

    def to_dict(self) -> dict[str, Any]:
        """Serialize the baseline path and original selectors for round trips."""
        return {**super().to_dict(), **self.selectors}
