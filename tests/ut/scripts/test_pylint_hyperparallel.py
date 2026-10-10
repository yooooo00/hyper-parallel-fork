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
"""License headers preserve author attribution, valid dates and Apache terms."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import astroid
from pylint.lint import PyLinter

from scripts.pylint_hyperparallel import HyperParallelChecker


class ApacheHeaderTests(unittest.TestCase):
    """Exercise the actual checker on temporary Python source files."""

    def _check(self, content):
        checker = HyperParallelChecker(PyLinter())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "example.py"
            path.write_text(content, encoding="utf-8")
            node = astroid.parse("pass")
            with patch.object(checker, "add_message") as message:
                checker._check_apache_header(node, str(path))
            return message.call_args_list

    def test_current_and_ranged_year_headers(self):
        """Both canonical copyright forms retain the complete Apache contract."""
        source = Path(__file__).read_text(encoding="utf-8")
        for year in ("2026", "2025-2026", "(c) 2024-2026"):
            for author in ("Huawei Technologies Co., Ltd", "Bytedance Ltd. and/or its affiliates"):
                with self.subTest(year=year, author=author):
                    content = source.replace("Copyright 2026", "Copyright " + year, 1)
                    content = content.replace("Huawei Technologies Co., Ltd", author, 1)
                    self.assertEqual(self._check(content), [])

    def test_invalid_or_misplaced_header_still_fails(self):
        """Year flexibility does not accept missing or misplaced license text."""
        source = Path(__file__).read_text(encoding="utf-8")
        invalid = (
            source.replace("Copyright 2026", "Copyright XXXX", 1),
            source.replace("Copyright 2026", "Copyright 0000", 1),
            source.replace("Copyright 2026", "Copyright 2026-2025", 1),
            source.replace("Copyright 2026", "Copyright 2999", 1),
            source.replace("Huawei Technologies Co., Ltd", "", 1),
            source.replace("http://www.apache.org/licenses/LICENSE-2.0", "license URL omitted", 1),
            source.replace("# Licensed under", "Licensed under", 1),
            '"""\n' + source + '\n"""',
            "value = 1\n" + source,
            source.replace("Licensed under the Apache License, Version 2.0", "license omitted", 1),
            source.replace("limitations under the License.", "terms omitted", 1),
            "# padding\n" * 16 + source,
        )
        for content in invalid:
            with self.subTest(header=content[:100]):
                messages = self._check(content)
                self.assertEqual(len(messages), 1)
                self.assertEqual(messages[0].args[0], "missing-apache-license-header")
