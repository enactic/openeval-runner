# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for evaluator."""

import os
from pathlib import Path


TESTS_DIR = Path(__file__).parent
os.environ["DATAFLOW_FILE"] = str(TESTS_DIR / "dataflow.yaml")

from openeval_runner.evaluator import evaluate  # noqa: E402


def test_run(capfd):
    """evaluate() completes successfully."""
    job = {"job_id": 1, "job.docker_tag": "dummy"}
    assert evaluate(job)
