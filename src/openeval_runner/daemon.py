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

"""OpenEval Runner daemon: poll for jobs and executes them."""

import time

from openeval_runner import converter, dora_runner, job_client

POLL_INTERVAL = 3


def run_job(job):
    """Execute a single job."""
    # TODO
    print("DEBUG: job start.")

    dora_runner.run_eval(job)
    dora_runner.run_reset(job)

    rrd_path = converter.convert(job)
    job_client.upload_rrd(job, rrd_path)

    print("DEBUG: job end.")


def main():
    """Poll for jobs and executes them."""
    # TODO
    print("DEBUG: daemon start.")
    while True:
        job = job_client.fetch_next()
        if job is None:
            print("waiting")
            time.sleep(POLL_INTERVAL)
            continue

        run_job(job)


if __name__ == "__main__":
    main()
