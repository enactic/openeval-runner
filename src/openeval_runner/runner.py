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

"""Evaluates queued jobs in an OpenArm Cell as a daemon."""

import time

import openarm_driver

from openeval_runner import converter, evaluator, job_client
from openeval_runner.config import logger, settings


def _stop_arms():
    for side in ("left_arm", "right_arm"):
        try:
            openarm_driver.SingleArmDriver(side).stop()
        except Exception:
            logger.exception("[arm=%s] stop failed", side)


def run_job(job):
    """Execute a single job."""
    logger.debug("[job=%s] started", job["job_id"])

    try:
        try:
            success = evaluator.evaluate(job)
            evaluator.reset(job)
        finally:
            _stop_arms()

        rrd_path = converter.convert(job)
        s3_key = job_client.upload_rrd(rrd_path)
        job_client.complete_job(job["job_id"], success, s3_key)

        logger.debug("[job=%s] completed", job["job_id"])
    except Exception as err:
        logger.exception("[job=%s] failed", job["job_id"])
        job_client.fail_job(job["job_id"], str(err))


def main():
    """Poll for jobs and executes them."""
    logger.info("started (poll_interval=%ds)", settings.POLL_INTERVAL)
    while True:
        job = job_client.fetch_next()
        if job is None:
            time.sleep(settings.POLL_INTERVAL)
            continue
        run_job(job)


if __name__ == "__main__":
    main()
