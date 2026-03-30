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

"""Evaluate a policy server."""

import os
import signal
import subprocess
import time

from openeval_runner.config import logger, DATAFLOW_FILE, RECORDER_BASE_DIRECTORY

NODE_NAME_PATTERN = "dora-openarm|opencv-video-capture"


def _kill(pgid, sig, fallback):
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, OSError):
        try:
            fallback()
        except OSError as err:
            logger.debug("kill failed: %s", err)


def _kill_process(proc):
    if proc.poll() is not None:
        return

    pid = proc.pid
    logger.info("Killing dora process (pid=%d)", pid)

    for sig, fallback, timeout in [
        (signal.SIGTERM, proc.terminate, 5),
        (signal.SIGKILL, proc.kill, 3),
    ]:
        _kill(pid, sig, fallback)
        try:
            proc.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            logger.warning("kill_process: dora did not exit after %s", sig.name)


def _pgrep():
    try:
        subprocess.run(
            ["pgrep", "-f", NODE_NAME_PATTERN], check=True, capture_output=True
        )
        return True
    except subprocess.CalledProcessError as err:
        logger.debug("pgrep failed: %s", err)
        return False


def _pkill(sig):
    cmd = ["pkill", f"-{sig.value}", "-f", NODE_NAME_PATTERN]
    try:
        subprocess.run(cmd, timeout=5, check=True, capture_output=True)
    except subprocess.CalledProcessError as err:
        logger.debug("pkill failed: %s", err)


def _kill_orphaned_workers():
    if not _pgrep():
        return
    _pkill(signal.SIGTERM)
    if not _pgrep():
        return
    time.sleep(2)
    _pkill(signal.SIGKILL)


def _run(phase, job, env, timeout):
    cmd = ["dora", "run", DATAFLOW_FILE, "--uv"]
    logger.info("[job=%s] %s: %s", job["id"], phase, " ".join(cmd))

    proc = None
    try:
        proc = subprocess.Popen(cmd, env=env, start_new_session=True)
        returncode = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.warning("[job=%s] %s timed out after %ds", job["id"], phase, timeout)
        return False
    except (OSError, subprocess.SubprocessError):
        logger.exception("[job=%s] %s: failed to run dora", job["id"], phase)
        return False
    finally:
        if proc is not None:
            _kill_process(proc)
        _kill_orphaned_workers()

    logger.info("[job=%s] %s finished: returncode=%d", job["id"], phase, returncode)
    return returncode == 0


def evaluate(job):
    """Evaluate a policy server."""
    # TODO: optimal timeout
    timeout = 90

    env = os.environ.copy() | {
        "IMAGE": job["job.docker_tag"],
        "DIRECTORY": RECORDER_BASE_DIRECTORY,
        "NAME": f"evaluate-{job['id']}",
        "TIMEOUT": str(timeout),
    }
    return _run("evaluate", job, env, timeout=timeout)


def reset(job):
    """Reset the evaluation environment."""
    # TODO: optimal timeout
    timeout = 90

    env = os.environ.copy() | {
        "IMAGE": job["task.reset_docker_tag"],
        "DIRECTORY": RECORDER_BASE_DIRECTORY,
        "NAME": f"reset-{job['id']}",
        "TIMEOUT": str(timeout),
    }
    return _run("reset", job, env, timeout=timeout)
