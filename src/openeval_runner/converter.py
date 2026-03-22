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

"""Convert recorder output to rerun.io `.rrd` format."""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import rerun as rr
import rerun.blueprint as rrb

from openeval_runner.config import logger, RECORDER_BASE_DIRECTORY

ARM_DATA_CATEGORIES = [
    ("action", "left_arm"),
    ("action", "right_arm"),
    ("obs", "left_arm"),
    ("obs", "right_arm"),
]


def _dataset_dir(job):
    return Path(RECORDER_BASE_DIRECTORY, job["id"])


def _arm_entity_path(episode, category, side):
    return f"ep{episode}/{category}/{side}"


def _camera_entity_path(episode, name):
    return f"ep{episode}/camera/{name}"


def _find_camera_names(dataset_dir):
    episode_data_dir = Path(dataset_dir, "episodes/0")
    return sorted(dir.name for dir in episode_data_dir.glob("*_image"))


def _build_blueprint(dataset_dir):
    # Blueprint controls the layout and appearance of views.

    # TODO: Support for multi-episode
    episode = 0

    camera_views = [
        rrb.Spatial2DView(
            origin=_camera_entity_path(episode, name), name=f"camera/{name}"
        )
        for name in _find_camera_names(dataset_dir)
    ]
    action_views = []
    obs_views = []
    for category, side in ARM_DATA_CATEGORIES:
        entity_path = _arm_entity_path(episode, category, side)
        if category == "action":
            action_views.append(
                rrb.TimeSeriesView(
                    origin=entity_path, name=f"{entity_path}/actions/{side}"
                )
            )
        else:
            obs_views.append(
                rrb.TimeSeriesView(origin=entity_path, name=f"{entity_path}/obs/{side}")
            )
    return rrb.Horizontal(
        rrb.Vertical(*camera_views),
        rrb.Vertical(*action_views),
        rrb.Vertical(*obs_views),
        column_shares=[0.3, 0.35, 0.35],
    )


def _log_arm_positions(rec, dataset_dir, episode, category, side):
    table = pq.read_table(
        Path(dataset_dir, "episodes", str(episode), category, side, "qpos.parquet")
    )
    # nanoseconds to float seconds
    timestamps = np.array(
        table.column("timestamp").cast(pa.int64()).to_pylist(), dtype="datetime64[ns]"
    )
    # shape: (T, n_joints)
    positions_arr = np.array(table.column("positions").to_pylist())

    # Iterate over joints to log each one as a separate entity.
    entity_path = _arm_entity_path(episode, category, side)
    n_joints = positions_arr.shape[1]
    for i in range(n_joints):
        rr.send_columns(
            f"{entity_path}/joint{i}",
            indexes=[rr.TimeColumn("timestamp", timestamp=timestamps)],
            columns=rr.Scalars.columns(scalars=positions_arr[:, i]),
            recording=rec,
        )


def _log_camera_images(rec, dataset_dir, episode):
    camera_images_dir = sorted(
        Path(dataset_dir, "episodes", str(episode)).glob("*_image")
    )
    for image_dir in camera_images_dir:
        for path in image_dir.iterdir():
            try:
                rr.set_time(
                    "timestamp",
                    timestamp=np.datetime64(int(path.stem), "ns"),
                    recording=rec,
                )
            except ValueError:
                logger.warning("Could not parse timestamp from filename: %s", path)

            entity_path = _camera_entity_path(episode, image_dir.name)
            rr.log(entity_path, rr.EncodedImage(path=str(path)), recording=rec)


def _log_episode(rec, dataset_dir, episode):
    episode_data_dir = Path(dataset_dir, "episodes/0")
    if not episode_data_dir.exists():
        logger.warning("episode_data_dir not found: %s", episode_data_dir)
        return

    for category, side in ARM_DATA_CATEGORIES:
        _log_arm_positions(rec, dataset_dir, episode, category, side)

    _log_camera_images(rec, dataset_dir, episode)


def convert(job):
    """Convert recorder output to rerun.io `.rrd` format."""
    dataset_dir = _dataset_dir(job)
    rrd_path = Path(dataset_dir, "result.rrd")

    rec = rr.RecordingStream(application_id="OpenEval")
    rec.save(str(rrd_path), default_blueprint=_build_blueprint(dataset_dir))

    # TODO: Support for multi-episode
    _log_episode(rec, dataset_dir, 0)

    return rrd_path
