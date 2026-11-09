# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import yaml
import os

# Get the absolute path of this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

COMMON_CONFIG_FILE_PATHS = [
    os.path.join(SCRIPT_DIR, '.build_image_layers.yaml'),
    os.path.expandvars("$ISAAC_ROS_WS/../scripts/.build_image_layers.yaml"),
    # scripts/
    os.path.join(SCRIPT_DIR, '../../../.build_image_layers.yaml'),

    # Isaac ROS CLI
    '/etc/isaac-ros-cli/.build_image_layers.yaml'
]


def read_yaml(yaml_file):
    with open(yaml_file, 'r') as f:
        return yaml.safe_load(f)


def get_isaac_ros_common_config_path():
    for path in COMMON_CONFIG_FILE_PATHS:
        if os.path.exists(path):
            return path
    searched_paths = "\n  ".join(COMMON_CONFIG_FILE_PATHS)
    raise FileNotFoundError(
        "Could not find .build_image_layers.yaml. Searched:\n  "
        f"{searched_paths}"
    )


def get_isaac_ros_common_config_values(config_path):
    return read_yaml(config_path)


def get_build_order(build_order, env):
    env_ordered = []
    for build_key in build_order:
        if build_key in env:
            env_ordered.append(build_key)
    for build_key in env:
        if build_key not in env_ordered:
            env_ordered.append(build_key)
    return env_ordered
