"""Runs gm_robotics.ppo_rgb_gm unchanged, with PushToMarker-v1 / SortZones-v1 registered first."""
import runpy

import gm_robotics.newenvs  # noqa: F401

runpy.run_module("gm_robotics.ppo_rgb_gm", run_name="__main__")
