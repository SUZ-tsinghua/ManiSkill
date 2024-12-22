from typing import Any, Dict, List, Optional, Union

import numpy as np
import sapien
import sapien.physx as physx
import torch
import trimesh

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.robots import Fetch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.tasks.mobile_manipulation.open_cabinet_drawer import OpenCabinetDrawerEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors, articulations
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.geometry.geometry import transform_points
from mani_skill.utils.io_utils import load_json
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Articulation, Link, Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig

CABINET_COLLISION_BIT = 29


@register_env(
    "PickFromCabinetDrawer-v1", 
    max_episode_steps=100
)
class PickFromCabinetDrawer(OpenCabinetDrawerEnv):
    """
    **Task Description:**
    Open the cabinet drawer and pick up the object inside.

    **Randomization:**
    - Robot is randomly initialized 1.6 to 1.8 meters away from the door and positioned to face it
    - Robot's base orientation is randomized by -9 to 9 degrees
    - Object is randomly placed inside the drawer
    - The drawer to open is randomly sampled from all drawers available to open

    **Success Conditions:**
    - The object is picked above the drawer
    """
    
    SUPPORTED_ROBOTS = ["fetch"]
    agent: Union[Fetch]
    handle_types = ["prismatic"]
    TRAIN_JSON = (
        PACKAGE_ASSET_DIR / "partnet_mobility/meta/info_cabinet_drawer_train.json"  # TODO: change this, maybe filter some cabinets that are not suitable for this task
    )
    
    cube_half_size = 0.02
    min_open_frac = 0.75

    def __init__(
        self, 
        *args, 
        robot_uids="fetch",
        robot_init_qpos_noise=0.02,
        reconfiguration_freq=None,
        num_envs=1,
        **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        train_data = load_json(self.TRAIN_JSON)
        self.all_model_ids = np.array(list(train_data.keys()))
        # self.all_model_ids = np.array(["1004", "1004"])
        if reconfiguration_freq is None:
            # if not user set, we pick a number
            if num_envs == 1:
                reconfiguration_freq = 1
            else:
                reconfiguration_freq = 0
        super(OpenCabinetDrawerEnv, self).__init__(
            *args,
            robot_uids=robot_uids,
            reconfiguration_freq=reconfiguration_freq,
            num_envs=num_envs,
            **kwargs,
        )

    def _load_scene(self, options: dict):
        self.ground = build_ground(self.scene)
        # temporarily turn off the logging as there will be big red warnings
        # about the cabinets having oblong meshes which we ignore for now.
        sapien.set_log_level("off")
        self._load_cabinets(self.handle_types)
        sapien.set_log_level("warn")
        # build the cube
        self.cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[1, 0, 0, 1],
            name="cube",
            initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
        )
        from mani_skill.agents.robots.fetch import FETCH_WHEELS_COLLISION_BIT

        self.ground.set_collision_group_bit(
            group=2, bit_idx=FETCH_WHEELS_COLLISION_BIT, bit=1
        )
        self.ground.set_collision_group_bit(
            group=2, bit_idx=CABINET_COLLISION_BIT, bit=1
        )

    def _after_reconfigure(self, options):
        super()._after_reconfigure(options)
        self.cube_zs = []
        self.cube_goal_zs = []
        # TODO: check the small gap, by opening the drawer, the cube should be placed inside the drawer
        gap = 0.01
        # put the cube inside the drawer
        # already test: +cube_half_size makes the cube on the surface of the drawer
        for cabinet in self._cabinets:
            collision_mesh = cabinet.get_first_collision_mesh()
            self.cube_zs.append(
                collision_mesh.bounding_box.bounds[1, 2] \
                -collision_mesh.bounding_box.bounds[0, 2] \
                -self.cube_half_size \
                -gap   # add a small gap
            )
            self.cube_goal_zs.append(
                collision_mesh.bounding_box.bounds[1, 2] \
                -collision_mesh.bounding_box.bounds[0, 2] \
                +self.cube_half_size \
                +gap   # add a small gap
            )
        self.cube_zs = common.to_tensor(self.cube_zs, device=self.device)
        self.cube_goal_zs = common.to_tensor(self.cube_goal_zs, device=self.device)
    
    def _initialize_episode(self, env_idx, options):
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)
            xyz = torch.zeros((b, 3))
            xyz[:, :2] = torch.rand((b, 2)) * 0.2 - 0.1
            xyz[:, 2] = self.cube_zs[env_idx]
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.cube.set_pose(Pose.create_from_pq(xyz, qs))

    def evaluate(self):
        # even though self.handle_link is a different link across different articulations
        # we can still fetch a joint that represents the parent joint of all those links
        # and easily get the qpos value.
        open_enough = self.handle_link.joint.qpos >= self.target_qpos
        handle_link_pos = self.handle_link_positions()

        link_is_static = (
            torch.linalg.norm(self.handle_link.angular_velocity, axis=1) <= 1
        ) & (torch.linalg.norm(self.handle_link.linear_velocity, axis=1) <= 0.1)

        is_grasped = self.agent.is_grasping(self.cube)
        high_enough = self.cube.pose.p[:, 2] >= self.cube_goal_zs
        return {
            "success": open_enough & link_is_static & is_grasped & high_enough,
            "handle_link_pos": handle_link_pos,
            "open_enough": open_enough,
            "is_grasped": is_grasped,
            "cube_height": self.cube.pose.p[:, 2],
        }
    
    def _get_obs_extra(self, info):
        obs = super()._get_obs_extra(info)

        if "cube" in self._hidden_objects:
            obs.update(
                obj_pose=self.cube.pose.raw_pose,
                tcp_to_obj_pos=self.cube.pose.p - self.agent.tcp.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # TODO
        return super().compute_dense_reward(obs, action, info)
    
    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        max_reward = 5.0    # TODO
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward