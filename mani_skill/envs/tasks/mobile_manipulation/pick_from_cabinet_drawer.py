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
    max_episode_steps=200
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
        PACKAGE_ASSET_DIR / "partnet_mobility/meta/info_pick_from_cabinet_drawer.json"  # change this, maybe filter some cabinets that are not suitable for this task
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
    
    def _load_cabinets(self, joint_types: List[str]):
        # we sample random cabinet model_ids with numpy as numpy is always deterministic based on seed, regardless of
        # GPU/CPU simulation backends. This is useful for replaying demonstrations.
        model_ids = self._batched_episode_rng.choice(self.all_model_ids)
        link_ids = self._batched_episode_rng.randint(0, 2**31)

        self._cabinets = []
        handle_links: List[List[Link]] = []
        handle_links_meshes: List[List[trimesh.Trimesh]] = []
        for i, model_id in enumerate(model_ids):
            # partnet-mobility is a dataset source and the ids are the ones we sampled
            # we provide tools to easily create the articulation builder like so by querying
            # the dataset source and unique ID
            cabinet_builder = articulations.get_articulation_builder(
                self.scene, f"partnet-mobility:{model_id}"
            )
            cabinet_builder.set_scene_idxs(scene_idxs=[i])
            cabinet_builder.initial_pose = sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0])
            cabinet = cabinet_builder.build(name=f"{model_id}-{i}")
            self.remove_from_state_dict_registry(cabinet)
            # this disables self collisions by setting the group 2 bit at CABINET_COLLISION_BIT all the same
            # that bit is also used to disable collision with the ground plane
            for link in cabinet.links:
                link.set_collision_group_bit(
                    group=2, bit_idx=CABINET_COLLISION_BIT, bit=1
                )
            self._cabinets.append(cabinet)
            handle_links.append([])
            handle_links_meshes.append([])

            # TODO (stao): At the moment code for selecting semantic parts of articulations
            # is not very simple. Will be improved in the future as we add in features that
            # support part and mesh-wise annotations in a standard querable format
            for link, joint in zip(cabinet.links, cabinet.joints):
                if joint.type[0] in joint_types:
                    handle_links[-1].append(link)
                    # save the first mesh in the link object that correspond with a handle
                    handle_links_meshes[-1].append(
                        link.generate_mesh(
                            filter=lambda _, render_shape: "handle"
                            in render_shape.name,
                            mesh_name="handle",
                        )[0]
                    )

        # we can merge different articulations/links with different degrees of freedoms into a single view/object
        # allowing you to manage all of them under one object and retrieve data like qpos, pose, etc. all together
        # and with high performance. Note that some properties such as qpos and qlimits are now padded.
        self.cabinet = Articulation.merge(self._cabinets, name="cabinet")
        self.add_to_state_dict_registry(self.cabinet)
        self.handle_link = Link.merge(
            # [links[link_ids[i] % len(links)] for i, links in enumerate(handle_links)],
            [handle_links[0][0] for links in handle_links],
            name="handle_link",
        )
        # store the position of the handle mesh itself relative to the link it is apart of
        self.handle_link_pos = common.to_tensor(
            np.array(
                [
                    # meshes[link_ids[i] % len(meshes)].bounding_box.center_mass
                    # for i, meshes in enumerate(handle_links_meshes)
                    meshes[-1].bounding_box.center_mass
                    for meshes in handle_links_meshes
                ]
            ),
            device=self.device,
        )

        self.handle_link_goal = actors.build_sphere(
            self.scene,
            radius=0.02,
            color=[0, 1, 0, 1],
            name="handle_link_goal",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]),
        )

    def _after_reconfigure(self, options):
        super()._after_reconfigure(options)
        self.cube_zs = []
        self.cube_goal_zs = []
        self.cube_xy_limits = []
        # TODO: check the small gap, by opening the drawer, the cube should be placed inside the drawer
        gap = 0.01
        shrink = 0.8    # shrink the xy limits, avoid too close to the edge
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
            self.cube_xy_limits.append(
                collision_mesh.bounding_box.bounds[:, :2] * shrink
            )
        self.cube_zs = common.to_tensor(self.cube_zs, device=self.device)
        self.cube_goal_zs = common.to_tensor(self.cube_goal_zs, device=self.device)
        self.cube_xy_limits = common.to_tensor(self.cube_xy_limits, device=self.device)
    
    def _initialize_episode(self, env_idx, options):
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)
            xyz = torch.zeros((b, 3))
            # the cube xy should be placed inside the drawer
            xy_scale = self.cube_xy_limits[env_idx, 1, :] - self.cube_xy_limits[env_idx, 0, :]
            xy_min = self.cube_xy_limits[env_idx, 0, :]
            xyz[:, :2] = torch.rand((b, 2)) * xy_scale + xy_min
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
            "cube_pos": self.cube.pose.p,
            "is_grasped": is_grasped,
            "cube_height": self.cube.pose.p[:, 2],
            "high_enough": high_enough,
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
        # reach handle reward
        tcp_to_handle_dist = torch.linalg.norm(
            self.agent.tcp.pose.p - info["handle_link_pos"], axis=1
        )
        reach_handle_reward = 1 - torch.tanh(5 * tcp_to_handle_dist)
        # open reward
        amount_to_open_left = torch.div(
            self.target_qpos - self.handle_link.joint.qpos, self.target_qpos
        )
        open_reward = 2 * (1 - amount_to_open_left)
        reach_handle_reward[
            amount_to_open_left < 0.999
        ] = 2  # if joint opens even a tiny bit, we don't need reach reward anymore
        open_reward[info["open_enough"]] = 3  # give max reward here
        # reach cube reward
        tcp_to_cube_dist = torch.linalg.norm(
            self.agent.tcp.pose.p - info["cube_pos"], axis=1
        )
        reach_cube_reward = 1 - torch.tanh(5 * tcp_to_cube_dist)
        reach_cube_reward[info["is_grasped"]] = 2   # once grasped, we don't need reach reward anymore
        # lift cube reward
        cube_to_goal_dist = torch.abs(self.cube_goal_zs - info["cube_height"])
        lift_cube_reward = 1 - torch.tanh(5 * cube_to_goal_dist)
        lift_cube_reward[info["high_enough"]] = 2
        # print(open_reward.shape)
        reward = reach_handle_reward + open_reward + (reach_cube_reward + lift_cube_reward) * info["open_enough"]
        reward[info["success"]] = 9.0
        return reward
    
    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        max_reward = 9.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward