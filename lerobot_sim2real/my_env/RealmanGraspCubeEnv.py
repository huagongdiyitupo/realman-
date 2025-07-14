from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Sequence, Union
import copy

import dacite
import numpy as np
import sapien
import torch
from sapien.render import RenderBodyComponent
from transforms3d.euler import euler2quat

import mani_skill.envs.utils.randomization as randomization
from mani_skill.envs.tasks.digital_twins.base_env import BaseDigitalTwinEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.logging_utils import logger
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig

import os
os.environ["WANDB_API_KEY"] = "bfbdcd39bffb203ce3d36f2af24d861e4a1a6ccc"
os.environ["WANDB_MODE"] = "offline"

# 注册realman双臂机器人
from mani_skill.agents.registration import register_agent
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
@register_agent()
class Realman(BaseAgent):
    uid = "Realman"
    urdf_path = "/home/lenovo/jdp_workspace/dm_teleoperation/daimon_joycon_teleoperation/daimon/dm_dual_arm_description_with_franka.urdf"

    # 定义关键帧，初始化机械臂关节角
    keyframes = dict(
        rest=Keyframe(
            qpos=np.array([
                0, 1.83956963 , 0,1.17592755 , 0,1.5852669 , 0,-1.52353741 ,0,-0.37200622 ,0,-0.36985699 ,0,-0.14799367,
                0,0
            ]),
            pose=sapien.Pose(),
        )
    )
    
    # arm_joint_names = [
    #     "left_joint1", "left_joint2", "left_joint3", "left_joint4", 
    #     "left_joint5", "left_joint6", "left_joint7",
    #     "right_joint1", "right_joint2", "right_joint3", "right_joint4", 
    #     "right_joint5", "right_joint6", "right_joint7",
    # ]
    arm_joint_names = [
        "left_joint1", "left_joint2", "left_joint3", "left_joint4",
        "left_joint5", "left_joint6", "left_joint7",
    ]

    gripper_joint_names = [
        "panda_finger_joint1",
        "panda_finger_joint2",
    ]

    arm_stiffness = [1e3]*7
    arm_damping = [1e2] * 7
    arm_force_limit = 100
    
    gripper_stiffness = 1e3
    gripper_damping = 1e2
    gripper_force_limit = 100

    @property
    def _controller_configs(self):
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=None,
            upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )
        
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            use_delta=True,
        )

        gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
            self.gripper_joint_names,
            lower=-0.01,  # a trick to have force when the object is thin
            upper=0.04,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            mimic={"panda_finger_joint1": {"joint": "panda_finger_joint2"}},
        )
                # , gripper=gripper_pd_joint_pos
        controller_configs = dict(
            pd_joint_delta_pos=dict(arm=arm_pd_joint_delta_pos, gripper=gripper_pd_joint_pos),
            pd_joint_pos=dict(arm=arm_pd_joint_pos, gripper=gripper_pd_joint_pos),
        )

        return deepcopy_dict(controller_configs)
    
    # @property
    # def _sensor_configs(self):
    #     return [
    #         CameraConfig(
    #             uid="hand_camera",
    #             pose=sapien.Pose(p=[-0.03, 0, -0.03], q=[0.8660254, 0.0, 0.5, 0.0]),
    #             width=128,
    #             height=128,
    #             fov=np.pi / 2,
    #             near=0.01,
    #             far=100,
    #             mount=self.robot.links_map["ls_camera2_link"],
    #         ),
    #         CameraConfig(
    #             uid="wrist_camera",
    #             pose=sapien.Pose(p=[-0.03, 0, 0.03], q=euler2quat(0, -np.pi/2, 0)),
    #             width=128,
    #             height=128,
    #             fov=np.pi / 2,
    #             near=0.01,
    #             far=100,
    #             mount=self.robot.links_map["left_Link7"],
    #         )
    #     ]
    
    def _after_loading_articulation(self):
        super()._after_loading_articulation()
        self.tcp_link = self.robot.links_map["panda_hand_tcp"]  # 专门的TCP点
        self.finger1_link = self.robot.links_map["panda_leftfinger"]
        self.finger2_link = self.robot.links_map["panda_rightfinger"]
        self.finger1_tip = self.finger1_link  # 如果没有单独的tip，就用finger本身
        self.finger2_tip = self.finger2_link

    @property
    def tcp_pos(self):
        return self.tcp_link.pose.p

    @property
    def tcp_pose(self):
        return self.tcp_link.pose

    
    def is_grasping(self, object: Actor, min_force=0.5, max_angle=110):
        """Check if the robot is grasping an object

        Args:
            object (Actor): The object to check if the robot is grasping
            min_force (float, optional): Minimum force before the robot is considered to be grasping the object in Newtons. Defaults to 0.5.
            max_angle (int, optional): Maximum angle of contact to consider grasping. Defaults to 85.
        """
        l_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger1_link, object
        )
        r_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger2_link, object
        )
        lforce = torch.linalg.norm(l_contact_forces, axis=1)
        rforce = torch.linalg.norm(r_contact_forces, axis=1)

        # direction to open the gripper
        ldirection = self.finger1_link.pose.to_transformation_matrix()[..., :3, 1]
        rdirection = -self.finger2_link.pose.to_transformation_matrix()[..., :3, 1]
        langle = common.compute_angle_between(ldirection, l_contact_forces)
        rangle = common.compute_angle_between(rdirection, r_contact_forces)
        lflag = torch.logical_and(
            lforce >= min_force, torch.rad2deg(langle) <= max_angle
        )
        rflag = torch.logical_and(
            rforce >= min_force, torch.rad2deg(rangle) <= max_angle
        )
        return torch.logical_and(lflag, rflag)
    
# there are many ways to parameterize an environment's domain randomization. This is a simple way to do it
# with dataclasses that can be created and modified by the user and passed into the environment constructor.
@dataclass
class RealmanGraspCubeDomainRandomizationConfig:
    ### task agnostic domain randomizations, many of which you can copy over to your own tasks ###
    initial_qpos_noise_scale: float = 0.02
    robot_color: Optional[Union[str, Sequence[float]]] = None
    """Color of the robot in RGB format in scale of 0 to 1 mapping to 0 to 255.
    If you want to randomize it just set this value to "random". If left as None which is
    the default, it will set the robot parts to white and motors to black. For more fine-grained choices on robot colors you need to modify
    mani_skill/assets/robots/so100/so100.urdf in the ManiSkill package."""
    randomize_lighting: bool = True
    max_camera_offset: Sequence[float] = (0.025, 0.025, 0.025)
    """max camera offset from the base camera position in x, y, and z axes"""
    camera_target_noise: float = 1e-3
    """scale of noise added to the camera target position"""
    camera_view_rot_noise: float = 5e-3
    """scale of noise added to the camera view rotation"""
    camera_fov_noise: float = np.deg2rad(2)
    """scale of noise added to the camera fov"""

    ### task-specific related domain randomizations that occur during scene loading ###
    # cube_half_size_range: Sequence[float] = (0.022 / 2, 0.028 / 2)
    cube_half_size_range: Sequence[float] = (0.044 / 2, 0.056 / 2)
    cube_friction_mean: float = 0.3
    cube_friction_std: float = 0.05
    cube_friction_bounds: Sequence[float] = (0.1, 0.5)
    randomize_cube_color: bool = True

    def dict(self):
        return {k: v for k, v in asdict(self).items()}


@register_env("RealmanGraspCube-v1", max_episode_steps=64)
class RealmanGraspCubeEnv(BaseDigitalTwinEnv):
    """
    **Task Description:**
    A simple task where the objective is to grasp a cube with the Realman arm and bring it up to a target rest pose.

    **Randomizations:**
    - the cube's xy position is randomized on top of a table in a region of size [0.2, 0.2] x [-0.2, -0.2]. It is placed flat on the table
    - the cube's z-axis rotation is randomized to a random angle

    **Success Conditions:**
    - the cube is lifted, grasped, and the robot returns to a rest pose above the surface of the table
    """

    # _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PickCube-v1_rt.mp4"
    SUPPORTED_ROBOTS = ["Realman"]
    SUPPORTED_OBS_MODES = ["none", "state", "state_dict", "rgb+segmentation"]
    agent: Realman

    def __init__(
        self,
        *args,
        robot_uids="Realman",
        control_mode="pd_joint_pos",
        greenscreen_overlay_path=None,
        domain_randomization_config: Union[
            RealmanGraspCubeDomainRandomizationConfig, dict
        ] = RealmanGraspCubeDomainRandomizationConfig(),
        domain_randomization=True,
        base_camera_settings=dict(
            fov=52 * np.pi / 180,
            pos=[0.5, 0.3, 0.35],
            target=[0.3, 0.0, 0.1],
        ),
        # 修改物体随机中心点，需要在env_config中修改
        spawn_box_pos=[0.3, 0.05],
        spawn_box_half_size=0.2 / 2,
        **kwargs,
    ):
        self.domain_randomization = domain_randomization
        """whether randomization is turned on or off."""
        self.domain_randomization_config = RealmanGraspCubeDomainRandomizationConfig()
        """domain randomization config"""
        merged_domain_randomization_config = self.domain_randomization_config.dict()
        if isinstance(domain_randomization_config, dict):
            common.dict_merge(
                merged_domain_randomization_config, domain_randomization_config
            )
            self.domain_randomization_config = dacite.from_dict(
                data_class=RealmanGraspCubeDomainRandomizationConfig,
                data=domain_randomization_config,
                config=dacite.Config(strict=True),
            )
        self.base_camera_settings = base_camera_settings
        """what the camera fov, position and target are when domain randomization is off. DR is centered around these settings"""

        if greenscreen_overlay_path is None:
            logger.warning(
                "No greenscreen overlay path provided, no greenscreen will be used"
            )
            self.rgb_overlay_mode = "none"

        # set the camera called "base_camera" to use the greenscreen overlay when rendering
        else:
            self.rgb_overlay_paths = dict(base_camera=greenscreen_overlay_path)

        self.spawn_box_pos = spawn_box_pos
        self.spawn_box_half_size = spawn_box_half_size
        super().__init__(
            *args, robot_uids=robot_uids, control_mode=control_mode, **kwargs
        )

    @property
    def _default_sim_config(self):
        return SimConfig(sim_freq=100, control_freq=100)

    @property
    def _default_sensor_configs(self):
        # we just set a default camera pose here for now. For sim2real we will modify this during training accordingly.
        # note that we pass in the camera mount which is created in the _load_scene function later. This mount lets us
        # randomize camera poses at each environment step. Here we just randomize some camera configuration like fov.
        if self.domain_randomization:
            camera_fov_noise = self.domain_randomization_config.camera_fov_noise * (
                2 * self._batched_episode_rng.rand() - 1
            )
        else:
            camera_fov_noise = 0
        return [
            CameraConfig(
                uid="hand_camera",
                # pose=sapien.Pose(p=[-0.03, 0, -0.03], q=[0.8660254, 0.0, 0.5, 0.0]),
                pose=sapien.Pose(), 
                width=128,
                height=128,
                fov=camera_fov_noise + self.base_camera_settings["fov"],
                near=0.01,
                far=100,
                # mount=["ls_camera2_link"],
                mount=self.camera_mount,
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        # this camera and angle is simply used for visualization purposes, not policy observations
        # pose = sapien_utils.look_at([0.5, 0.3, 0.35], [0.3, 0.0, 0.1])
        pose = sapien_utils.look_at([0.5, 1.5, 0.8], [0.1, 0.6, 0.1])
        return CameraConfig(
            "render_camera", pose, 256, 256, 52 * np.pi / 180, 0.01, 100
        )

    def _load_agent(self, options: dict):
        # load the robot arm at this initial pose
        super()._load_agent(
            options,
            sapien.Pose(p=[-0.1, 0, -0.92], q=euler2quat(0, 0, 0)),
            build_separate=True
            if self.domain_randomization
            and self.domain_randomization_config.robot_color == "random"
            else False,
        )

    def _load_lighting(self, options: dict):
        if self.domain_randomization:
            if self.domain_randomization_config.randomize_lighting:
                ambient_colors = self._batched_episode_rng.uniform(0.2, 0.5, size=(3,))
                for i, scene in enumerate(self.scene.sub_scenes):
                    scene.render_system.ambient_light = ambient_colors[i]
        else:
            self.scene.set_ambient_light([0.3, 0.3, 0.3])
        self.scene.add_directional_light(
            [1, 1, -1], [1, 1, 1], shadow=False, shadow_scale=5, shadow_map_size=2048
        )
        self.scene.add_directional_light([0, 0, -1], [1, 1, 1])

    def _load_scene(self, options: dict):
        # we use a predefined table scene builder which simply adds a table and floor to the scene
        # where the 0, 0, 0 position is the center of the table
        self.table_scene = TableSceneBuilder(self)
        self.table_scene.build()

        # some default values for cube geometry
        half_sizes = (
            np.ones(self.num_envs)
            * (
                self.domain_randomization_config.cube_half_size_range[1]
                + self.domain_randomization_config.cube_half_size_range[0]
            )
            / 2
        )
        colors = np.zeros((self.num_envs, 3))
        colors[:, 0] = 1
        frictions = (
            np.ones(self.num_envs) * self.domain_randomization_config.cube_friction_mean
        )

        # randomize cube sizes, colors, and frictions
        if self.domain_randomization:
            # note that we use self._batched_episode_rng instead of torch.rand or np.random as it ensures even with a different number of parallel
            # environments the same seed leads to the same RNG, which is important for reproducibility as geometric changes here aren't saveable in environment state
            half_sizes = self._batched_episode_rng.uniform(
                low=self.domain_randomization_config.cube_half_size_range[0],
                high=self.domain_randomization_config.cube_half_size_range[1],
            )
            if self.domain_randomization_config.randomize_cube_color:
                colors = self._batched_episode_rng.uniform(low=0, high=1, size=(3,))
            frictions = self._batched_episode_rng.normal(
                self.domain_randomization_config.cube_friction_mean,
                self.domain_randomization_config.cube_friction_std,
            )
            frictions = frictions.clip(
                *self.domain_randomization_config.cube_friction_bounds
            )

        self.cube_half_sizes = common.to_tensor(half_sizes, device=self.device)
        colors = np.concatenate([colors, np.ones((self.num_envs, 1))], axis=-1)

        # build our cubes
        cubes = []
        for i in range(self.num_envs):
            # create a different cube in each parallel environment
            # using our randomized colors, frictions, and sizes
            builder = self.scene.create_actor_builder()
            friction = frictions[i]
            material = sapien.pysapien.physx.PhysxMaterial(
                static_friction=friction,
                dynamic_friction=friction,
                restitution=0,
            )
            builder.add_box_collision(
                half_size=[half_sizes[i]] * 3, material=material, density=200  # 25
            )
            builder.add_box_visual(
                half_size=[half_sizes[i]] * 3,
                material=sapien.render.RenderMaterial(
                    base_color=colors[i],
                ),
            )
            builder.initial_pose = sapien.Pose(p=[0, 0, half_sizes[i]])
            builder.set_scene_idxs([i])
            cube = builder.build(name=f"cube-{i}")
            cubes.append(cube)
            self.remove_from_state_dict_registry(cube)

        # since we are building many different cubes but simulating in parallel, we need to merge them into a single actor
        # so we can access each different cube's information with a single object
        self.cube = Actor.merge(cubes, name="cube")
        self.add_to_state_dict_registry(self.cube)

        # we want to only keep the robot and the cube in the render, everything else is greenscreened.
        self.remove_object_from_greenscreen(self.agent.robot)
        self.remove_object_from_greenscreen(self.cube)

        # a hardcoded initial joint configuration for the robot to start from
        self.rest_qpos = torch.tensor(
            Realman.keyframes["rest"].qpos,  # 使用 Realman 的初始关节角
            device=self.device,
        )
        # hardcoded pose for the table that places it such that the robot base is at 0 and on the edge of the table.
        self.table_pose = Pose.create_from_pq(
            p=[-0.12 + 0.737, 0, -0.9196429], q=euler2quat(0, 0, np.pi / 2)
        )

        # we build a 3rd-view camera mount to put cameras on which let us randomize camera poses at each timestep
        builder = self.scene.create_actor_builder()
        builder.initial_pose = sapien.Pose()
        self.camera_mount = builder.build_kinematic("camera_mount")

        # randomize or set a fixed robot color
        if self.domain_randomization_config.robot_color is not None:
            for link in self.agent.robot.links:
                for i, obj in enumerate(link._objs):
                    # modify the i-th object which is in parallel environment i
                    render_body_component: RenderBodyComponent = (
                        obj.entity.find_component_by_type(RenderBodyComponent)
                    )
                    if render_body_component is not None:
                        for render_shape in render_body_component.render_shapes:
                            for part in render_shape.parts:
                                if (
                                    self.domain_randomization
                                    and self.domain_randomization_config.robot_color
                                    == "random"
                                ):
                                    part.material.set_base_color(
                                        self._batched_episode_rng[i]
                                        .uniform(low=0.0, high=1.0, size=(3,))
                                        .tolist()
                                        + [1]
                                    )
                                else:
                                    part.material.set_base_color(
                                        list(
                                            self.domain_randomization_config.robot_color
                                        )
                                        + [1]
                                    )

    def sample_camera_poses(self, n: int):
        # a custom function to sample random camera poses
        # the way this works is we first sample "eyes", which are the camera positions
        # then we use the noised_look_at function to sample the full camera poses given the sampled eyes
        # and a target position the camera is pointing at
        if self.domain_randomization:
            # in case these haven't been moved to torch tensors on the environment device
            self.base_camera_settings["pos"] = common.to_tensor(
                self.base_camera_settings["pos"], device=self.device
            )
            self.base_camera_settings["target"] = common.to_tensor(
                self.base_camera_settings["target"], device=self.device
            )
            self.domain_randomization_config.max_camera_offset = common.to_tensor(
                self.domain_randomization_config.max_camera_offset, device=self.device
            )

            eyes = randomization.camera.make_camera_rectangular_prism(
                n,
                scale=self.domain_randomization_config.max_camera_offset,
                center=self.base_camera_settings["pos"],
                theta=0,
                device=self.device,
            )
            return randomization.camera.noised_look_at(
                eyes,
                target=self.base_camera_settings["target"],
                look_at_noise=self.domain_randomization_config.camera_target_noise,
                view_axis_rot_noise=self.domain_randomization_config.camera_view_rot_noise,
                device=self.device,
            )
        else:
            return sapien_utils.look_at(
                eye=self.base_camera_settings["pos"],
                target=self.base_camera_settings["target"],
            )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # we randomize the pose of the cube accordingly so that the policy can learn to pick up the cube from
        # many different orientations and positions.
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            # move the table back so that the robot is at 0 and on the edge of the table.
            self.table_scene.table.set_pose(self.table_pose)

            # sample a random initial joint configuration for the robot
            self.agent.robot.set_qpos(
                self.rest_qpos + torch.randn(size=(b, self.rest_qpos.shape[-1])) * 0.02
            )
            self.agent.robot.set_pose(
                Pose.create_from_pq(p=[-0.1, 0, -0.92], q=euler2quat(0, 0, 0))
            )

            # initialize the cube at a random position and rotation around the z-axis
            spawn_box_pos = self.agent.robot.pose.p + torch.tensor(
                [self.spawn_box_pos[0], self.spawn_box_pos[1], 0]
            )
            xyz = torch.zeros((b, 3))
            xyz[:, :2] = (
                torch.rand((b, 2)) * self.spawn_box_half_size * 2
                - self.spawn_box_half_size
            )
            xyz[:, :2] += spawn_box_pos[env_idx, :2]
            xyz[:, 2] = self.cube_half_sizes[env_idx]
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.cube.set_pose(Pose.create_from_pq(xyz, qs))

            # randomize the camera poses
            self.camera_mount.set_pose(self.sample_camera_poses(n=b))

    def _before_control_step(self):
        # update the camera poses before agent actions are executed
        if self.domain_randomization:
            self.camera_mount.set_pose(self.sample_camera_poses(n=self.num_envs))
            if self.gpu_sim_enabled:
                self.scene._gpu_apply_all()

    # 获取 qpos 和 controller 状态
    def _get_obs_agent(self):
        # the default get_obs_agent function in ManiSkill records qpos and qvel. However
        # Realman arm qvel are likely too noisy to learn from and not implemented.
        obs = dict(qpos=self.agent.robot.get_qpos())
        controller_state = self.agent.controller.get_state()
        if len(controller_state) > 0:
            obs.update(controller=controller_state)
        return obs

    def _get_obs_extra(self, info: Dict):
        # we ensure that the observation data is always retrievable in the real world, using only real world
        # available data (joint positions or the controllers target joint positions in this case).
        
        # obs = dict(
        #     dist_to_rest_qpos=self.agent.controller.controllers["arm"]._target_qpos[:, :-1]
        #     - self.rest_qpos[:-1],
        # )
        # obs = dict(
        #     dist_to_rest_qpos=self.agent.controller.controllers["arm"]._target_qpos[:, :-2]
        #     - self.rest_qpos[:self.agent.controller.controllers["arm"]._target_qpos[:, :-2]],
        # )
        obs = dict(
            dist_to_rest_qpos=self.agent.controller.controllers["arm"]._target_qpos[:, :]
            - self.rest_qpos[:14][1::2],
        )
        if self.obs_mode_struct.state:
            # state based policies can gain access to more information that helps learning
            obs.update(
                is_grasped=info["is_grasped"],
                obj_pose=self.cube.pose.raw_pose,
                tcp_pos=self.agent.tcp_pos,
                tcp_to_obj_pos=self.cube.pose.p - self.agent.tcp_pos,
            )
        return obs

    def evaluate(self):
        # evaluation function to generate some useful metrics/flags and evaluate the success of the task

        tcp_to_obj_dist = torch.linalg.norm(
            self.cube.pose.p - self.agent.tcp_pos,
            axis=-1,
        )
        reached_object = tcp_to_obj_dist < 0.03
        is_grasped = self.agent.is_grasping(self.cube)
        # print(f"测试：{dir(self.agent.controller)}")
        target_qpos = self.agent.controller.controllers["arm"]._target_qpos.clone()

        # 这里target_qpos是7维，所以取rest_qpos的左臂=qpos[:14][1::2]
        rest_qpos = self.rest_qpos[:14][1::2]
        distance_to_rest_qpos = torch.linalg.norm(
            target_qpos - rest_qpos, axis=-1
        )
        reached_rest_qpos = distance_to_rest_qpos < 0.2

        cube_lifted = self.cube.pose.p[..., -1] >= (self.cube_half_sizes + 1e-3)

        success = cube_lifted & is_grasped & reached_rest_qpos

        # determine if robot is touching the table. for safety reasons we want the robot to avoid hitting the table when grasping the cube
        l_contact_forces = self.scene.get_pairwise_contact_forces(
            self.agent.finger1_link, self.table_scene.table
        )
        r_contact_forces = self.scene.get_pairwise_contact_forces(
            self.agent.finger2_link, self.table_scene.table
        )
        lforce = torch.linalg.norm(l_contact_forces, axis=1)
        rforce = torch.linalg.norm(r_contact_forces, axis=1)
        touching_table = torch.logical_or(
            lforce >= 1e-2,
            rforce >= 1e-2,
        )
        return {
            "is_grasped": is_grasped,
            "reached_object": reached_object,
            "distance_to_rest_qpos": distance_to_rest_qpos,
            "touching_table": touching_table,
            "cube_lifted": cube_lifted,
            "success": success,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # note the info object is the data returned by the evaluate function. We can reuse information
        # to save compute time.
        # this reward function essentially has two stages, before and after grasping.
        # in stage 1 we reward the robot for reaching the object and grasping it.
        # in stage 2 if the robot is grasping the object, we reward it for controlling the robot returning to the predefined rest pose.
        # in all stages we penalize the robot for touching the table.
        # this reward function is very simple and can easily be improved to learn more robust behaviors or solve more complex problems.

        tcp_to_obj_dist = torch.linalg.norm(
            self.cube.pose.p - self.agent.tcp_pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = reaching_reward + info["is_grasped"]
        place_reward = torch.exp(-2 * info["distance_to_rest_qpos"])
        reward += place_reward * info["is_grasped"]
        reward -= 2 * info["touching_table"].float()
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        # for more stable RL we often also permit defining a noramlized reward function where you manually scale the reward down by its max value like so
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 3
