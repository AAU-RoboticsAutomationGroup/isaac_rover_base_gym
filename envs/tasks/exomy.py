import math
import numpy as np
import os
import torch
import xml.etree.ElementTree as ET
import random
from utils.torch_jit_utils import *
from tasks.base.vec_task import VecTask
#from utils.kinematics import Rover
import torchgeometry as tgm
from isaacgym import gymutil, gymtorch, gymapi
from scipy.spatial.transform import Rotation as R
from utils.kinematicsUpdated import Ackermann
from tasks.camera import camera
from isaacgym.terrain_utils import *


##################################################
# Implemented so far
##################################################
# Exomy_reward - 11/04 kl. 10:35 
    # Pos_reward
    # Vel_reward
# Exomy_terrain - 11/04 kl. 10:35
    # 1 uniform terrain instead of 8 different
# Exomy_heading - 17/04 kl. 23
    # Spawn new points


class Exomy(VecTask):

    def __init__(self, cfg, sim_device, graphics_device_id, headless):

        self.cfg = cfg
        #self.Kinematics = Rover()
        self.max_episode_length = self.cfg["env"]["maxEpisodeLength"]
        self.cfg["env"]["numObservations"] = 10
        self.cfg["env"]["numActions"] = 2
        self.max_effort_vel = 5.2
        self.max_effort_pos = math.pi/2
        super().__init__(config=self.cfg, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless)
        
        
        # Retrieves buffer for Actor root states.
        # position([0:3]), rotation([3:7]), linear velocity([7:10]), and angular velocity([10:13])
        # Buffer has shape (num_environments, num_actors * 13).
        dofs_per_env = 15

        self.root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        


        # Convert buffer to vector, one is created for the robot and for the marker.
        vec_root_tensor = gymtorch.wrap_tensor(self.root_tensor).view(self.num_envs, 2, 13)
        vec_dof_tensor = gymtorch.wrap_tensor(self.dof_state_tensor).view(self.num_envs, dofs_per_env, 2)
        #print(vec_dof_tensor)
        # Position vector for robot
        self.root_states = vec_root_tensor[:, 0, :]
        self.root_positions = self.root_states[:, 0:3]
        # Rotation of robot
        self.root_quats = self.root_states[:, 3:7]
        # Linear Velocity of robot
        self.root_linvels = self.root_states[:, 7:10]
        # Angular Velocity of robot
        self.root_angvels = self.root_states[:, 10:13]
        

        self.target_root_positions = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self.target_root_positions[:, 2] = 0

        #print(self.target_root_positions)

        # Marker position
        self.marker_states = vec_root_tensor[:, 1, :]
        self.marker_positions = self.marker_states[:, 0:3]


        # self.dof_states = vec_dof_tensor
        # self.dof_positions = vec_dof_tensor[..., 0]
        # self.dof_velocities = vec_dof_tensor[..., 1]
        
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_states = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_positions = self.dof_states.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_velocities = self.dof_states.view(self.num_envs, self.num_dof, 2)[..., 1]


        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        self.initial_root_states = self.root_states.clone()
        self.initial_dof_states = self.dof_states.clone()
        
        
        # Control tensor
        self.all_actor_indices = torch.arange(self.num_envs * 2, dtype=torch.int32, device=self.device).reshape((self.num_envs, 2))

        
        cam_pos = gymapi.Vec3(-1.0, -0.6, 0.8)
        cam_target = gymapi.Vec3(1.0, 1.0, 0.15)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def create_sim(self):
        # implement sim set up and environment creation here
        #    - set up-axis
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        
        
        #    - set up gravity
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.82  
        #    - call super().create_sim with device args (see docstring)
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        #    - set time step length
        self.dt = self.sim_params.dt
        #    - setup asset
        self._create_exomy_asset()
        #    - create ground plane
        self._create_ground_plane()
        #    - set up environments
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        





    def _create_exomy_asset(self):
        pass


    def _create_ground_plane(self):
        """
        terrain_width = 20.
        terrain_length = 60.
        #plane_params = gymapi.PlaneParams()
        horizontal_scale = 0.5 # [m]
        vertical_scale = 0.005  # [m]
        num_rows = int(terrain_width/horizontal_scale)
        num_cols = int(terrain_length/horizontal_scale)
        #num_rows = int(plane_params/horizontal_scale)
        #num_cols = int(plane_params/horizontal_scale)
        heightfield = np.zeros((num_rows, num_cols), dtype=np.int16)

        def new_sub_terrain(): return SubTerrain(width=num_rows, length=num_cols, vertical_scale=vertical_scale, horizontal_scale=horizontal_scale)

        heightfield[0:num_rows, :] = wave_terrain(new_sub_terrain(), num_waves=10., amplitude=0.5).height_field_raw
        #heightfield[0:num_rows, :] = random_uniform_terrain(new_sub_terrain(), min_height=-0.1, max_height=0.1, step=0.1, downsampled_scale=0.5).height_field_raw

        # add the terrain as a triangle mesh
        vertices, triangles = convert_heightfield_to_trimesh(heightfield, horizontal_scale=horizontal_scale, vertical_scale=vertical_scale, slope_threshold=3.5)
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = vertices.shape[0]
        tm_params.nb_triangles = triangles.shape[0]
        tm_params.transform.p.x = -1.
        tm_params.transform.p.y = -1.
        self.gym.add_triangle_mesh(self.sim, vertices.flatten(), triangles.flatten(), tm_params)
        """
        num_terains = 8
        terrain_width = 5.
        terrain_length = 200.
        horizontal_scale = 0.25  # [m]
        vertical_scale = 0.005  # [m]
        num_rows = int(terrain_width/horizontal_scale)
        num_cols = int(terrain_length/horizontal_scale)
        heightfield = np.zeros((num_terains*num_rows, num_cols), dtype=np.int16)


        def new_sub_terrain(): return SubTerrain(width=num_rows, length=num_cols, vertical_scale=vertical_scale, horizontal_scale=horizontal_scale)


        heightfield[0:num_rows, :] = wave_terrain(new_sub_terrain(), num_waves=0., amplitude=0.).height_field_raw
        heightfield[num_rows:2*num_rows, :] = random_uniform_terrain(new_sub_terrain(), min_height=-0.05, max_height=0.1, step=0.1, downsampled_scale=0.5).height_field_raw
        heightfield[2*num_rows:3*num_rows, :] = random_uniform_terrain(new_sub_terrain(), min_height=-0.05, max_height=0.15, step=0.15, downsampled_scale=0.5).height_field_raw
        heightfield[3*num_rows:4*num_rows, :] = random_uniform_terrain(new_sub_terrain(), min_height=-0.1, max_height=0.2, step=0.2, downsampled_scale=0.5).height_field_raw
        heightfield[4*num_rows:5*num_rows, :] = random_uniform_terrain(new_sub_terrain(), min_height=-0.1, max_height=0.25, step=0.25, downsampled_scale=0.5).height_field_raw
        heightfield[5*num_rows:6*num_rows, :] = random_uniform_terrain(new_sub_terrain(), min_height=-0.15, max_height=0.3, step=0.3, downsampled_scale=0.5).height_field_raw
        heightfield[6*num_rows:7*num_rows, :] = random_uniform_terrain(new_sub_terrain(), min_height=-0.15, max_height=0.35, step=0.35, downsampled_scale=0.5).height_field_raw
        heightfield[7*num_rows:8*num_rows, :] = random_uniform_terrain(new_sub_terrain(), min_height=-0.2, max_height=0.4, step=0.4, downsampled_scale=0.5).height_field_raw

        # add the terrain as a triangle mesh
        vertices, triangles = convert_heightfield_to_trimesh(heightfield, horizontal_scale=horizontal_scale, vertical_scale=vertical_scale, slope_threshold=3.5)
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = vertices.shape[0]
        tm_params.nb_triangles = triangles.shape[0]
        tm_params.transform.p.x = -4.
        tm_params.transform.p.y = -8.
        self.gym.add_triangle_mesh(self.sim, vertices.flatten(), triangles.flatten(), tm_params)

    def set_targets(self, env_ids):
        num_sets = len(env_ids)
        # set target position randomly with x, y in (-2, 2) and z in (1, 2)
        #print("ASDO:JNHSAOJPNHDJNO:HASDJUOIP")
        alpha = math.pi * torch.rand(num_sets, device=self.device) - 90
        TargetRadius = 2
        TargetCordx = 0
        TargetCordy = 0
        RobotCordx = self.root_positions[env_ids,0]
        #print("Updating targets")
        x = TargetRadius * torch.cos(alpha) + TargetCordx + abs(RobotCordx)
        y = TargetRadius * torch.sin(alpha) + TargetCordy
        self.target_root_positions[env_ids, 0] = x
        self.target_root_positions[env_ids, 1] = y
        self.target_root_positions[env_ids, 2] = 0.5
        self.marker_positions[env_ids] = self.target_root_positions[env_ids]
        # copter "position" is at the bottom of the legs, so shift the target up so it visually aligns better
        #self.marker_positions[env_ids, 2] += 0.4
        actor_indices = self.all_actor_indices[env_ids, 1].flatten()
        self.gym.set_actor_root_state_tensor_indexed(self.sim,self.root_tensor, gymtorch.unwrap_tensor(actor_indices), num_sets)

        return actor_indices


    def _create_envs(self,num_envs,spacing, num_per_row):
       # define plane on which environments are initialized
        lower = gymapi.Vec3(0.01*-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(0.01 * spacing, spacing, spacing)

        asset_root = "../assets"
        exomy_asset_file = "urdf/exomy_modelv2/urdf/exomy_model.urdf"
        
        # if "asset" in self.cfg["env"]:
        #     asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
        #     asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)

        # asset_path = os.path.join(asset_root, asset_file)
        # asset_root = os.path.dirname(asset_path)
        # asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.disable_gravity = False
        asset_options.armature = 0.01
        # use default convex decomposition params
        asset_options.vhacd_enabled = False

        print("Loading asset '%s' from '%s'" % (exomy_asset_file, asset_root))
        exomy_asset = self.gym.load_asset(self.sim, asset_root, exomy_asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(exomy_asset)
        #print(self.num_dof)
        #################################################
        # get joint limits and ranges for Franka
        exomy_dof_props = self.gym.get_asset_dof_properties(exomy_asset)
        exomy_lower_limits = exomy_dof_props["lower"]
        exomy_upper_limits = exomy_dof_props["upper"]
        exomy_ranges = exomy_upper_limits - exomy_lower_limits
        exomy_mids = 0.5 * (exomy_upper_limits + exomy_lower_limits)
        exomy_num_dofs = len(exomy_dof_props)

        #################################################
        # set default DOF states
        default_dof_state = np.zeros(exomy_num_dofs, gymapi.DofState.dtype)
        default_dof_state["pos"] = exomy_mids


        
        exomy_dof_props["driveMode"] = [
            gymapi.DOF_MODE_VEL, #0  #L BOGIE
            gymapi.DOF_MODE_POS,  #1  #ML POS
            gymapi.DOF_MODE_VEL,  #2  #ML DRIVE
            gymapi.DOF_MODE_POS,  #3   #FL POS
            gymapi.DOF_MODE_VEL,  #4  #FL DRIVE
            gymapi.DOF_MODE_VEL, #5  #REAR BOGIE
            gymapi.DOF_MODE_POS,  #6  #RL POS
            gymapi.DOF_MODE_VEL,  #7  #RL DRIVE
            gymapi.DOF_MODE_POS,  #8  #RR POS
            gymapi.DOF_MODE_VEL,  #9  #RR DRIVE
            gymapi.DOF_MODE_VEL, #10 #R BOGIE
            gymapi.DOF_MODE_POS,  #11 #MR POS 
            gymapi.DOF_MODE_VEL,  #12 #MR DRIVE
            gymapi.DOF_MODE_POS,  #13 #FR POS
            gymapi.DOF_MODE_VEL,  #14 #FR DRIVE
        ]

        exomy_dof_props["stiffness"].fill(800.0)
        exomy_dof_props["damping"].fill(0.01)
        exomy_dof_props["friction"].fill(0.5)
        pose = gymapi.Transform()
        pose.p.z = 0.2
        # asset is rotated z-up by default, no additional rotations needed
        pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)

        self.exomy_handles = []
        self.envs = []

        #Create marker
        
        default_pose = gymapi.Transform()
        default_pose.p.z = 0.0
        default_pose.p.x = 0.1        
        marker_options = gymapi.AssetOptions()
        marker_options.fix_base_link = True
        marker_asset = self.gym.create_sphere(self.sim, 0.1, marker_options)

        #self.cameras = camera()

        for i in range(num_envs):
            # Create environment
            env0 = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.envs.append(env0)

            
            exomy0_handle = self.gym.create_actor(
                env0,  # Environment Handle
                exomy_asset,  # Asset Handle
                pose,  # Transform of where the actor will be initially placed
                "exomy",  # Name of the actor
                i,  # Collision group that actor will be part of
                1,  # Bitwise filter for elements in the same collisionGroup to mask off collision
            )
            self.exomy_handles.append(exomy0_handle)

            
    
            # Configure DOF properties
            # Set initial DOF states
            # gym.set_actor_dof_states(env0, exomy0_handle, default_dof_state, gymapi.STATE_ALL)
            # Set DOF control properties
            self.gym.set_actor_dof_properties(env0, exomy0_handle, exomy_dof_props)
            #print(self.gym.get_actor_dof_properties((env0, exomy0_handle))
            #self.cameras.add_camera(env0, self.gym, exomy0_handle)

            # Spawn marker
            marker_handle = self.gym.create_actor(env0, marker_asset, default_pose, "marker", i, 1, 1)
            self.gym.set_rigid_body_color(env0, marker_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(1, 0, 0))

    def reset_idx(self, env_ids):
        # set rotor speeds
        
        num_resets = len(env_ids)

        target_actor_indices = self.set_targets(env_ids)


        # Set orientation of robot as random around Z
        r = []
        actor_indices = self.all_actor_indices[env_ids, 0].flatten()
        for i in range(num_resets):
            r.append(R.from_euler('zyx', [(random.random() * 2 * math.pi), 0, 0], degrees=False).as_quat())

        RQuat = torch.cuda.FloatTensor(r)

        #Rot = torch.rand(num_resets, 3, device=self.device) * 2 * math.pi
        #Rot[..., 0] = 0
        #Rot[..., 1] = math.pi
        #Rot[..., 2] = math.pi
        #print(Rot[0])
        #print(Rot)
        #QuatRot = tgm.angle_axis_to_quaternion(Rot)
        
        #print(RQuat)
        
        self.root_states[env_ids] = self.initial_root_states[env_ids]
        self.root_states[env_ids, 0] = 0#torch_rand_float(-1.5, 1.5, (num_resets, 1), self.device).flatten()
        self.root_states[env_ids, 1] = 0#torch_rand_float(-1.5, 1.5, (num_resets, 1), self.device).flatten()
        self.root_states[env_ids, 2] = 0.2#torch_rand_float(-0.2, 1.5, (num_resets, 1), self.device).flatten()

        #print(tgm.quaternion_to_angle_axis(self.root_states[0, 3:7]))
        #print(self.root_states[0, 3:7])


        #Sets orientation
        self.root_states[env_ids, 3:7] = RQuat

        self.dof_states = self.initial_dof_states
        self.gym.set_actor_root_state_tensor_indexed(self.sim,self.root_tensor, gymtorch.unwrap_tensor(actor_indices), num_resets)
        #print(self.root_states[0])
        self.dof_positions = 0
        
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_states), gymtorch.unwrap_tensor(actor_indices), num_resets)

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        
        return torch.unique(torch.cat([target_actor_indices, actor_indices]))

        #Used to reset a single environment
        



    def pre_physics_step(self, actions):
        # 
        #set_target_ids = (self.progress_buf % 1000 == 0).nonzero(as_tuple=False).squeeze(-1)
       # if  torch.any(self.progress_buf % 1000 == 0):
            #print(self.marker_positions)
        target_actor_indices = torch.tensor([], device=self.device, dtype=torch.int32)
        #if len(set_target_ids) > 0:
            #target_actor_indices = self.set_targets(set_target_ids)
    
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        #print(len(reset_env_ids))
        #print(self.reset_buf)
        actor_indices = torch.tensor([], device=self.device, dtype=torch.int32)
        #print(reset_env_ids.size())
        if len(reset_env_ids) > 0:
            actor_indices = self.reset_idx(reset_env_ids)

        # Set new targets when an agent reaches current target
        target_dist = torch.sqrt(torch.square(self.target_root_positions[:,0:2] - self.root_positions[:,0:2]).sum(-1))
        reset_targets = torch.where(target_dist < 0.2, 1, 0) 
        reset_targets_ids = reset_targets.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            target_actor_indices = self.set_targets(reset_targets_ids)

        reset_indices = torch.unique(torch.cat([target_actor_indices]))
        if len(reset_indices) > 0:
            self.gym.set_actor_root_state_tensor_indexed(self.sim, self.root_tensor, gymtorch.unwrap_tensor(reset_indices), len(reset_indices))
        # if  (self.progress_buf % 500 == 0).nonzero(as_tuple=False).squeeze(-1):
        #     print(self.marker_positions)
        #print(self.marker_positions)
        #print("exomy")
        #print(self.target_root_positions)
        actions_tensor = torch.zeros(self.num_envs * self.num_dof, device=self.device, dtype=torch.float)
        _actions = actions.to(self.device)

        # actions_tensor[::self.num_dof] = actions.to(self.device).squeeze()
        # #print(np.shape(_actions))
        #print(np.shape(_actions))
        # print(actions.size())

        #DRV_LF_joint_dof_handle = self.gym.find_actor_dof_handle(self.envs[0], self.exomy_handles[0], "DRV_LF_joint")
        #max = 100
        #max = 2
        #actions_tensor = actions.to(self.device).squeeze() * 400
        #pos, vel = self.Kinematics.Get_AckermannValues(1,1)
        _actions[:,0] = _actions[:,0] * 30 # Vi ganger med 30 for at få velocities imellem -30 og 30
        _actions[:,1] = _actions[:,1] * 30 
        steering_angles, motor_velocities = Ackermann(_actions[:,0], _actions[:,1])
        
        actions_tensor[1::15]=(steering_angles[:,2])   #1  #ML POS
        actions_tensor[2::15]=(motor_velocities[:,2])  #2  #ML DRIVE
        actions_tensor[3::15]=(steering_angles[:,0])   #3   #FL POS
        actions_tensor[4::15]=(motor_velocities[:,0])  #4  #FL DRIVE
        actions_tensor[6::15]=(steering_angles[:,4])   #6  #RL POS
        actions_tensor[7::15]=(motor_velocities[:,4])  #7  #RL DRIVE
        actions_tensor[8::15]=(steering_angles[:,5])   #8  #RR POS
        actions_tensor[9::15]=(motor_velocities[:,5])  #9  #RR DRIVE
        actions_tensor[11::15]=(steering_angles[:,3])  #11 #MR POS  
        actions_tensor[12::15]= (motor_velocities[:,3])#12 #MR DRIVE
        actions_tensor[13::15]=(steering_angles[:,1])  #13 #FR POS
        actions_tensor[14::15]=(motor_velocities[:,1]) #14 #FR DRIVE
        #print(motor_velocities[:,0])
        self.motor_velocities = motor_velocities
        self.steering_angles = steering_angles
        self.lin_vel = _actions[:,0]
        self.ang_vel = _actions[:,0]

        # 
        self.gym.set_dof_velocity_target_tensor(self.sim, gymtorch.unwrap_tensor(actions_tensor)) #)
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(actions_tensor)) #)
        #forces = gymtorch.unwrap_tensor(actions_tensor)
        #self.gym.set_dof_actuation_force_tensor(self.sim, forces)
        
    def post_physics_step(self):
        # implement post-physics simulation code here
        #    - e.g. compute reward, compute observations
        """
            # refresh body tensor
        """

        self.progress_buf += 1
        #self.cameras.render_cameras(self.gym, self.sim)
        #print(torch.max(self.progress_buf))
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        root_quat = R.from_quat(self.root_quats.cpu())
        self.root_euler = torch.from_numpy(root_quat.as_euler('xyz')).to(self.device)
        #self.root_euler = tgm.quaternion_to_angle_axis(self.root_quats)
        time = torch.tensor(self.dt)
        
        

        #print(self.vec_root_tensor)
        self.compute_observations()
        self.compute_rewards()

    def compute_observations(self):
        self.obs_buf[..., 0:2] = (self.target_root_positions[..., 0:2] - self.root_positions[..., 0:2])
        self.obs_buf[..., 2] = ((self.root_euler[..., 2])  - (math.pi/2)) + (math.pi / (2 * math.pi))
        self.obs_buf[..., 3] = self.lin_vel
        self.obs_buf[..., 4] = self.ang_vel
        self.obs_buf[..., 5] = self.progress_buf
        # Heading
        # Target Dist
        
        print(self.obs_buf)
        #print(self.root_angvels)
        #print(self.obs_buf[0, 2:5])
        #print(tgm.quaternion_to_angle_axis(self.root_quats)[0])
        # self.obs_buf[..., 0:3] = (self.target_root_positions - self.root_positions) / 3
        # self.obs_buf[..., 3:7] = self.root_quats
        # self.obs_buf[..., 7:10] = self.root_linvels / 2
        # self.obs_buf[..., 10:13] = self.root_angvels / math.pi
        return self.obs_buf

    def compute_rewards(self):

        
        #print(target_dist)
        #print(target_heading)
        #print(root_euler)
        #heading_diff = target_heading - root_euler
        #print(heading_diff)

        self.rew_buf[:], self.reset_buf[:] = compute_exomy_reward(self.root_positions,
            self.target_root_positions, self.root_quats, self.root_euler,
            self.reset_buf, self.progress_buf, self.max_episode_length, self.dt, self.motor_velocities)        


@torch.jit.script
def compute_exomy_reward(root_positions, target_root_positions,
        root_quats, root_euler, reset_buf, progress_buf, max_episode_length, dt, motor_velocities):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, float, Tensor) -> Tuple[Tensor, Tensor]

    # distance to target
    #target_heading = torch.tensor(len(target_root_positions))
    target_dist = torch.sqrt(torch.square(target_root_positions[:,0:2] - root_positions[:,0:2]).sum(-1))
    #print("target_dist", target_dist)
    target_vector = target_root_positions[..., 0:2] - root_positions[..., 0:2]
    #print(torch.max(heading_diff))

    # Heading penalty. Penalty for pointing away from target. Penalty increases with target_dist (heading close to target is not as relevant)
    eps = 1e-7
    dot =  ((target_vector[..., 0] * torch.cos(root_euler[..., 2] - (math.pi/2))) + (target_vector[..., 1] * torch.sin(root_euler[..., 2] - (math.pi/2)))) / ((torch.sqrt(torch.square(target_vector[..., 0]) + torch.square(target_vector[..., 1]))) * torch.sqrt(torch.square(torch.cos(root_euler[..., 2] - (math.pi/2))) + torch.square(torch.sin(root_euler[..., 2] - (math.pi/2)))))
    angle = torch.clamp(dot, min = (-1 + eps), max = (1 - eps))
    heading_diff = torch.arccos(angle)
    heading_penalty = heading_diff * heading_diff * target_dist * 0.01

    # position reward
    pos_reward = 1.0 / (1.0 + target_dist * target_dist)
    
    # Reversing penalty: Den kører baglaens: reward = -(velocity1 + velocity2) * 0.5
    velocityML = motor_velocities[:,2]
    velocityMR = motor_velocities[:,3]
    velocityCondition = torch.where(((velocityML > 0) | (velocityMR > 0)), 0, 1)
    vel_penalty = ((velocityML + velocityMR) * velocityCondition) * 0.01

    # Goal reward for at komme indenfor xx meter af current target
    goal_reward = torch.where(target_dist < 0.2, 1, 0) * 0.1
    
    # Penalty for moving too far away from target
    distanceReset_penalty = torch.where(target_dist > 4, 1, 0) * 0.1

    # Penalty for tilting
    penaltyAngle = 0.35 #radians # absolut
    tiltFlag = torch.where((root_euler[:,0] > penaltyAngle) | (root_euler[:,1] > penaltyAngle), 1, 0)
    tiltX = torch.where((tiltFlag == 1) & (root_euler[:,0] > root_euler[:,1]), 1, 0)
    tiltY = torch.where((tiltFlag == 1) & (root_euler[:,0] < root_euler[:,1]), 1, 0)
    tilt_penalty = tiltX * root_euler[:,0] * root_euler[:,0] + tiltY * root_euler[:,1] * root_euler[:,1]
    

    # Penalty for not reaching target within max_episode_length
    timeReset_penalty = torch.where(progress_buf >= max_episode_length - 1, 1, 0) * 0.1

    # Reward function:
    reward = pos_reward + vel_penalty + goal_reward - heading_penalty - distanceReset_penalty - tilt_penalty - timeReset_penalty - 0.01
    #print(reward)
    #print(reward[0:10])
    #print((torch.max(reward), torch.argmax(reward)))

    ones = torch.ones_like(reset_buf)
    die = torch.zeros_like(reset_buf)
    # resets due to episode length'
    reset = torch.where(progress_buf >= max_episode_length - 1, ones, die)
    reset = torch.where(target_dist >= 4, ones, reset)

    
    return reward, reset        