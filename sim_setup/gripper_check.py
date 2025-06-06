import os
import argparse
import math
import numpy as np
from isaacgym import gymapi
from isaacgym import gymutil
import random

''''
WARNING: This file has Y AXIS UP.


'''




axes_geom = gymutil.AxesGeometry(0.1)

#TODO: add make Z axis up



gym = gymapi.acquire_gym()

num_ens = 1
num_objects = 1

sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
sim_type = gymapi.SIM_PHYSX
if sim_type == gymapi.SIM_FLEX:
      sim_params.substeps = 4
      sim_params.flex.solver_type = 5
      sim_params.flex.num_outer_iterations = 4
      sim_params.flex.num_inner_iterations = 20
      sim_params.flex.relaxation = 0.75
      sim_params.flex.warm_start = 0.8
elif sim_type == gymapi.SIM_PHYSX:
      sim_params.substeps = 2
      sim_params.physx.solver_type = 1
      sim_params.physx.num_position_iterations = 25
      sim_params.physx.num_velocity_iterations = 0
      sim_params.physx.num_threads = 0
      sim_params.physx.use_gpu = True
      sim_params.physx.rest_offset = 0.001

compute_device_id = 0
graphics_device_id = 0


sim = gym.create_sim(compute_device_id, graphics_device_id, sim_type, sim_params)
print("Sim created !")

plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)

print("added ground plane")

asset_root = "/home/bikram/Documents/isaacgym/assets"

# assets/urdf/rg2_description/urdf/rg2.urdf
# sektion_asset_file = "urdf/sektion_cabinet_model/urdf/sektion_cabinet.urdf"
# /home/bikram/Documents/isaacgym/assets/urdf/iiwa_rg2/iiwa_rg2.urdf
sektion_asset_file = "urdf/rg2_description/urdf/rg2.urdf"
# sektion_asset_file = "urdf/iiwa_rg2/iiwa_rg2.urdf"



asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.override_inertia = True


franka_asset = gym.load_asset(sim, asset_root, sektion_asset_file, asset_options)
robot_props = gym.get_asset_dof_properties(franka_asset)

robot_props["driveMode"][:].fill(gymapi.DOF_MODE_POS)
robot_props["stiffness"][:].fill(500.)
robot_props["damping"][:].fill(40.)






viewer = gym.create_viewer(sim, gymapi.CameraProperties())

num_envs = 1
spacing = 2.0
env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
env_upper = gymapi.Vec3(spacing, spacing, spacing)
envs = []
for i in range(num_envs):
    env = gym.create_env(sim, env_lower, env_upper, num_envs)
    envs.append(env)


pose = gymapi.Transform()

pose.p = gymapi.Vec3(10.0, 10., 3.)  
pose.r = gymapi.Quat(0, 0, 0, 1)
cabinet_handle = gym.create_actor(envs[0], franka_asset, pose, "cube", 0, 1)

close_state = {
      'gripper_joint':-0.391,
      'l_finger_passive_joint':-0.39099,
      'r_finger_1_joint':0.39086,
      'r_finger_passive_joint':0.3909,
      'l_finger_2_joint':0.3911,
      'r_finger_2_joint':-0.3909
}

joint_names = gym.get_actor_dof_names(env, cabinet_handle)
close_jt = []
for joints in joint_names:
      close_jt.append(close_state[joints])
print(close_jt)
gym.set_actor_dof_properties(env, cabinet_handle, robot_props)


while not gym.query_viewer_has_closed(viewer):
      t = gym.get_sim_time(sim)
      gym.simulate(sim)
      gym.fetch_results(sim, True)
      gym.step_graphics(sim)
      gym.draw_viewer(viewer, sim, False)
      gym.sync_frame_time(sim)
      print(t)
      if t > 7.:
            gym.set_actor_dof_velocity_targets(env, cabinet_handle, close_jt)
      # t += 1
      dof_states = gym.get_actor_dof_states(env, cabinet_handle, gymapi.STATE_ALL)
      joint_pos = [dof_state['pos'] for dof_state in dof_states]
      joint_vel = [dof_state['vel'] for dof_state in dof_states]
      # joint_names = gym.get_actor_dof_names(env, cabinet_handle)
      # print(joint_names)
      # joint_vel = [dof_state['vel'] for dof_state in dof_states]


gym.destroy_viewer(viewer)
gym.destroy_sim(sim)

