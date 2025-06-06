import os
import argparse
import math
import numpy as np
from isaacgym import gymapi
from isaacgym import gymutil
import random
from pprint import pprint
from python.utils.robot_kinematic_utils import *
from copy import deepcopy
from PIL import Image
import matplotlib.pyplot as plt

def deg2rad(values):
      const = math.pi/ 180.
      return const*values

# axes_geom = gymutil.AxesGeometry(0.1)
#TODO: add sphereical objects from kuka_bin.py



gym = gymapi.acquire_gym()

num_ens = 1
num_objects = 1

sim_params = gymapi.SimParams()
# sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
sim_type = gymapi.SIM_PHYSX
if sim_type == gymapi.SIM_FLEX:
      sim_params.substeps = 4
      sim_params.flex.solver_type = 5
      sim_params.flex.num_outer_iterations = 8
      sim_params.flex.num_inner_iterations = 20
      sim_params.flex.relaxation = 0.75
      sim_params.flex.warm_start = 0.8
elif sim_type == gymapi.SIM_PHYSX:
      sim_params.substeps = 2
      sim_params.physx.solver_type = 1
      sim_params.physx.num_position_iterations = 25
      sim_params.physx.num_velocity_iterations = 8
      sim_params.physx.num_threads = 0
      sim_params.physx.use_gpu = True
      sim_params.physx.rest_offset = 0.001

compute_device_id = 0
graphics_device_id = 0


sim = gym.create_sim(compute_device_id, graphics_device_id, sim_type, sim_params)
print("Sim created !")


plane_params = gymapi.PlaneParams()

plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
plane_params.distance = 0
plane_params.static_friction = 1
plane_params.dynamic_friction = 1
plane_params.restitution = 0
gym.add_ground(sim, plane_params)

print("added ground plane")

asset_root = "/home/bikram/Documents/isaacgym/assets"
sektion_asset_file = "urdf/sektion_cabinet_model/urdf/sektion_cabinet.urdf"

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = False
# asset_options.override_inertia = True
asset_options.thickness = 1

sektion_ = gym.load_asset(sim, asset_root, sektion_asset_file, asset_options)



# robot_asset_file = "urdf/kuka_iiwa_lbr_support/urdf/lbr_iiwa_14_r820.urdf"
# robot_asset_file = "urdf/iiwa_description/urdf/iiwa14.urdf"
# robot_asset_file = "urdf/kuka_allegro_description/kuka.urdf"
# franka = "urdf/franka_description/robots/franka_panda.urdf"
# robot_asset_file = "urdf/iiwa_rg2/iiwa_rg2.urdf"
robot_asset_file = "urdf/iiwa_rg2/iiwa_wsg.urdf"
asset_options.fix_base_link = True
# asset_options.flip_visual_attachments = True
# asset_options.disable_gravity = True
robot_ = gym.load_asset(sim, asset_root, robot_asset_file, asset_options)
asset_options.disable_gravity = False


'''
for iiwa nearly perfect tuning with just iiwa kp = 1100, kd = 125
for iiwa with wsg kp = 800 kd = 400 for more slower smoother please decrease kp

applied_torque = pos_err*kp + vel_err*kd


'''

mode = "ik"
if mode == "ik":
      robot_props = gym.get_asset_dof_properties(robot_)
      robot_props["driveMode"][:7].fill(gymapi.DOF_MODE_POS)
      robot_props["stiffness"][:7].fill(700.)
      robot_props["damping"][:7].fill(400.)

      robot_props["driveMode"][7:].fill(gymapi.DOF_MODE_POS)
      robot_props["stiffness"][7:].fill(1100.)
      robot_props["damping"][7:].fill(135.)
      robot_props["friction"][7:].fill(1.)
else:
      robot_props = gym.get_asset_dof_properties(robot_)



print(robot_props)

'''

table



'''

table_dims = gymapi.Vec3(0.8, 1.5, 0.04)
table_pose = gymapi.Transform()
table_pose.p = gymapi.Vec3(12.7, 0.5 * table_dims.y + 4.3, 0.7)
asset_options.thickness = 1
asset_options.armature = 0.001
table_ = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, asset_options)


cube_size = 0.05
cube_color = gymapi.Vec3(np.random.uniform(0,1), np.random.uniform(0,1), np.random.uniform(0,1))
cube_pose = gymapi.Transform()
cube_pose.p = gymapi.Vec3(12.7, 0.5 * table_dims.y + 4.3, 1)
# cube_pose.p = gymapi.Vec3(12.65, 5.097, 0.967)
asset_options.thickness = 0.002
asset_options.fix_base_link = False
cube_ = gym.create_box(sim, cube_size, cube_size, cube_size, asset_options)


# table_asset_file = "urdf/square_table.urdf"
# asset_options.fix_base_link = True
# table_ = gym.load_asset(sim, asset_root, table_asset_file, asset_options)




viewer = gym.create_viewer(sim, gymapi.CameraProperties())


spacing = 2.0
env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
env_upper = gymapi.Vec3(spacing, spacing, spacing)


env = gym.create_env(sim, env_lower, env_upper, 1) 



pose = gymapi.Transform()
# x, z, y
pose.p = gymapi.Vec3(11.88, 4.0, 0.0) 
pose.r = gymapi.Quat(0, 0, 0.7068252, 0.7073883)
cabinet_handle = gym.create_actor(env, sektion_, pose, "cabinet", 0, 0)


pose.p = gymapi.Vec3(12.0, 5.0, 0.7)
pose.r = gymapi.Quat( 0, 0, 0.7068252, 0.7073883)
robot_handle = gym.create_actor(env, robot_, pose, "robot", 0, 0)


gym.set_actor_dof_properties(env, robot_handle, robot_props)

table_handle = gym.create_actor(env, table_, table_pose, "table", 0, 0)
cube_handle = gym.create_actor(env, cube_, cube_pose, "cube", 0, 0)
gym.set_rigid_body_color(env, cube_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, cube_color)

# pose.p = gymapi.Vec3(10.0, 5.0, 0.)
# pose.r = gymapi.Quat(0, 0, 0, 1)
# table_handle = gym.create_actor(envs[0], table_, pose, "table", 0, 1)

inital_pos_deg = np.array([-84.53, 32.81, 3.86, -74.98, -1.14, 60.02, 126.36])
inital_pos_rad = deg2rad(inital_pos_deg)
num_dofs = 7
dof_pos = [0.]* num_dofs
# dof_pos[5] = 1.1965584

dof_pos = np.array(inital_pos_rad)
gripper_pos = np.array([-0.03, 0.03])
dof_pos = np.concatenate((dof_pos, gripper_pos), axis = 0).tolist()
# print(dof_pos.shape)



inital_pos_deg = np.array([-84.53, 32.81, 3.86, -74.98, -1.14, 60.02, 16.36])
inital_pos_rad = deg2rad(inital_pos_deg)
# gripper_init = np.array([ 0.0009644612, -0.00011103508, -0.0007428767, -0.00064719736, 0.0002961707, -0.00023958732])
# inital_pos_rad = np.concatenate((inital_pos_rad, gripper_init), 0).tolist()

camera_props = gymapi.CameraProperties()
camera_props.width = 512
camera_props.height = 512
camera_handle = gym.create_camera_sensor(env, camera_props)

gym.set_camera_location(camera_handle, env, gymapi.Vec3(12.7, 0.5 * table_dims.y + 4.3, 4), gymapi.Vec3(12.7, 0.5 * table_dims.y + 4.3, 1))


nprinted = True

while not gym.query_viewer_has_closed(viewer):
      t = gym.get_sim_time(sim)
      gym.simulate(sim)
      gym.render_all_camera_sensors(sim)
      gym.fetch_results(sim, True)
      gym.step_graphics(sim)
      gym.draw_viewer(viewer, sim, False)
      gym.sync_frame_time(sim)
      t += 1
      jt_pos, jt_vel = get_joint_pos_vel(gym, env, robot_handle)
     
      color_image = gym.get_camera_image(sim, env, viewer_handle, gymapi.IMAGE_COLOR)
      # print(color_image.dtype)
      plt.imsave("./cam.png", color_image)
      # print(jt_pos)
      # jt_names = get_joint_names(gym, env, robot_handle)
      # print(jt_names)
      # robot_props = gym.get_asset_dof_properties(robot_)
      # print(robot_props)
      # print(robot_props)
      # print("=====")
      # print("lower")
      # print(robot_props["lower"])
      # print("======")
      # print("upper")
      # print(robot_props["upper"])
      # print(t)
      # if t < 6.:
      #       gym.set_actor_dof_velocity_targets(env, robot_handle, inital_pos_rad)
      if t >= 6. and t < 12.:
            gym.set_actor_dof_position_targets(env, robot_handle, dof_pos)
      # if t >= 12:
      #       if t > 12. and nprinted:
      #             print("Now")
      #             nprinted = False
      #       gym.set_actor_dof_position_targets(env, robot_handle, stop3)


      # print(stop3)
      # break

      # t = gym.get_actor_rigid_body_states(env, cube_handle, gymapi.STATE_POS)




gym.destroy_viewer(viewer)
gym.destroy_sim(sim)

