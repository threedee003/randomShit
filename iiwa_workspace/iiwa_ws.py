import os
import argparse
import math
import numpy as np
from isaacgym import gymapi
from isaacgym import gymutil
import random
from pprint import pprint



axes_geom = gymutil.AxesGeometry(0.1)
ROOT = "/home/bikram/Documents/isaacgym/assets"


class iiwaScene:
      def __init__(self):
            self.gym = gymapi.acquire_gym()
            self.sim_params = gymapi.SimParams()
            self.sim_params.up_axis = gymapi.UP_AXIS_Z
            self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
            sim_type = gymapi.SIM_PHYSX
            compute_device_id = 0
            graphics_device_id = 0

            if sim_type == gymapi.SIM_FLEX:
                  self.sim_params.substeps = 4
                  self.sim_params.flex.solver_type = 5
                  self.sim_params.flex.num_outer_iterations = 4
                  self.sim_params.flex.num_inner_iterations = 20
                  self.sim_params.flex.relaxation = 0.75
                  self.sim_params.flex.warm_start = 0.8


            elif sim_type == gymapi.SIM_PHYSX:
                  self.sim_params.substeps = 2
                  self.sim_params.physx.solver_type = 1
                  self.sim_params.physx.num_position_iterations = 25
                  self.sim_params.physx.num_velocity_iterations = 0
                  self.sim_params.physx.num_threads = 0
                  self.sim_params.physx.use_gpu = True
                  self.sim_params.physx.rest_offset = 0.001

            self.sim = self.gym.create_sim(compute_device_id, graphics_device_id, sim_type, self.sim_params)
            self.plane_params = gymapi.PlaneParams()

            self.plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
            self.plane_params.distance = 0
            self.plane_params.static_friction = 1
            self.plane_params.dynamic_friction = 1
            self.plane_params.restitution = 0
            self.gym.add_ground(self.sim, self.plane_params)

            cabinet_urdf = "urdf/sektion_cabinet_model/urdf/sektion_cabinet.urdf"
            iiwa_urdf = "urdf/iiwa_rg2/iiwa_rg2.urdf"

            self.asset_options = gymapi.AssetOptions()
            self.asset_options.fix_base_link = False
            self.asset_options.thickness = 0.01

            cabinet_ = self.gym.load_asset(self.sim, ROOT, cabinet_urdf, self.asset_options)
            self.asset_options.fix_base_link = True
            robot_ = self.gym.load_asset(self.sim, ROOT, iiwa_urdf, self.asset_options)
            robot_props = self.gym.get_asset_dof_properties(robot_)
            # for iiwa joints
            robot_props["driveMode"][:7].fill(gymapi.DOF_MODE_POS)
            robot_props["stiffness"][:7].fill(1100.)
            robot_props["damping"][:7].fill(125.)
            # for gripper rg2
            robot_props["driveMode"][7:].fill(gymapi.DOF_MODE_POS)
            robot_props["stiffness"][7:].fill(18000.)
            robot_props["damping"][7:].fill(200.)
                  
            table_ = self._create_table_()
            cube_ = self._create_cube()
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())

            spacing = 2.0
            env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
            env_upper = gymapi.Vec3(spacing, spacing, spacing)
            self.env = self.gym.create_env(self.sim, env_lower, env_upper, 1) 
       
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(11.88, 4.0, 0.0)
            pose.r = gymapi.Quat(0, 0, 0.7068252, 0.7073883)
            self.cabinet_handle = self.gym.create_actor(self.env, cabinet_, pose, "cabinet", 0, 0)
            
            pose.p = gymapi.Vec3(12.0, 5.0, 0.7)
            pose.r = gymapi.Quat(0, 0, 0, 1)
            self.robot_handle = self.gym.create_actor(self.env, robot_, pose, "robot", 0, 0)
            print("Viwer created")
            self.table_handle = self.gym.create_actor(self.env, table_, self.table_pose, "table", 0, 0)
            self.cube_handle = self.gym.create_actor(self.env, cube_, self.cube_pose, "cube", 0, 0)
            self.gym.set_rigid_body_color(self.env, self.cube_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, self.cube_color)





      def __del__(self):
            print("scene deleted")


      def _create_table_(self):
            table_dims = gymapi.Vec3(0.8, 1.5, 0.04)
            self.table_pose = gymapi.Transform()
            self.table_pose.p = gymapi.Vec3(12.7, 0.5 * table_dims.y + 4.3, 0.7)
            self.asset_options.thickness = 1
            self.asset_options.armature = 0.001
            table_ = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, self.asset_options)
            return table_
      
      def _create_cube(self):
            cube_size = 0.05
            self.cube_color = gymapi.Vec3(np.random.uniform(0,1), np.random.uniform(0,1), np.random.uniform(0,1))
            self.cube_pose = gymapi.Transform()
            self.cube_pose.p = gymapi.Vec3(12.7, 5.02, 1)
            self.asset_options.thickness = 0.002
            self.asset_options.fix_base_link = False
            cube_ = self.gym.create_box(self.sim, cube_size, cube_size, cube_size, self.asset_options)
            return cube_
      
      def get_viewer(self):
            return self.viewer
      
      def get_sim(self):
            return self.sim
      
      def get_gym(self):
            return self.gym
      



if __name__ == '__main__':
      scene = iiwaScene()
      gym = scene.get_gym()
      viewer = scene.get_viewer()
      sim = scene.get_sim()
      while not gym.query_viewer_has_closed(viewer):
            t = gym.get_sim_time(sim)
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, False)
            gym.sync_frame_time(sim)

      del scene