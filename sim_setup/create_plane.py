import os
import argparse
import math
import numpy as np
from isaacgym import gymapi
from isaacgym import gymutil

axes_geom = gymutil.AxesGeometry(0.1)

#TODO: add sphereical objects from kuka_bin.py



gym = gymapi.acquire_gym()

num_ens = 1
num_objects = 1

sim_params = gymapi.SimParams()
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
gym.add_ground(sim, plane_params)

print("added ground plane")

viewer = gym.create_viewer(sim, gymapi.CameraProperties())



while not gym.query_viewer_has_closed(viewer):
      t = gym.get_sim_time(sim)
      gym.simulate(sim)
      gym.fetch_results(sim, True)
      gym.step_graphics(sim)
      gym.draw_viewer(viewer, sim, False)
      gym.sync_frame_time(sim)


gym.destroy_viewer(viewer)
gym.destroy_sim(sim)

