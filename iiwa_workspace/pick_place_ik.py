import os
import argparse
import math
import numpy as np
from isaacgym import gymapi
from isaacgym import gymutil
import random
from pprint import pprint
import time

from python.iiwa_workspace.iiwa_ws import *
from python.FKIK.fkik import FKIK


def transform_object_to_tool0(object_quat):
      object_rot = R.from_quat(object_quat)
      flip_rot = R.from_euler('x', 180, degrees=True)
      xy_rot = R.from_euler('z', 0, degrees=True)
      new_rot = object_rot * flip_rot * xy_rot
      return new_rot.as_quat()


def main():
      scene = iiwaScene(control_type='velocity')
      gym = scene.get_gym()
      viewer = scene.get_viewer()
      sim = scene.get_sim()
      env = scene.get_env()
      fkik = FKIK()
      calculate = True
      open = False
      grasped = False
      while not gym.query_viewer_has_closed(viewer):
            t = scene.step()


            
            
            if t > 2. and calculate:
                  curr_jt = [0.]*7
                  cube_pos, cube_or = scene.get_state(scene.cube_handle)
                  # grasp_quat = np.array([0.855, 0.517, 0.039, -0.006])
                  grasp_quat = transform_object_to_tool0(object_quat=cube_or)
                  x, r = scene.to_robot_frame(pos=cube_pos+np.array([0., 0., 0.25]), quat=grasp_quat)
                  stop1 = fkik.get_ik(qinit=curr_jt, pos=x, quat=r)
                  x, r = scene.to_robot_frame(pos=cube_pos+np.array([0, 0, 0.02]), quat=grasp_quat)
                  stop2 = fkik.get_ik(qinit=stop1, pos=x, quat=r)

                  calculate = False



            if not calculate:
                  if not scene.reached_jt(stop1) and open == False:
                        scene.reach_jt_position(stop1)
                        print("go for stop1")
                  if scene.reached_jt(stop1) and open == False:
                        print("gripper_open")
                        scene.gripper_action('open')
                        open = True
                  elif open and not scene.reached_jt(stop2) and not grasped:
                        print("go for stop2")
                        scene.reach_jt_position(stop2)

                  if scene.reached_jt(stop2, eps=0.031):
                        scene.gripper_action('close')
                        
                        
                  




            # if scene.:
            #       scene.reach_jt_position(stop2)
            # break
