
from python.iiwa_workspace.iiwa_ws import iiwaScene
from isaacgym import gymapi
import numpy as np

class Pick(iiwaScene):
    def __init__(self,
                 scale_reward: float = 1.0
                 ):
        self.scale_reward = scale_reward
        pass
    
    #NOTE: To be used only if velocity control is used.
    def reach_jt_position(self, desired_jt: np.ndarray):
        kp = 6.
        kd = 1.
        assert self.control_type == 'velocity', f"only for velocity control"
        dof_states = self.gym.get_actor_dof_states(self.env, self.robot_handle, gymapi.STATE_ALL)
        joint_pos = [dof_state['pos'] for dof_state in dof_states]
        joint_vel = [dof_state['vel'] for dof_state in dof_states]
        jt_error = desired_jt-np.array(joint_pos)
        vel_cmd = (kp*jt_error - kd*np.array(joint_vel)).tolist()
        self.gym.set_actor_dof_velocity_targets(self.env, self.robot_handle, vel_cmd)



    def apply_arm_action(self, action: np.ndarray):
        if self.control_type == 'position':
            dof_states = self.gym.get_actor_dof_states(self.env, self.robot_handle, gymapi.STATE_ALL)
            joint_pos = [dof_state['pos'] for dof_state in dof_states]
            gripper_jt1 = joint_pos[7]
            gripper_jt2 = joint_pos[8]
            action[7] = gripper_jt1
            action[8] = gripper_jt2
            self.gym.set_actor_dof_position_targets(self.env, self.robot_handle, action.tolist())
        elif self.control_type == 'velocity':
            pass

        else:
            raise NotImplementedError("I will not implement it.")
        
    
    def gripper_action(self, open: bool)

    def compute_reward(self):
        x = None
        y = None
        reward = 0.
        dist = np.linalg.norm(x-y)
        reaching_reward = 1 - np.tanh(10.0*dist)
        reward += reaching_reward
        grasping_reward = 0.25 if self.check_grasp() else 0.
        reward += grasping_reward
        if self.scale_reward is not None:
            reward *= self.scale_reward / 2.25

        return reward
