from manipgen.envs.franka_open import FrankaOpen
from manipgen.envs.franka_grasp_handle import FrankaGraspHandle
from isaacgymenvs.utils.torch_jit_utils import quat_conjugate, quat_rotate
import torch
import numpy as np

class FrankaClose(FrankaOpen):
    """We only need to change the initial and target object dof positions for close task.
    """
    def _set_object_init_and_target_dof_pos(self):
        """ Set initial and target object dof positions for close task """
        self.rest_object_dof_pos = self.object_dof_upper_limits.clone()
        self.target_object_dof_pos = self.object_dof_lower_limits.clone()
    
    def reset_idx(self, env_ids: torch.Tensor, validation_set: bool = False, switch_object: bool = False, init_states = None):
        FrankaGraspHandle.reset_idx(self, env_ids, validation_set, switch_object, init_states)

        # record position of fingertip center in the handle frame
        self.init_fingertip_centered_pos_local[:] = quat_rotate(
            quat_conjugate(self.handle_quat), self.fingertip_centered_pos - self.handle_pos
        )
        self.prev_fingertip_centered_pos_local = quat_rotate(
            quat_conjugate(self.handle_quat), self.fingertip_centered_pos - self.handle_pos
        )

        if not self.sample_mode and self.object_meta["type"] == "door" and self.cfg_env.rl.partial_close_for_door:
            self.target_object_dof_pos = self.init_object_dof_pos - np.deg2rad(self.cfg_env.rl.partial_close_degree)
            self.target_object_dof_pos = torch.clamp(self.target_object_dof_pos, self.object_dof_lower_limits, self.object_dof_upper_limits)
