import numpy as np
import multiprocessing
import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs, action_space,
                 recurrent_hidden_state_size):
        if isinstance(obs,tuple):
            obs_shape = obs[0].shape[1:] # 0 is for num_procs
            states_size = obs[1].shape[1:]
            self.has_states = True
        else:
            obs_shape = obs.shape[1:]
            states_size = (0,)
            self.has_states = False

        self.batches = []
        self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.recurrent_hidden_states = torch.zeros( num_steps + 1, num_processes, recurrent_hidden_state_size)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds_intrinsic = torch.zeros(num_steps + 1, num_processes, 1)
        self.value_preds_extrinsic = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.masks = torch.ones(num_steps + 1, num_processes, 1)
        self.delta = torch.zeros(num_steps, num_processes, 1)

        # Masks that indicate whether it's a true terminal state
        # or time limit end state
        self.bad_masks = torch.ones(num_steps + 1, num_processes, 1)
        self.states = torch.zeros(num_steps + 1, num_processes, *states_size)
        self.num_steps = num_steps
        self.step = 0
        self.set_obs(0,obs)

    def set_obs(self, step, obs):
        if isinstance(obs,tuple):
            self.obs[step].copy_(obs[0])
            self.states[step].copy_(obs[1])
        else:
            self.obs[step].copy_(obs)
    
    def get_obs(self, i):
        if self.has_states:
            return (self.obs[i],self.states[i])
        else:
            return self.obs[i]

    def to(self, device):
        self.obs = self.obs.to(device)
        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds_intrinsic = self.value_preds_intrinsic.to(device)
        self.value_preds_extrinsic = self.value_preds_extrinsic.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)
        self.states = self.states.to(device)
        self.delta = self.delta.to(device)



    def insert(self, obs, recurrent_hidden_states, actions, action_log_probs,
               value_preds_int, value_preds_ext, rewards, masks, bad_masks):

        self.set_obs(self.step + 1, obs)
        self.recurrent_hidden_states[self.step + 1].copy_(recurrent_hidden_states)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds_intrinsic[self.step].copy_(value_preds_int)
        self.value_preds_extrinsic[self.step].copy_(value_preds_ext)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])
        self.states[0].copy_(self.states[-1])


    def compute_returns_extrinsic(self,
                        next_value,
                        use_gae,
                        gamma,
                        gae_lambda):


        self.value_preds_extrinsic[-1] = next_value
        gae = 0
        for step in reversed(range(self.rewards.size(0))):
            delta = self.rewards[step] + gamma * self.value_preds_extrinsic[
                step + 1] * self.masks[step +
                                       1] - self.value_preds_extrinsic[step]
            gae = delta + gamma * gae_lambda * self.masks[step +
                                                          1] * gae
            self.returns[step] = gae + self.value_preds_extrinsic[step]

  
    def compute_deltas(self,
                        next_value_int,
                        use_gae,
                        gamma,
                        gae_lambda):

        self.value_preds_intrinsic[-1] = next_value_int
        gae = 0
        #for step in reversed(range(self.rewards.size(0))):
        self.delta = gamma * self.value_preds_intrinsic[1:]*self.masks[1:] - self.value_preds_intrinsic[:-1]
            # Took out the mask here carefull

                    
    def feed_forward_generator(self,
                               advantages,
                               num_mini_batch=None,
                               mini_batch_size=None):

        self.advantages = advantages
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(num_processes, num_steps, num_processes * num_steps,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch

        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True)


        for indices in sampler:
            yield self.prepare_batch((mini_batch_size, indices))

    def prepare_batch(self, args):
        mini_batch_size, indices = args
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps

        obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]
        recurrent_hidden_states_batch = self.recurrent_hidden_states[:-1].view(-1,self.recurrent_hidden_states.size(-1))[indices]
        actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
        masks_batch = self.masks[:-1].view(-1, 1)[indices]
        old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices]
        value_preds_batch_ext = self.value_preds_extrinsic[:-1].view(-1, 1)[indices]
        value_preds_batch_int = self.value_preds_intrinsic[:-1].view(-1, 1)[indices]
        return_batch = self.returns[:-1].view(-1, 1)[indices]
        TD_batch = self.delta.view(-1, 1).clone()
        adv_targ = self.advantages.view(-1, 1)[indices]

        # COMPUTE COEF MATRIX
        GAMM = 0.99
        LAMB = 0.95
        # COMPUTE IN NUMPY, IT IS FASTER
        coef_mat = np.zeros([mini_batch_size, batch_size], "float32")
        masks = self.masks.view(-1, 1)[:-1].cpu().detach().numpy()

        for i in range(mini_batch_size):
            coef = 1.0
            for j in range(indices[i], batch_size):
                if j > indices[i] and (not masks[j] or j % num_steps == 0):
                    break
                coef_mat[i][j] = coef
                coef *= GAMM * LAMB
        coef_mat = torch.Tensor(coef_mat).to(self.masks.device)
        return (obs_batch, recurrent_hidden_states_batch, actions_batch, \
              return_batch, masks_batch, old_action_log_probs_batch, \
              value_preds_batch_ext, value_preds_batch_int, adv_targ, TD_batch, coef_mat)

    def mycallback(self, x):
        self.batches.append(x)
        print('LEN THERE', len(self.batches))
