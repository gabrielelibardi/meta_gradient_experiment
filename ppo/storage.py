import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs, action_space, recurrent_hidden_state_size):

        obs_shape = obs.shape[1:]
        self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.recurrent_hidden_states = torch.zeros( num_steps + 1, num_processes, recurrent_hidden_state_size)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.rewards_intrinsic = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.value_preds_intrinsic = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns_intrinsic = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)

        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]

        self.actions = torch.zeros(num_steps, num_processes, action_shape)

        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()

        self.masks = torch.ones(num_steps + 1, num_processes, 1)

        # Masks that indicate whether it's a true terminal state
        # or time limit end state
        self.bad_masks = torch.ones(num_steps + 1, num_processes, 1)
        self.num_steps = num_steps
        self.step = 0
        self.set_obs(0, obs)

    def set_obs(self, step, obs):
        self.obs[step].copy_(obs)

    def get_obs(self, i):
        return self.obs[i]

    def to(self, device):
        self.obs = self.obs.to(device)
        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)
        self.rewards_intrinsic = self.rewards_intrinsic.to(device)
        self.value_preds_intrinsic = self.value_preds_intrinsic.to(device)
        self.returns_intrinsic = self.returns_intrinsic.to(device)

    def insert(self, obs, recurrent_hidden_states, actions, action_log_probs,
               value_preds, rewards, masks, bad_masks):
        self.set_obs(self.step + 1,obs)
        self.recurrent_hidden_states[self.step + 1].copy_(recurrent_hidden_states)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds_intrinsic[self.step].copy_(value_preds) # changed to intrinsic
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])

    def compute_returns_intrinsic(self,
                                  next_value,
                                  use_gae,
                                  gamma,
                                  gae_lambda,
                                  use_proper_time_limits=True):
        if use_proper_time_limits:
            if use_gae:
                gae = 0
                self.value_preds_intrinsic[-1] = next_value
                for step in reversed(range(self.rewards_intrinsic.size(0))):

                    delta = self.rewards_intrinsic[step] + gamma * self.value_preds_intrinsic[step + 1] \
                        * self.masks[step + 1] - self.value_preds_intrinsic[step]

                    gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae

                    self.returns_intrinsic[step] = gae + self.value_preds_intrinsic[step]
            else:
                self.returns_intrinsic[-1] = next_value
                for step in reversed(range(self.rewards_intrinsic.size(0))):
                    self.returns_intrinsic[step] = self.returns_intrinsic[step + 1] * \
                        gamma * self.masks[step + 1] + self.rewards_intrinsic[step]

    def compute_returns(self,
                        next_value,
                        use_gae,
                        gamma,
                        gae_lambda,
                        use_proper_time_limits=True):
        if use_proper_time_limits:
            if use_gae:
                gae = 0
                self.value_preds[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):

                    delta = self.rewards[step] + gamma * self.value_preds[step + 1] \
                        * self.masks[step + 1] - self.value_preds[step]

                    gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae

                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = self.returns[step + 1] * \
                        gamma * self.masks[step + 1] + self.rewards[step]

    def feed_forward_generator(self,
                               advantages,
                               advantages_intrinsic,
                               num_mini_batch=None,
                               mini_batch_size=None):
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
            obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]
            recurrent_hidden_states_batch = self.recurrent_hidden_states[:-1].view(-1, self.recurrent_hidden_states.size(-1))[indices]
            actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            value_preds_int_batch = self.value_preds_intrinsic[:-1].view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            return_int_batch = self.returns_intrinsic[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices]

            adv_targ = advantages.view(-1, 1)[indices]
            adv_targ_int = advantages_intrinsic.view(-1, 1)[indices]

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch,\
                  adv_targ, adv_targ_int, value_preds_int_batch.clone(), return_int_batch.clone()
