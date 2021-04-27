import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from shutil import copy2


class PPO:
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True):

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(actor_critic.policy.parameters(), lr=lr, eps=eps)

    def update(self, rollouts):

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):

            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(self.num_mini_batch)

            for sample in data_generator:

                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch = sample

                ###############################################################

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch, actions_batch)

                # Compute  mixed advantage
                adv_targ = return_batch - value_preds_batch
                adv_targ = (adv_targ - adv_targ.mean()) / (adv_targ.std() + 1e-5)

                # Compute normal action loss
                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
                action_loss = - torch.min(surr1, surr2).mean()

                # Compute normal value loss
                value_loss = 0.5 * (return_batch - values).pow(2).mean()

                # Compute normal loss
                loss = value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef

                # Clean grads from previous iteration
                self.optimizer.zero_grad()

                # Normal backward pass
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        meta_value_loss, meta_action_loss, meta_loss = -1000, -1000, -1000

        return value_loss, meta_value_loss, action_loss, meta_action_loss, loss, meta_loss


class MetaPPO:
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 meta_lr=1e-4,
                 max_grad_norm=None,
                 use_clipped_value_loss=True):

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(actor_critic.policy.parameters(), lr=lr, eps=eps)
        self.meta_optimizer = optim.Adam(actor_critic.meta_net.parameters(), lr=meta_lr, eps=eps)

    def update(self, rollouts):

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):

            data_generator = rollouts.feed_forward_generator(
                self.num_mini_batch)

            for sample in data_generator:

                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                return_batch, masks_batch, old_action_log_probs_batch = sample

                ###############################################################

                # compute intrinsic values and rewards
                return_batch_int, value_batch_int = self.actor_critic.predict_intrinsic(
                    obs_batch, actions_batch)

                values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch, actions_batch)

                # compute intrinsic adv
                adv_targ = return_batch_int - values
                adv_targ = (adv_targ - adv_targ.mean()) / (adv_targ.std() + 1e-5)

                # Compute normal action loss
                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
                action_loss = - torch.min(surr1, surr2).mean()

                # Compute normal value loss
                value_loss = 0.5 * (return_batch_int - values).pow(2).mean()

                # Compute normal loss
                loss = value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef

                # Clean grads from previous iteration in both optimizers
                self.optimizer.zero_grad()
                self.meta_optimizer.zero_grad()

                # Normal backward pass
                loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

                # META STUFF ##################################################

                _, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch, actions_batch)

                # compute extrinsic adv
                adv_targ = return_batch - value_batch_int
                adv_targ = (adv_targ - adv_targ.mean()) / (adv_targ.std() + 1e-5)

                # Compute meta action loss
                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
                meta_action_loss = - torch.min(surr1, surr2).mean()

                # Compute meta value loss
                meta_value_loss = 0.5 * (return_batch - value_batch_int).pow(2).mean()

                # Compute meta loss
                # The entropy is already accounted for in the other loss
                meta_loss = meta_value_loss * self.value_loss_coef + meta_action_loss  # - dist_entropy * self.entropy_coef

                # Meta backward pass
                meta_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.meta_optimizer.step()

            num_updates = self.ppo_epoch * self.num_mini_batch

            value_loss_epoch /= num_updates
            action_loss_epoch /= num_updates
            dist_entropy_epoch /= num_updates

            return value_loss, meta_value_loss, action_loss, meta_action_loss, loss, meta_loss


def ppo_rollout(num_steps, envs, actor_critic, rollouts, det=False):

    for step in range(num_steps):
        # Sample actions
        with torch.no_grad():
            value, action, action_log_prob, recurrent_hidden_states, _ = actor_critic.act(
                rollouts.get_obs(step), rollouts.recurrent_hidden_states[step], rollouts.masks[step], deterministic=det)

        # Obser reward and next obs
        obs, reward, done, infos = envs.step(action)
        
        # If done then clean the history of observations.
        masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
        bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])

        rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks, bad_masks)


def ppo_update(agent, actor_critic, rollouts, use_gae, gamma, gae_lambda):
    
    with torch.no_grad():
        next_value = actor_critic.get_value(
            rollouts.get_obs(-1), rollouts.recurrent_hidden_states[-1], rollouts.masks[-1]).detach()

    rollouts.compute_returns(next_value, use_gae, gamma, gae_lambda)
    value_loss, meta_value_loss, action_loss, meta_action_loss, loss, meta_loss = agent.update(rollouts)
    rollouts.after_update()

    return value_loss, meta_value_loss, action_loss, meta_action_loss, loss, meta_loss


def ppo_save_model(actor_critic, fname, iter):
    torch.save(actor_critic.state_dict(), fname + ".tmp")
    os.rename(fname + '.tmp', fname)
    copy2(fname,fname+".{}".format(iter))
