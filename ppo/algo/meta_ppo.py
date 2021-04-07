import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from shutil import copy2
from copy import deepcopy


class MetaPPO():
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
        self.meta_optimizer = optim.Adam(actor_critic.intrinsic_reward_net.parameters(), lr=lr, eps=eps)

    def compute_loss(self, action_logp, old_action_logp, adv, values, old_values, returns, dist_entropy):

        ratio = torch.exp(action_logp - old_action_logp)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv
        action_loss = - torch.min(surr1, surr2).mean()

        if self.use_clipped_value_loss:
            value_pred_clipped = old_values + (values - old_values).clamp(-self.clip_param, self.clip_param)
            value_losses = (values - returns).pow(2)
            value_losses_clipped = (value_pred_clipped - returns).pow(2)
            value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = 0.5 * (returns - values).pow(2).mean()

        loss = value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef

        return loss, value_loss, action_loss

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
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                adv_targ = sample

                # Generate intrinsic rewards
                int_rewards, _ = self.actor_critic.predict_intrinsic(obs_batch, actions_batch)
                mix_return_batch = return_batch + int_rewards

                ###############################################################

                # Compute  mixed advantage
                adv_targ = mix_return_batch - value_preds_batch
                adv_targ = (adv_targ - adv_targ.mean()) / (adv_targ.std() + 1e-5)

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch, actions_batch)

                # Compute normal loss
                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
                action_loss = - torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - mix_return_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - mix_return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (mix_return_batch - values).pow(2).mean()

                loss = value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef

                # Clean grads from previous iteration in both optimizers
                self.optimizer.zero_grad()
                self.meta_optimizer.zero_grad()

                # Normal backward pass
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

                ###############################################################

                adv_targ = return_batch - (value_preds_batch - int_rewards.detach())
                adv_targ = (adv_targ - adv_targ.mean()) / (adv_targ.std() + 1e-5)

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch, actions_batch)

                # Compute meta loss
                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
                action_loss = - torch.min(surr1, surr2).mean()

                values -= int_rewards.detach()
                value_preds_batch -= int_rewards.detach()
                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                meta_loss = value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef

                # Meta backward pass
                meta_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.meta_optimizer.step()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch


    def update(self, in_rollouts, ex_rollouts):
        # Advantages are computed for both rollouts and are kept separate
        in_advantages = in_rollouts.returns[:-1] - in_rollouts.value_preds[:-1]
        in_advantages = (in_advantages - in_advantages.mean()) / (in_advantages.std() + 1e-5)

        ex_advantages = ex_rollouts.returns[:-1] - ex_rollouts.value_preds[:-1]
        ex_advantages = (ex_advantages - ex_advantages.mean()) / (
            ex_advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        kl_div_epoch = 0

        for e in range(self.ppo_epoch):
            # From the 2 advantages 2 generators are created from which we sample simultaneusly ex_sample and in_sample
            if self.actor_critic.is_recurrent:
                data_generator_in = in_rollouts.recurrent_generator(in_advantages, self.num_mini_batch)
            else:
                data_generator_in = in_rollouts.feed_forward_generator(in_advantages, self.num_mini_batch)

            if self.actor_critic.is_recurrent:
                data_generator_ex = ex_rollouts.recurrent_generator(ex_advantages, self.num_mini_batch)
            else:
                data_generator_ex = ex_rollouts.feed_forward_generator(ex_advantages, self.num_mini_batch)

            for ex_sample, in_sample in zip(data_generator_ex, data_generator_in):

                self.optimizer.zero_grad()
                self.meta_optimizer.zero_grad()

                # compute loss first for intrinsic reward then external reward update of eta with meta_optimizer
                value_loss, action_loss, dist_entropy, kl_div = self.sample_update(in_sample)
                self.optimizer.step()

                value_loss, action_loss, dist_entropy, kl_div = self.sample_update(ex_sample)
                self.meta_optimizer.step()
                #para_tensor = list(self.rew_fun.parameters())
                #print(para_tensor[0].grad)
                #print(para_tensor[1].grad)


                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                kl_div_epoch += kl_div.item()

        num_updates = self.ppo_epoch * self.num_mini_batch
        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        kl_div_epoch /= num_updates
        mean_loss = value_loss_epoch * self.value_loss_coef +  action_loss_epoch
        mean_loss += dist_entropy_epoch*self.entropy_coef if self.actor_behaviors is None else kl_div_epoch*self.entropy_coef
        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, kl_div_epoch, mean_loss


def ppo_rollout(num_steps, envs, actor_critic, rollouts, det=False):

    for step in range(num_steps):
        # Sample actions
        with torch.no_grad():
            value, action, action_log_prob, recurrent_hidden_states, _ = actor_critic.act(
                rollouts.get_obs(step), rollouts.recurrent_hidden_states[step],rollouts.masks[step], deterministic = det)

        # Obser reward and next obs
        obs, reward, done, infos = envs.step(action)

        # If done then clean the history of observations.
        masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
        bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])

        rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks, bad_masks)


def ppo_update(agent, actor_critic, rollouts, use_gae, gamma, gae_lambda, use_proper_time_limits):

    with torch.no_grad():
        next_value = actor_critic.get_value(rollouts.get_obs(-1), rollouts.recurrent_hidden_states[-1],
            rollouts.masks[-1]).detach()

    rollouts.compute_returns(next_value, use_gae, gamma, gae_lambda, use_proper_time_limits)
    value_loss, action_loss, dist_entropy, kl_div, loss = agent.update(rollouts)
    rollouts.after_update()

    return value_loss, action_loss, dist_entropy, kl_div, loss


def ppo_save_model(actor_critic, fname, iter):
    #avoid overwrite last model for safety
    torch.save(actor_critic.state_dict(), fname + ".tmp")
    os.rename(fname + '.tmp', fname)
    copy2(fname,fname+".{}".format(iter))