import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from shutil import copy2


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
                 meta_lr=2e-4,
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

        self.optimizer = optim.Adam(actor_critic.policy.parameters(), lr=3e-4, eps=eps)
        self.meta_optimizer = optim.Adam(actor_critic.meta_net.parameters(), lr=1e-4, eps=eps)

    def update(self, rollouts):
        
        advantages = rollouts.returns[:-1] - rollouts.value_preds_extrinsic[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        
        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        

        for e in range(self.ppo_epoch):

            data_generator = rollouts.feed_forward_generator(advantages, self.num_mini_batch)
            
            for sample in data_generator:
                
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                return_batch_ext, masks_batch, old_action_log_probs_batch, \
                value_preds_batch_ext, value_preds_batch_int, adv_targ_ext, TD_batch, coef_mat = sample

                # SUMMARY VALUES
                print()
                print("summary obs {}, shape {}".format(obs_batch.sum(), obs_batch.shape))
                print("summary act {}, shape {}".format(actions_batch.sum(), actions_batch.shape))
                print("summary ext_ret {}, shape {}".format(return_batch_ext.sum(), return_batch_ext.shape))
                print("summary old_action_log_prob {}, shape {}".format(old_action_log_probs_batch.sum(), old_action_log_probs_batch.shape))
                print("summary adv_ext {}, shape {}".format(adv_targ_ext.sum(), adv_targ_ext.shape))
                print("summary TD {}, shape {}".format(TD_batch.sum(), TD_batch.shape))
                print("summary coef matrix {}, shape {}".format(coef_mat.sum(), coef_mat.shape))
                print()

                # COMPUTE ADVANTAGES LIKE "SIMULATE GAE"
                
                rewards_int = self.actor_critic.predict_intrinsic_rewards(rollouts.obs[:-1].view(-1, *rollouts.obs.size()[2:]), rollouts.actions.view(-1, rollouts.actions.size(-1)))
                print("summary int rewards prediction {}, values {}".format(rewards_int.sum(), rewards_int))
                
                delta = rewards_int + TD_batch                                                                           
                adv_targ_int = torch.matmul(coef_mat, delta)
                return_batch_int =  adv_targ_int + value_preds_batch_int
                adv_targ_int = (adv_targ_int - adv_targ_int.mean()) / (adv_targ_int.std() + 1e-5)
                
                ###############################################################

                # Clean grads from previous iteration in both optimizers
                
                

                values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch, actions_batch)
                
                print("intrinsi value predictions: {}".format(values[0:5]))
                
                # Compute normal action loss
                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ_int
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ_int
                action_loss = - torch.min(surr1, surr2).mean()

                # Compute normal value loss
                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch_int + (values - value_preds_batch_int).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch_int).pow(2)
                    value_losses_clipped = (value_pred_clipped - return_batch_int).pow(2)
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch_int - values).pow(2).mean()

                # Compute normal loss
                loss = value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef
                print("policy loss: {}".format(loss))
                print("intrinsi value loss: {}".format(value_loss))
                print("action loss : {}".format(action_loss))
                print("entropy : {}".format(dist_entropy))
                

                # Normal backward pass
                loss.backward()
                """for param in self.actor_critic.policy.parameters():
                    if param.grad is None:
                        print('ATTENTION! None found')
                    else:
                        assert not torch.isnan(param.grad.data).any()"""
                    
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                self.optimizer.zero_grad()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

                # META STUFF ##################################################

                meta_values = self.actor_critic.get_extrinsic_value(obs_batch, recurrent_hidden_states_batch, masks_batch)

                _, action_log_probs_new, _, _ = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch, actions_batch)


                # Compute meta action loss
                ratio = torch.exp(action_log_probs_new - old_action_log_probs_batch)
                surr1 = ratio * adv_targ_ext
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ_ext
                meta_action_loss = - torch.min(surr1, surr2).mean()

                # Compute meta value loss
                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch_ext + (meta_values - value_preds_batch_ext).clamp(-self.clip_param, self.clip_param)
                    value_losses = (meta_values - value_preds_batch_ext).pow(2)
                    value_losses_clipped = (value_pred_clipped - return_batch_ext).pow(2)
                    meta_value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                else:
                    meta_value_loss = 0.5 * (return_batch_ext - meta_values).pow(2).mean()

                # Compute meta loss
                # The entropy is already accounted for in the other loss
                meta_loss = meta_value_loss * self.value_loss_coef + meta_action_loss  # - dist_entropy * self.entropy_coef
                print("meta loss: {}".format(meta_loss))
                print("meta value loss: {}".format(meta_value_loss))

                # Meta backward pass
                meta_loss.backward()
                """for param in self.actor_critic.meta_net.parameters():
                    if param.grad is None:
                        print('ATTENTION! None found')
                    else:
                        assert not torch.isnan(param.grad.data).any()"""
                        
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.meta_optimizer.step()
                self.meta_optimizer.zero_grad()
                import ipdb; ipdb.set_trace()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates


        return value_loss, meta_value_loss, action_loss, meta_action_loss, loss, meta_loss


def ppo_rollout(num_steps, envs, actor_critic, rollouts, det=False):

    for step in range(num_steps):
        # Sample actions
        with torch.no_grad():
            intrinsic_value, extrinsic_value, action, action_log_prob, recurrent_hidden_states, _ = actor_critic.act(
                rollouts.get_obs(step), rollouts.recurrent_hidden_states[step], rollouts.masks[step], deterministic=det)

        # Obs reward and next obs
        obs, reward, done, infos = envs.step(action)
        
        # If done then clean the history of observations.
        masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
        bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])

        rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, intrinsic_value, extrinsic_value, reward, masks, bad_masks)


def ppo_update(agent, actor_critic, rollouts, use_gae, gamma, gae_lambda):

    # Intrinsic returns
    with torch.no_grad():
        next_intrinsic_value = actor_critic.get_intrinsic_value(
            rollouts.get_obs(-1), rollouts.recurrent_hidden_states[-1], rollouts.masks[-1]).detach()
    rollouts.compute_deltas(next_intrinsic_value, gamma, gae_lambda, use_gae)

    # Extrinsic returns
    with torch.no_grad():
        next_extrinsic_value = actor_critic.get_extrinsic_value(
            rollouts.get_obs(-1), rollouts.recurrent_hidden_states[-1], rollouts.masks[-1]).detach()
    rollouts.compute_returns_extrinsic(next_extrinsic_value, gamma, gae_lambda, use_gae)

    value_loss, meta_value_loss, action_loss, meta_action_loss, loss, meta_loss = agent.update(rollouts)
    rollouts.after_update()

    return value_loss, meta_value_loss, action_loss, meta_action_loss, loss, meta_loss


def ppo_save_model(actor_critic, fname, iter):
    torch.save(actor_critic.state_dict(), fname + ".tmp")
    os.rename(fname + '.tmp', fname)
    copy2(fname,fname+".{}".format(iter))

