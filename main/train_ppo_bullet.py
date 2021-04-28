import os
import sys
import time
import argparse
import numpy as np
import torch
import csv

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')

from ppo import utils
from ppo.algo.meta_ppo import MetaPPO
from ppo.envs import make_vec_envs
from ppo.model import MetaPolicy
from ppo.storage import RolloutStorage
from ppo.algo.meta_ppo import ppo_rollout, ppo_update, ppo_save_model
from bullet.make_pybullet_env import make_pybullet_env

class LossWriter(object):
    def __init__(self, log_dir, fieldnames = ('r', 'l', 't'), header=''):
        
        assert log_dir is not None
        
        os.mkdir(log_dir + '/loss_monitor')
        filename = '{}/loss_monitor/loss_monitor.csv'.format(log_dir)
        self.f = open(filename, "wt")
        self.f.write(header)
        self.logger = csv.DictWriter(self.f, fieldnames=fieldnames)
        self.logger.writeheader()
        self.f.flush()

    def write_row(self, training_info):
        if self.logger:
            self.logger.writerow(training_info)
            self.f.flush()


def main():
    args = get_args()
    

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(1)

    device = torch.device(args.device)
    utils.cleanup_log_dir(args.log_dir)
    env_make = make_pybullet_env(args.task, log_dir=args.log_dir, frame_skip=args.frame_skip)
    envs = make_vec_envs(env_make, args.num_processes, args.log_dir, device, args.frame_stack)
    actor_critic = MetaPolicy(envs.observation_space, envs.action_space)
    loss_writer = LossWriter(args.log_dir, fieldnames= ('V_loss','action_loss','meta_action_loss','meta_value_loss','meta_loss', 'loss'))

    if args.restart_model:
        actor_critic.load_state_dict(
            torch.load(args.restart_model, map_location=device))

    actor_critic.to(device)

    agent = MetaPPO(
        actor_critic, args.clip_param, args.ppo_epoch,
        args.num_mini_batch, args.value_loss_coef,
        args.entropy_coef, lr=args.lr, eps=args.eps,
        max_grad_norm=args.max_grad_norm)

    obs = envs.reset()
    rollouts = RolloutStorage(args.num_steps, args.num_processes, obs, envs.action_space, actor_critic.recurrent_hidden_state_size)
    rollouts.to(device)  # they live in GPU, converted to torch from the env wrapper

    start = time.time()
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes

    for j in range(num_updates):

        ppo_rollout(args.num_steps, envs, actor_critic, rollouts)

        value_loss, meta_value_loss, action_loss, meta_action_loss, loss, meta_loss = ppo_update(
            agent, actor_critic, rollouts, args.use_gae, args.gamma, args.gae_lambda)
        
        loss_writer.write_row({'V_loss': value_loss.item(), 'action_loss': action_loss.item(), 'meta_action_loss':meta_action_loss.item(),'meta_value_loss':meta_value_loss.item(),'meta_loss': meta_loss.item(), 'loss': loss.item()} )
        
        if (j % args.save_interval == 0 or j == num_updates - 1) and args.log_dir != "":
            ppo_save_model(actor_critic, os.path.join(args.log_dir, "model.state_dict"), j)

        if j % args.log_interval == 0:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            s = "Update {}, num timesteps {}, FPS {} \n".format(
                j, total_num_steps, int(total_num_steps / (time.time() - start)))
            s += "Loss {:.5f}, meta loss {:.5f}, value_loss {:.5f}, meta_value_loss {:.5f}, action_loss {:.5f}, meta action loss {:.5f}".format(
                loss.item(), meta_loss.item(), value_loss.item(), meta_value_loss.item(), action_loss.item(), meta_action_loss.item())
            print(s, flush=True)


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--lr', type=float, default=7e-4, help='learning rate (default: 7e-4)')
    parser.add_argument(
        '--eps', type=float, default=1e-5,
        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument(
        '--alpha', type=float, default=0.99,
        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument(
        '--gamma', type=float, default=0.99,
        help='discount factor for rewards (default: 0.99)')
    parser.add_argument(
        '--use-gae', action='store_true', default=False,
        help='use generalized advantage estimation')
    parser.add_argument(
        '--gae-lambda', type=float, default=0.95,
        help='gae lambda parameter (default: 0.95)')
    parser.add_argument(
        '--entropy-coef', type=float, default=0.01,
        help='entropy term coefficient (default: 0.01)')
    parser.add_argument(
        '--value-loss-coef', type=float, default=0.5,
        help='value loss coefficient (default: 0.5)')
    parser.add_argument(
        '--max-grad-norm', type=float, default=0.5,
        help='max norm of gradients (default: 0.5)')
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument(
        '--num-processes', type=int, default=16,
        help='how many training CPU processes to use (default: 16)')
    parser.add_argument(
        '--num-steps', type=int, default=5,
        help='number of forward steps in A2C (default: 5)')
    parser.add_argument(
        '--ppo-epoch', type=int, default=4,
        help='number of ppo epochs (default: 4)')
    parser.add_argument(
        '--num-mini-batch', type=int, default=32,
        help='number of batches for ppo (default: 32)')
    parser.add_argument(
        '--clip-param', type=float, default=0.2,
        help='ppo clip parameter (default: 0.2)')
    parser.add_argument(
        '--log-interval', type=int, default=1,
        help='log interval, one log per n updates (default: 10)')
    parser.add_argument(
        '--save-interval', type=int, default=100,
        help='save interval, one save per n updates (default: 100)')
    parser.add_argument(
        '--eval-interval', type=int, default=None,
        help='eval interval, one eval per n updates (default: None)')
    parser.add_argument(
        '--num-env-steps', type=int, default=10e7,
        help='number of environment steps to train (default: 10e6)')
    parser.add_argument(
        '--log-dir', default='/tmp/ppo/',
        help='directory to save agent logs (default: /tmp/ppo)')
    parser.add_argument(
        '--recurrent-policy', action='store_true', default=False,
        help='use a recurrent policy')
    parser.add_argument(
        '--restart-model', default='',
        help='Restart training using the model given (Gianni)')
    parser.add_argument(
        '--device', default='cpu', help='Device to run on')
    parser.add_argument(
        '--frame-skip', type=int, default=0,
        help='Number of frame to skip for each action')
    parser.add_argument(
        '--frame-stack', type=int, default=1,
        help='Number of frame to stack in observation')
    parser.add_argument(
        '--task', default='LunarLander-v2',
        help='which of the pybullet task')
    args = parser.parse_args()
    args.log_dir = os.path.expanduser(args.log_dir)
    return args


if __name__ == "__main__":
    main()
