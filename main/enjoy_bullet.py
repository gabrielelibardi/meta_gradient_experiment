import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')

from ppo.model import MetaPolicy
from collections import deque
import torch
from matplotlib import pyplot as plt

from ppo.envs import make_vec_envs
from bullet.make_pybullet_env import make_pybullet_env
import time

parser = argparse.ArgumentParser(description='RL')
parser.add_argument(
    '--arenas-dir', default='', help='yaml dir')
parser.add_argument(
    '--load-model', default='',
    help='directory to save agent logs (default: )')
parser.add_argument(
    '--device', default='cuda', help='Cuda device  or cpu (default:cuda:0 )')
parser.add_argument(
    '--non-det', action='store_true', default=True,
    help='whether to use a non-deterministic policy')
parser.add_argument(
    '--recurrent-policy', action='store_true', default=False,
    help='use a recurrent policy')
parser.add_argument(
    '--realtime', action='store_true', default=False,
    help='If to plot in realtime. ')
parser.add_argument(
    '--silent', action='store_true', default=False, help='stop plotting ')
parser.add_argument(
    '--frame-skip', type=int, default=2,
    help='Number of frame to skip for each action')
parser.add_argument(
    '--frame-stack', type=int, default=4, help='Number of frame to stack')
parser.add_argument(
    '--reduced-actions', action='store_true', default=False,
    help='Use reduced actions set')
parser.add_argument(
    '--cnn', default='Fixup',
    help='Type of cnn. Options are CNN,Impala,Fixup,State')
parser.add_argument(
    '--state-stack', type=int, default=4,
    help='Number of steps to stack in states')
parser.add_argument(
    '--task', default='HalfCheetahPyBulletEnv-v0',
    help='which of the pybullet task')

args = parser.parse_args()
args.det = not args.non_det
device = torch.device(args.device)
env_make = make_pybullet_env(args.task, frame_skip=args.frame_skip)
env = make_vec_envs(env_make, 1, None, device, args.frame_stack)

env.render(mode="human")
base_kwargs = {'recurrent': args.recurrent_policy}
actor_critic = MetaPolicy(env.observation_space, env.action_space)
if args.load_model:
    actor_critic.load_state_dict(
        torch.load(args.load_model, map_location=device))
actor_critic.to(device)

recurrent_hidden_states = torch.zeros(1,
                                      actor_critic.recurrent_hidden_state_size).to(
    device)
masks = torch.zeros(1, 1).to(device)

obs = env.reset()
fig = plt.figure()
step = 0
S = deque(maxlen=100)
done = False
all_obs = []
episode_reward = 0

while not done:

    with torch.no_grad():
        value, action, action_log_prob, recurrent_hidden_states, dist_entropy = actor_critic.act(
            obs, recurrent_hidden_states, masks, deterministic=args.det)

    # Obser reward and next obs
    obs, reward, done, info = env.step(action)
    time.sleep(0.02)

    # p.resetDebugVisualizerCamera(distance, yaw, -20, humanPos)
    env.render("human")
    masks.fill_(0.0 if done else 1.0)

    step += 1
    episode_reward += reward

    if not args.silent:
        fig.clf()
        S.append(dist_entropy.item())
        plt.plot(S)
        plt.draw()
        plt.pause(0.03)

    term = 'goal' in info[0].keys()

    print(
        'Step {} Entropy {:3f} reward {:2f} value {:3f} done {}, bad_'
        'transition {} total reward {}'.format(
            step, dist_entropy.item(), reward.item(), value.item(), done, term,
            episode_reward.item()))

    if done:
        print("EPISODE: {} steps: ", episode_reward, step, flush=True)
        obs = env.reset()
        step = 0
        episode_reward = 0
        done = False
