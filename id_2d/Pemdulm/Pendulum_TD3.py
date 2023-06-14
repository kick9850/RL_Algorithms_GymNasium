import argparse
from itertools import count
import os, sys, random
import numpy as np
import gymnasium as gym
import torch
from gym_RL_Algorithms.Basic.TD3 import TD3
parser = argparse.ArgumentParser()

parser.add_argument('--mode', default='train', type=str) # mode = 'train' or 'test'
parser.add_argument("--env_name", default="Pendulum-v1")  # OpenAI gym environment name， BipedalWalker-v2
parser.add_argument('--tau',  default=0.005, type=float) # target smoothing coefficient
parser.add_argument('--target_update_interval', default=1, type=int)
parser.add_argument('--iteration', default=5, type=int)

parser.add_argument('--learning_rate', default=3e-4, type=float)
parser.add_argument('--gamma', default=0.99, type=int) # discounted factor
parser.add_argument('--capacity', default=50000, type=int) # replay buffer size
parser.add_argument('--num_iteration', default=100000, type=int) #  num of  games
parser.add_argument('--batch_size', default=100, type=int) # mini batch size
parser.add_argument('--seed', default=1, type=int)

# optional parameters
parser.add_argument('--num_hidden_layers', default=2, type=int)
parser.add_argument('--sample_frequency', default=256, type=int)
parser.add_argument('--activation', default='Relu', type=str)
parser.add_argument('--render', default=False, type=bool) # show UI or not
parser.add_argument('--log_interval', default=50, type=int) #
parser.add_argument('--load', default=False, type=bool) # load model
parser.add_argument('--render_interval', default=100, type=int) # after render_interval, the env.render() will work
parser.add_argument('--policy_noise', default=0.2, type=float)
parser.add_argument('--noise_clip', default=0.5, type=float)
parser.add_argument('--policy_delay', default=2, type=int)
parser.add_argument('--exploration_noise', default=0.1, type=float)
parser.add_argument('--max_episode', default=2000, type=int)
parser.add_argument('--print_log', default=5, type=int)
args = parser.parse_args()



# Set seeds
# env.seed(args.seed)
# torch.manual_seed(args.seed)
# np.random.seed(args.seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
script_name = os.path.basename(__file__)
env = gym.make(args.env_name)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
min_Val = torch.tensor(1e-7).float().to(device) # min value

directory = './save_model/exp' + script_name + args.env_name +'./'
'''
Implementation of TD3 with pytorch 
Original paper: https://arxiv.org/abs/1802.09477
Not the author's implementation !
'''

def main():
    agent = TD3(state_dim,
                action_dim,
                max_action,
                args.capacity,
                directory,
                args.batch_size,
                args.policy_noise,
                args.policy_delay,
                args.noise_clip,
                args.gamma,
                args.tau)
    ep_r = 0

    if args.mode == 'test':
        agent.load()
        for i in range(args.iteration):
            state, info = env.reset()
            for t in count():
                action = agent.select_action(state)
                next_state, reward, done, _, info = env.step(np.float32(action))
                ep_r += reward
                env.render()
                if done or t ==2000 :
                    print("Ep_i \t{}, the ep_r is \t{:0.2f}, the step is \t{}".format(i, ep_r, t))
                    break
                state = next_state

    elif args.mode == 'train':
        print("====================================")
        print("Collection Experience...")
        print("====================================")
        if args.load: agent.load()
        for i in range(args.num_iteration):
            state, info = env.reset()
            for t in range(2000):
                action = agent.select_action(state)
                action = action + np.random.normal(0, args.exploration_noise, size=env.action_space.shape[0])
                action = action.clip(env.action_space.low, env.action_space.high)
                next_state, reward, done,_, info = env.step(action)
                ep_r += reward
                if args.render and i >= args.render_interval : env.render()
                agent.memory.push((state, next_state, action, reward, np.float64(done)))
                if i+1 % 10 == 0:
                    print('Episode {},  The memory size is {} '.format(i, len(agent.memory.storage)))
                if len(agent.memory.storage) >= args.capacity-1:
                    agent.update(10)

                state = next_state
                if done or t == args.max_episode -1:
                    agent.writer.add_scalar('ep_r', ep_r, global_step=i)
                    if i % args.print_log == 0:
                        print("Ep_i \t{}, the ep_r is \t{:0.2f}, the step is \t{}".format(i, ep_r, t))
                    ep_r = 0
                    break

            if i % args.log_interval == 0:
                agent.save()

    else:
        raise NameError("mode wrong!!!")

if __name__ == '__main__':
    main()