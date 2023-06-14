import gymnasium as gym
from gym_RL_Algorithms.Basic.SAC_PER import Agent
import time

import torch
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import numpy as np
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-env", type=str, default="Pendulum-v1", help="Name of the Environment")
    parser.add_argument("-ep", type=int, default=10000, help="Number of Episodes to train, default = 100")
    parser.add_argument("-bs", "--buffer_size", type=int, default=int(1e6), help="Size of the Replay buffer, default= 1e6")
    parser.add_argument("-bsize", "--batch_size", type=int, default=256,
                        help="Batch size for the optimization process, default = 256")
    parser.add_argument("-seed", type=int, default=0, help="Seed for the env and torch network weights, default is 0")
    parser.add_argument("-lr", type=float, default=5e-4, help="Learning Rate, default = 5e-4")
    parser.add_argument("-g", type=float, default=0.99, help="discount factor gamma, default = 0.99")
    parser.add_argument("-wd", type=float, default=0, help="Weight decay, default = 0")
    parser.add_argument("-ls", "--layer_size", type=int, default=256,
                        help="Number of nodes per neural network layer, default = 256")
    parser.add_argument("--print_every", type=int, default=100,
                        help="Prints every x episodes the average reward over x episodes")
    parser.add_argument("-info", type=str, default='sac_per', help="tensorboard test run information")
    parser.add_argument("-device", type=str, default="cuda:0", help="Change to CPU computing or GPU, default=cuda:0")
    parser.add_argument("--saved_model", type=str, default=None, help="Load a saved model to perform a test run!")
    parser.add_argument("-t", "--tau", type=float, default=1e-2, help="Softupdate factor tau, default is 1e-2")

    args = parser.parse_args()

    saved_model = args.saved_model

    env_name = args.env
    env = gym.make(env_name)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    action_high = env.action_space.high[0]
    action_low = env.action_space.low[0]

    writer = SummaryWriter("./save_model/" + args.env+ "/SAC_PER")
    agent = Agent(state_size=state_size,
                  action_size=action_size,
                  random_seed=args.seed,
                  action_prior="uniform",
                  buffer_size=args.buffer_size,
                  batch_size=args.batch_size, # minibatch size
                  gamma=args.g,  # discount factor
                  tau=args.tau,  # for soft update of target parameters
                  lr_actor=args.lr,  # learning rate of the actor
                  lr_critic=args.lr,  # learning rate of the critic
                  weight_decay=args.wd,  # 1e-2        # L2 weight decay
                  hidden_size=args.layer_size
                  )  # "normal"

    start_time = time.time()
    if saved_model != None:
        agent.actor_local.load_state_dict(torch.load(saved_model))
        agent.actor_local.eval()
        for i_episode in range(1):

            state, info = env.reset()

            while True:
                action = agent.act(state)
                action_v = action[0].numpy()
                action_v = np.clip(action_v * action_high, action_low, action_high)
                next_state, reward, done, _, info = env.step(action_v)
                next_state = next_state
                state = next_state

                if done:
                    break
    else:
        n_episodes = 200
        max_t = 1000
        scores_deque = deque(maxlen=args.print_every)
        t = 0
        for i_episode in range(1, n_episodes + 1):

            state, info = env.reset()
            score = 0
            for t in range(max_t):
                t += 1
                action = agent.act(state)
                action_v = action[0].numpy()
                action_v = np.clip(action_v * action_high, action_low, action_high)
                next_state, reward, done, _, info = env.step(action_v)
                agent.step(state, action, reward, next_state, done, t)
                state = next_state
                score += reward

                if done:
                    break

            scores_deque.append(score)
            writer.add_scalar("Reward", score, i_episode)
            writer.add_scalar("average_X", np.mean(scores_deque), i_episode)

            print(
                '\rEpisode {} Reward: {:.2f}  Average100 Score: {:.2f}'.format(i_episode, score, np.mean(scores_deque)),
                end="")
            if i_episode % args.print_every == 0:
                print('\rEpisode {}  Reward: {:.2f}  Average100 Score: {:.2f}'.format(i_episode, score,
                                                                                      np.mean(scores_deque)))

        torch.save(agent.actor_local.state_dict(), "./save_model/SAC_PER/" + args.env + "/" + args.info + ".pt")
    end_time = time.time()
    env.close()
    print("Training took: {} min".format((end_time - start_time) / 60))
    # writer.add_hparams()

if __name__ == "__main__":
    main()