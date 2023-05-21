# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 12:24:34 2019

@author: Z0014354
"""

import gymnasium as gym
from gym_RL_Algorithms.SAC import Agent
import time

import torch
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import numpy as np
import argparse


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-env", type=str, default="Pendulum-v1", help="Environment name")
    parser.add_argument("-info", type=str, default='sac', help="Information or name of the run")
    parser.add_argument("-ep", type=int, default=1000, help="The amount of training episodes, default is 100")
    parser.add_argument("-seed", type=int, default=0, help="Seed for the env and torch network weights, default is 0")
    parser.add_argument("-lr", type=float, default=5e-4,
                        help="Learning rate of adapting the network weights, default is 5e-4")
    parser.add_argument("-a", "--alpha", type=float,
                        help="entropy alpha value, if not choosen the value is leaned by the agent")
    parser.add_argument("-layer_size", type=int, default=256,
                        help="Number of nodes per neural network layer, default is 256")
    parser.add_argument("-repm", "--replay_memory", type=int, default=int(1e6),
                        help="Size of the Replay memory, default is 1e6")
    parser.add_argument("--print_every", type=int, default=100,
                        help="Prints every x episodes the average reward over x episodes")
    parser.add_argument("-bs", "--batch_size", type=int, default=256, help="Batch size, default is 256")
    parser.add_argument("-t", "--tau", type=float, default=1e-2, help="Softupdate factor tau, default is 1e-2")
    parser.add_argument("-g", "--gamma", type=float, default=0.99, help="discount factor gamma, default is 0.99")
    parser.add_argument("--saved_model", type=str, default=None, help="Load a saved model to perform a test run!")
    args = parser.parse_args()

    env_name = args.env
    seed = args.seed
    n_episodes = args.ep
    GAMMA = args.gamma
    TAU = args.tau
    HIDDEN_SIZE = args.layer_size
    BUFFER_SIZE = int(args.replay_memory)
    BATCH_SIZE = args.batch_size  # minibatch size
    LR_ACTOR = args.lr  # learning rate of the actor
    LR_CRITIC = args.lr  # learning rate of the critic
    FIXED_ALPHA = args.alpha
    saved_model = args.saved_model

    t0 = time.time()
    writer = SummaryWriter("./save_model/"+args.env+"/SAC")
    env = gym.make(env_name)
    action_high = env.action_space.high[0]
    action_low = env.action_space.low[0]

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    agent = Agent(state_size=state_size,
                  action_size=action_size,
                  random_seed=seed,
                  hidden_size=HIDDEN_SIZE,
                  action_prior="uniform", # "normal"
                  gamma=GAMMA,
                  lr_actor=LR_ACTOR,
                  lr_critic=LR_CRITIC,
                  buffer_size=BUFFER_SIZE,
                  batch_size=BATCH_SIZE,
                  alpha=FIXED_ALPHA,
                  tau=TAU)

    if saved_model != None:
        agent.actor_local.load_state_dict(torch.load(saved_model))
        agent.actor_local.eval()
        for i_episode in range(1):

            state, info = env.reset()
            state = state.reshape((1, state_size))

            while True:
                action = agent.act(state)
                action_v = action[0].numpy()
                action_v = np.clip(action_v * action_high, action_low, action_high)
                next_state, reward, done, _, info = env.step(action_v)
                next_state = next_state.reshape((1, state_size))
                state = next_state

                if done:
                    break
    else:
        max_t=500
        print_every = args.print_every

        scores_deque = deque(maxlen=100)
        average_100_scores = []
        for i_episode in range(1, n_episodes + 1):

            state, info = env.reset()
            state = state.reshape((1, state_size))
            score = 0
            for t in range(max_t):
                action = agent.act(state)
                action_v = action.numpy()
                action_v = np.clip(action_v * action_high, action_low, action_high)
                next_state, reward, done, _, info = env.step(action_v)
                next_state = next_state.reshape((1, state_size))
                agent.step(state, action, reward, next_state, done, t)
                state = next_state
                score += reward

                if done:
                    break

            scores_deque.append(score)
            writer.add_scalar("Reward", score, i_episode)
            writer.add_scalar("average_X", np.mean(scores_deque), i_episode)
            average_100_scores.append(np.mean(scores_deque))

            print(
                '\rEpisode {} Reward: {:.2f}  Average100 Score: {:.2f}'.format(i_episode, score, np.mean(scores_deque)),
                end="")
            if i_episode % print_every == 0:
                print('\rEpisode {}  Reward: {:.2f}  Average100 Score: {:.2f}'.format(i_episode, score,
                                                                                      np.mean(scores_deque)))

        torch.save(agent.actor_local.state_dict(), "./save_model/SAC/" + args.env + "/" + args.info + ".pt")
    t1 = time.time()
    env.close()
    print("training took {} min!".format((t1 - t0) / 60))

if __name__ == "__main__":
    main()