from collections import namedtuple
from itertools import count
import gymnasium as gym
import torch
from gym_RL_Algorithms.Basic.PPO import PPO

def main():
    # Parameters
    gamma = 0.99
    render = False
    seed = 1
    log_interval = 10

    env = gym.make('CartPole-v1', render_mode='human').unwrapped
    num_state = env.observation_space.shape[0]
    num_action = env.action_space.n
    torch.manual_seed(seed)
    Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state'])

    agent = PPO(num_state, num_action, gamma, log_interval)
    for i_epoch in range(1000):
        state, info = env.reset()
        if render: env.render()

        for t in count():
            action, action_prob = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            trans = Transition(state, action, action_prob, reward, next_state)
            if render: env.render()
            agent.store_transition(trans)
            state = next_state

            if done:
                if len(agent.buffer) >= agent.batch_size: agent.update(i_epoch)
                agent.writer.add_scalar('liveTime/livestep', t, global_step=i_epoch)
                break


if __name__ == '__main__':
    main()
    print("end")