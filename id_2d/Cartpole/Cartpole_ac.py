import gymnasium as gym
from itertools import count
import torch
from gym_RL_Algorithms.Basic.AC import Agent

def main():
    # Parameters
    env = gym.make('CartPole-v1')
    env = env.unwrapped

    torch.manual_seed(1)

    state_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    # Hyperparameters
    learning_rate = 0.01
    gamma = 0.99
    episodes = 2000
    render = False

    agent = Agent(state_space=state_space,
                  action_space=action_space,
                  hidden_size=32,
                  lr=learning_rate,
                  gamma=gamma
                  )
    running_reward = 10

    live_time = []
    for i_episode in range(episodes):
        state, info = env.reset()
        for t in count():
            action = agent.select_action(state)
            state, reward, done, _, info = env.step(action)
            if render:
                env.render()
            agent.model.rewards.append(reward)

            if done or t >= 1000:
                break
        running_reward = running_reward * 0.99 + t * 0.01
        live_time.append(t)
        agent.plot(live_time)
        if i_episode % 100 == 0:
            print(f"episode {i_episode}, reward : {reward}")
            #modelPath = './save_model/AC_CartPole_Model/ModelTraing'+str(i_episode)+'Times.pkl'
            #torch.save(model, modelPath)
        agent.finish_episode()

if __name__ == '__main__':
    main()