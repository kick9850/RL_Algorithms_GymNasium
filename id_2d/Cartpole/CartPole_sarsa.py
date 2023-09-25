import gymnasium as gym
from gym_RL_Algorithms.Basic.Sarsa import Sarsa

def main():
    env = gym.make('CartPole-v1', render_mode='human')
    # Q 테이블 초기화
    state_space_size = env.observation_space.shape[0]
    action_space_size = env.action_space.n
    print(action_space_size)
    agent = Sarsa(env=env,
                  state_num=state_space_size,
                  action_num=action_space_size,
                  learning_rate=0.1,
                  discount_factor=0.99,
                  epsilon=0.2)

    num_episodes = 1000

    # Q-learning 알고리즘
    for episode in range(num_episodes):
        state, _ = env.reset()
        action = agent.get_action(state)

        done = False
        total_reward = 0
        agent.eps = max(0.1, 1.0 * (0.995 ** episode))
        while not done:
            # 환경에서 행동 수행하고 다음 상태 및 보상 얻기
            next_state, reward, done, _, _ = env.step(action)
            next_action = agent.get_action(next_state)
            agent.Update_sarsa(action=action, state=state, next_action=next_action, next_state=next_state,
                               reward=reward)
            state = next_state
            action = next_action
            total_reward += reward
        print(f"{episode} episode / 최종 누적 보상:", total_reward)


if __name__ == '__main__' :
    main()
    print('end')