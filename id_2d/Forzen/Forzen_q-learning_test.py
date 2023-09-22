import gymnasium as gym
from gym_RL_Algorithms.Basic.Q_Learning import Qlearning


def main():
    env = gym.make('CartPole-v1', render_mode='human')
    # Q 테이블 초기화
    state_space_size = env.observation_space.shape[0]
    action_space_size = env.action_space.n

    agent = Qlearning(env=env,
                      state_num=state_space_size,
                      action_num=action_space_size,
                      lr=0.1,
                      gamma=0.99,
                      epsilon=0.2)

    num_episodes = 1000

    # Q-learning 알고리즘
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.get_action(state)

            # 환경에서 행동 수행하고 다음 상태 및 보상 얻기
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward
            agent.Update_Q_table(action=action,state=state,next_state=next_state,reward=reward)
            state = next_state
        # 에피소드가 끝날 때마다 epsilon 값을 감소시킴 (탐험을 점차 감소)
        agent.epsilon *= 0.995

        print(f"{episode} episode / 최종 누적 보상:", total_reward)


if __name__ == '__main__' :
    main()
    print('end')