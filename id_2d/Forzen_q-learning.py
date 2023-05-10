import gymnasium as gym
import numpy as np

# 환경 생성
env = gym.make('FrozenLake-v1', render_mode='human')
env2 = gym.make('FrozenLake-v1', render_mode='human')
env3 = gym.make('FrozenLake-v1', render_mode='human')
env4 = gym.make('FrozenLake-v1', render_mode='human')

envs = [env, env2, env3, env4]

# Q-Table 초기화
q_table = []
q_table1 = np.zeros([env.observation_space.n, env.action_space.n])
q_table2 = np.zeros([env2.observation_space.n, env2.action_space.n])
q_table3 = np.zeros([env3.observation_space.n, env3.action_space.n])
q_table4 = np.zeros([env4.observation_space.n, env4.action_space.n])

# 하이퍼파라미터 설정
learning_rate = 0.8
discount_factor = 0.95
num_episodes = 2000
max_steps_per_episode = 100

# 에피소드 반복
for episode in range(num_episodes):
    # 환경 초기화
    state1, info = env.reset()
    state2, info = env2.reset()
    state3, info = env3.reset()
    state4, info = env4.reset()
    done = False
    t = 0

    # 타임 스텝 반복
    while t < max_steps_per_episode and not done:

        # 행동 선택
        action1 = np.argmax(q_table1[state1, :] + np.random.randn(1, env.action_space.n) * (1.0 / (episode + 1)))
        action2 = np.argmax(q_table2[state2, :] + np.random.randn(1, env2.action_space.n) * (1.0 / (episode + 1)))
        action3 = np.argmax(q_table3[state3, :] + np.random.randn(1, env3.action_space.n) * (1.0 / (episode + 1)))
        action4 = np.argmax(q_table4[state4, :] + np.random.randn(1, env4.action_space.n) * (1.0 / (episode + 1)))

        # 환경에서 행동 수행
        next_state1, reward1, terminated1, truncated1, info = env.step(action1)
        next_state2, reward2, terminated2, truncated2, info = env2.step(action2)
        next_state3, reward3, terminated3, truncated3, info = env3.step(action3)
        next_state4, reward4, terminated4, truncated4, info = env4.step(action4)

        # Q-Table 업데이트
        q_table1[state1, action1] = q_table1[state1, action1] + learning_rate * \
                                 (reward1 + discount_factor * np.max(q_table1[next_state1, :]) - q_table1[state1, action1])
        q_table2[state2, action2] = q_table2[state2, action2] + learning_rate * \
                                    (reward2 + discount_factor * np.max(q_table2[next_state2, :]) - q_table2[state2, action2])
        q_table3[state3, action3] = q_table3[state3, action3] + learning_rate * \
                                    (reward3 + discount_factor * np.max(q_table3[next_state3, :]) - q_table3[state3, action3])
        q_table4[state4, action4] = q_table4[state4, action4] + learning_rate * \
                                    (reward4 + discount_factor * np.max(q_table4[next_state4, :]) - q_table4[state4, action4])

        # 상태 업데이트
        state1 = next_state1
        state2 = next_state2
        state3 = next_state3
        state4 = next_state4
        t += 1

        if terminated1 == terminated2 == terminated3 == terminated4 == True:
            done = True

    if episode % 100 == 0:
        print(f'Episode {episode} complete.')
        q_tables = [q_table1, q_table2, q_table3, q_table4]
        q_table_max = np.maximum.reduce(q_tables)
        print(q_table_max)
        q_table1= q_table_max
        q_table2= q_table_max
        q_table3= q_table_max
        q_table4= q_table_max

print('Training complete.')

# 학습된 Q-Table 출력
print('Learned Q-Table:')
print(q_table_max)
