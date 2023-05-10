import gymnasium as gym
import numpy as np

# 환경 생성
env = gym.make('FrozenLake-v1', render_mode='human')

# Q-Table 초기화
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# 하이퍼파라미터 설정
learning_rate = 0.8
discount_factor = 0.95
epsilon = 0.2
num_episodes = 2000
max_steps_per_episode = 100

# 에피소드 반복
for episode in range(num_episodes):
    # 환경 초기화
    state, info = env.reset()
    done = False
    t = 0

    # 타임 스텝 반복
    while t < max_steps_per_episode and not done:
        # epsilon-greedy 정책에 따라 행동 선택
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state, :])

        # 환경에서 행동 수행
        next_state, reward, terminated, truncated, info = env.step(action)

        # epsilon-greedy 정책에 따라 다음 행동 선택
        if np.random.uniform(0, 1) < epsilon:
            next_action = env.action_space.sample()
        else:
            next_action = np.argmax(q_table[next_state, :])

        # Q-Table 업데이트
        q_table[state, action] = q_table[state, action] + learning_rate * \
                                 (reward + discount_factor * q_table[next_state, next_action] - q_table[state, action])

        # 상태 및 행동 업데이트
        state = next_state
        action = next_action
        t += 1
        done = terminated or truncated
    if episode % 100 == 0:
        print(f'Episode {episode} complete.')
        print(f'{episode} reward : {reward}')

print('Training complete.')

# 학습된 Q-Table 출력
print('Learned Q-Table:')
print(q_table)
