import gymnasium as gym
import numpy as np
import multiprocessing as mp

# FrozenLake 환경 생성
env = gym.make('FrozenLake-v1')
num_states = env.observation_space.n
num_actions = env.action_space.n

# Q-table 초기화
q_table = np.zeros([num_states, num_actions])

# 하이퍼파라미터 설정
num_episodes = 100
max_steps_per_episode = 100
learning_rate = 0.8
discount_rate = 0.95

# 학습 함수
def train(q_table, state, action, reward, next_state, done):
    # Q-learning 알고리즘
    old_value = q_table[state, action]
    next_max = np.max(q_table[next_state, :])
    new_value = (1 - learning_rate) * old_value + learning_rate * (reward + discount_rate * next_max)
    q_table[state, action] = new_value

    # 종료 여부에 따라 에피소드 반복 여부 결정
    if done:
        return False
    else:
        return True

# Q-learning 학습을 위한 함수
def q_learning_worker(q_table, start_idx, end_idx, result_queue):
    # 환경 초기화
    env = gym.make('FrozenLake-v1')
    for i_episode in range(start_idx, end_idx):
        # 에피소드 시작
        state, info = env.reset()
        t = 0
        done = False

        while t < max_steps_per_episode and not done:
            # 행동 선택
            action = np.argmax(q_table[state, :] + np.random.randn(1, num_actions) * (1.0 / (i_episode + 1)))
            # 다음 상태, 보상, 종료 여부 얻기
            next_state, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated
            # Q-table 학습
            episode_continue = train(q_table, state, action, reward, next_state, done)
            # 상태 업데이트
            state = next_state
            # 에피소드 반복 여부 결정
            if not episode_continue:
                break
            t += 1
        # 결과 전달
        result_queue.put(q_table)

# 메인 함수
if __name__ == '__main__':
    # 프로세스 개수 설정
    num_processes = 4
    # 결과를 저장할 큐 생성
    result_queue = mp.Queue()
    # 프로세스 리스트 생성
    process_list = []

    # Q-learning 학습 시작
    for i in range(num_processes):
        start_idx = i * num_episodes // num_processes
        end_idx = (i + 1) * num_episodes // num_processes
        process = mp.Process(target=q_learning_worker, args=(q_table, start_idx, end_idx, result_queue))
        process_list.append(process)
        process.start()

    # 학습 결과 수집
    for i in range(num_processes):
        q_table += result_queue.get()

    # 프로세스 종료
    for process in process_list:
        process.join()

    # 최종 Q-table 출력
    print(q_table)
