import gym
import numpy as np


def main():
    # 환경 생성
    env = gym.make('Humanoid-v4', render_mode="human")
    num_episodes = 1000000

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        timesteps = 0

        while not done:
            # 환경을 시각화
            env.render()

            # 무작위 행동 선택
            action = env.action_space.sample()

            # 행동 수행 및 다음 상태, 보상, 종료 여부 얻기
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward
            timesteps += 1

            # 다음 상태로 이동
            state = next_state

        print(f"Episode {episode + 1}: Timesteps = {timesteps}, Total Reward = {total_reward}")


if __name__ == "__main__":
    main()
