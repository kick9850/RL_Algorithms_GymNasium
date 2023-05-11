import gymnasium as gym

def main():
    # 환경 생성
    env = gym.make('CartPole-v1', render_mode='human')
    num_episodes = 10

    for episode in range(num_episodes):
        # 환경 초기화
        observation, info = env.reset()
        done = False
        total_reward = 0
        timesteps = 0

        while not done:
            # 환경을 시각화
            env.render()

            # 무작위 행동 선택
            action = env.action_space.sample()

            # 행동 수행 및 다음 상태, 보상, 종료 여부 얻기
            observation, reward, done, info, _ = env.step(action)

            total_reward += reward
            timesteps += 1

        print(f"Episode {episode + 1}: Timesteps = {timesteps}, Total Reward = {total_reward}")

    # 환경 종료
    env.close()

if __name__ == "__main__":
    main()
