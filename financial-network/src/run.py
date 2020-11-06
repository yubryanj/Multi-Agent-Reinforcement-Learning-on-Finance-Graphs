import gym
import financial_network


if __name__ == "__main__":
    environment = gym.make('Financial_network-v0')

    observations = environment.reset()
    for _ in range(5):
        action = [0]
        observations, rewards, done, info = environment.step(action)
        environment.render()