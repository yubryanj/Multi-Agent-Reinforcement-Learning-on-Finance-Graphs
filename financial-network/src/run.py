import gym
from financial_network.envs.financial_network_env import Financial_Network_Env

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, A2C

DEBUG = True


if __name__ == "__main__":
    # environment = gym.make('Financial_network-v0')
    
    # PPO requires the environment to be wrapped in a DummyVecEnv
    environment = DummyVecEnv([lambda: Financial_Network_Env()])

    print(f'Beginning to train model...') if DEBUG else None
    # Train the model
    model = PPO2(MlpPolicy, environment, verbose=1).learn(10000)
    print(f'Training completed!') if DEBUG else None

    print(f'Testing model on environment')
    observations = environment.reset()
    done = False
    while not done:
        action, _ = model.predict(observations)
        observations, rewards, done, info = environment.step(action)
        environment.render()