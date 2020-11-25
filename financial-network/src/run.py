from financial_network.envs.financial_network_env import Financial_Network_Env

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

import os

DEBUG = True


if __name__ == "__main__":
    
    # PPO requires the environment to be wrapped in a DummyVecEnv
    environment = DummyVecEnv([lambda: Financial_Network_Env()])

    print(f'Beginning to train model...') if DEBUG else None
    # Train the model
    model = PPO2(MlpPolicy, environment, verbose=0,tensorboard_log="log/").learn(int(1e7))
    print(f'Training completed!') if DEBUG else None

    # Directory to the model results
    model_dir = f"models/"

    # Make the directory if the model does not exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # save the model
    model.save('models/weights.zip')


    # store the rewards
    rewards = []

    print(f"Evaluating performance: running 10 episode and storing the results.")
    for _ in range(10):
        observations = environment.reset()
        done = False
        reward = None
        while not done:
            action, _ = model.predict(observations)
            observations, reward, done, info = environment.step(action)
        
        rewards.append(reward[0])

    print(rewards)

        