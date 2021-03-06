from financial_network.envs.financial_network_env import Financial_Network_Env

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

import os
import numpy as np

DEBUG = False


if __name__ == "__main__":

    number_of_banks = 0

    # Directory to the model results
    model_dir = f"../models/"
    weights_dir = f"../models/weights_{number_of_banks}_agents.zip"
    
    # PPO requires the environment to be wrapped in a DummyVecEnv
    environment = DummyVecEnv([lambda: Financial_Network_Env(number_of_banks=number_of_banks)])

    # Define the agent model
    agent = PPO2(MlpPolicy, environment, verbose=0)

    # If there are not weights already
    if not os.path.exists(weights_dir):

        print(f'Beginning to train model...') if DEBUG else None
        
        # Train the model
        agent.learn(int(5e5))

        print(f'Training completed!') if DEBUG else None

        # Make the directory if the model does not exist
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
    
        # save the model
        agent.save(weights_dir)
    
    else:
        print(f"Loading the weights!") if DEBUG else None
        
        # Load the learned weights
        agent = PPO2.load(weights_dir)

        print(f"Weights successfully loaded!") if DEBUG else None

    # store the rewards
    net_positions = []

    print(f"Evaluating performance: running 100 episode and storing the results.")
    
    for _ in range(100):
        # Specify the environment
        observations = environment.envs[0].reset(evaluate=True)

        # Add a dimension because the dummyvec is a  wrapper
        observations = np.expand_dims(observations,0)

        done = False
        net_position = None

        while not done:
            action, _ = agent.predict(observations)
            observations, reward, done, info = environment.step(action)

            net_position = int(info[0]['net_position'])
        
        # Save the results of the episode
        net_positions.append(net_position)
        
        # Reset the counter
        net_position = None

    print(f'Average net_position: {np.mean(net_positions)}')

        
