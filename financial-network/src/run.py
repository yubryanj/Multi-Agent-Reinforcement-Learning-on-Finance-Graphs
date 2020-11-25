from financial_network.envs.financial_network_env import Financial_Network_Env

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

DEBUG = True


if __name__ == "__main__":
    
    # PPO requires the environment to be wrapped in a DummyVecEnv
    environment = DummyVecEnv([lambda: Financial_Network_Env()])

    print(f'Beginning to train model...') if DEBUG else None
    # Train the model
    model = PPO2(MlpPolicy, environment, verbose=1,tensorboard_log="log/").learn(int(1e7))
    print(f'Training completed!') if DEBUG else None

    # save the model
    model.save('models/weights.zip')


    # store the rewards
    rewards = []

    print(f"Evaluating performance: running 10 episode and storing the results.")
    for _ in range(10):
        observations = environment.reset()
        done = False
        while not done:
            action, _ = model.predict(observations)
            observations, reward, done, info = environment.step(action)
        
        rewards.append(reward[0])

    print(rewards)

        