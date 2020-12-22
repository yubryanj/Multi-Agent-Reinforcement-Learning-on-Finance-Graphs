import os
from tqdm import tqdm
from agent import Agent
from common.replay_buffer import Buffer
import torch
import os
import numpy as np
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, args, env) -> None:
        self.args = args
        self.noise = args.noise_rate
        self.epsilon = args.epsilon
        self.episode_limit = args.max_episode_length
        self.env = env
        self.agents = self._init_agents()
        self.buffer = Buffer(args)
        self.save_path = f'{self.args.save_dir}/{self.args.scenario_name}'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)


    def _init_agents(self):
        """
        Creates each agent with unique identifiers
        :params None
        :output agents  Vector of uniquely identified agents
        """
        agents = []
        for i in range(self.args.n_agents):
            agent = Agent(i, self.args)
            agents.append(agent)
        return agents

    
    def train(self):
        returns = []

        for time_step in tqdm(range(self.args.time_steps)):
            # Reset the environment
            if time_step % self.episode_limit == 0:
                state = self.env.reset()

            u = []
            actions = []

            with torch.no_grad():
                # for each agent
                for agent_identifier, agent in enumerate(self.agents):
                    # select an action
                    action = agent.select_action(state[agent_identifier], self.noise, self.epsilon)

                    # Store the actions
                    u.append(action)
                    actions.append(action)

                # Gives no action to the un-named agents
                # TODO: This may be removed! All agents will be given actions in the financial graph
                for i in range(self.args.n_agents, self.args.n_players):
                    actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])

                # Get the next state of the world
                next_state, reward, done, info = self.env.step(actions)

                # Add current transitoin into the buffer
                self.buffer.store_episodes(state[:self.args.n_agents], u, reward[:self.args.n_agents], next_state[:self.args.n_agents])

                # Update the state
                state = next_state

                # If there are enough samples in the buffer
                if self.buffer.current_size >= self.args.batch_size:

                    # Get a sample from the buffer of (s,a,r,s')
                    transitions = self.buffer.sample(self.args.batch_size)

                    # Train each agent
                    for agent in self.agents:

                        # Get a list of the other agents
                        other_agents = self.agents.copy()
                        other_agents.remove(agent)

                        # Train the current agent on the world transitions
                        agent.train(transitions, other_agents)

                # Evaluate the learning
                if time_step >0 and time_step % self.args.evaluate_rate == 0:
                    returns.append(self.evaluate())

                # Generate Noise
                self.noise = max(0.05, self.noise - 0.0000005)
                self.epsilon = max(0.05, self.noise - 0.0000005)

                # Save the weights
                np.save(f'{self.save_path}/returns.pkl', returns)


    def evaluate(self):
        """
        Evaluate the model by conducting a series of episodes
        :param  None
        :output average_reward  the average cumulative reward across all episodes
        """
        # Allocate list for storing each episode's cumulative rewards
        returns = []

        # Conduct a series of episdoes
        for _ in range(self.args.evaluate_episodes):

            # Reset the environment
            state = self.env.reset()
            rewards = 0

            # Obatin the results for a series of timesteps
            for _ in range(self.args.evaluate_episode_length):
                actions = []
                with torch.no_grad():
                    # Select an action for each agent
                    for agent_identifer, agent in enumerate(self.agents):
                        action = agent.select_action(state[agent_identifer], 0, 0)
                        actions.append(action)

                
                for i in range(self.args.n_agents, self.args.n_players):
                    actions.append([0,0,0,0,0])

                # Incurr the state transition by taking the action
                state, reward, _, _ = self.env.step(actions)

                # Update the cumulative rward
                rewards += reward[0]

            # Store the rewards
            returns.append(rewards)

        return np.mean(returns)