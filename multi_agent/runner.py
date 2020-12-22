from tqdm import tqdm
from agent import Agent
from common.replay_buffer import Buffer
import torch
import os
import numpy as np
import matplotlib.pyplot as plt


class Runner:
    def __init__(self, args, env):
        self.args = args
        self.noise = args.noise_rate
        self.epsilon = args.epsilon
        self.episode_limit = args.max_episode_len
        self.env = env
        self.agents = self._init_agents()
        self.buffer = Buffer(args)
        self.save_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def _init_agents(self):
        agents = []
        for i in range(self.args.n_agents):
            agent = Agent(i, self.args)
            agents.append(agent)
        return agents

    def run(self):
        returns = []

        # Run this loop repeatedly
        for time_step in tqdm(range(self.args.time_steps)):
            # reset the environment
            if time_step % self.episode_limit == 0:
                # get the first sample
                state = self.env.reset()
            u = []
            actions = []

            with torch.no_grad():
                # For each agent
                for agent_id, agent in enumerate(self.agents):
                    # select an action
                    action = agent.select_action(state[agent_id], self.noise, self.epsilon)
                    # store the action 
                    u.append(action)
                    actions.append(action)

            # Gives a random action to each non-controlled agent
            for i in range(self.args.n_agents, self.args.n_players):
                actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
            
            # Take the next action; retrieve next tate, reward, done, and additional information
            next_state, reward, done, info = self.env.step(actions)

            # Store the episode in the replay buffer
            self.buffer.store_episode(state[:self.args.n_agents], u, reward[:self.args.n_agents], next_state[:self.args.n_agents])

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
                    agent.learn(transitions, other_agents)

            # Evaluate the learning
            if time_step >0 and time_step % self.args.evaluate_rate == 0:
                returns.append(self.evaluate())

            # Generate Noise
            self.noise = max(0.05, self.noise - 0.0000005)
            self.epsilon = max(0.05, self.noise - 0.0000005)

            # Save the weights
            np.save(f'{self.save_path}/returns.pkl', returns)



    def evaluate(self):
        returns = []
        for episode in range(self.args.evaluate_episodes):
            # reset the environment
            s = self.env.reset()
            rewards = 0
            # Obtain the results for a series of trainings
            for time_step in range(self.args.evaluate_episode_len):
                # Show the results
                self.env.render()
                actions = []
                # Zero out the gradients
                with torch.no_grad():
                    for agent_id, agent in enumerate(self.agents):
                        # Select the action for the given agent
                        action = agent.select_action(s[agent_id], 0, 0)
                        actions.append(action)
                
                # TODO: What is happening here? Is this just filling in code?
                for i in range(self.args.n_agents, self.args.n_players):
                    actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
                
                # Take the next action
                s_next, r, done, info = self.env.step(actions)
                # Update the rewards
                rewards += r[0]
                # Update the state
                s = s_next
            # Store the cumulative rewards
            returns.append(rewards)
            print('Returns is', rewards)
        return sum(returns) / self.args.evaluate_episodes
