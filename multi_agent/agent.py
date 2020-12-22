import numpy as np
import torch
from policies.MADDPG.maddpg import MADDPG

class Agent:
    def __init__(self, agent_identifier, args) -> None:
        """
        Initializes the agent
        :param  args                arguments
        :param  agent_identifier    identifier of the current agent
        """
        self.args = args
        self.agent_id = agent_identifier
        self.policy = None  #TODO: Update this to mddpg


    def select_action(self, observation, noise_rate, epsilon):
        """
        Returns an action based on the given observation
        :param  observation current observation of the world
        :param  noise_rate  noise applied during the action selection process
        :param  epsilon     threshold for epsilon greedy selection
        :output action      action decision of the current agent
        """

        # Take the epsilon step
        if np.random.uniform() < epsilon:
            action = np.random.uniform(-self.args.high_action, self.args.high_action, self.args.action_shape[self.agent_id])
        else:
            # Take the greedy step  
            # TODO: Write me to do greedy action selection!
            action = np.random.uniform(-self.args.high_action, self.args.high_action, self.args.action_shape[self.agent_id])

        return action.copy()


    def train(self, transitions, other_agents):
        if self.policy is not None:
        # Trains the agent
            self.policy.train(transitions, other_agents)