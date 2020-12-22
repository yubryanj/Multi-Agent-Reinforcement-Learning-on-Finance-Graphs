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
        self.policy = MADDPG(args, agent_identifier)


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

            # Convert the observation vector into a tensor
            inputs = torch.tensor( observation, dtype=torch.float32).unsqueeze(0)

            # Gets the action from the actor given the observations
            pi = self.policy.actor_network(inputs).squeeze(0)

            # Concert the action tensor into an np array
            action = pi.cpu().numpy()

            # Generate some noise
            noise = noise_rate + self.args.high_action * np.random.randn(*action.shape)

            # Apply the noise
            action += noise
            
            # Clip the action to the acceptable bounds
            action = np.clip(action, -self.args.high_action, self.args.high_action)

        return action.copy()


    def learn(self, transitions, other_agents):
        if self.policy is not None:
        # Trains the agent
            self.policy.train(transitions, other_agents)