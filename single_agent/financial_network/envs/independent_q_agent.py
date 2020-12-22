import numpy as np
from numpy.lib import financial
from policies.dqn import DQN


class IndependentQAgent:
    def __init__(self, environment):
        """
        :attribute  environment         environment in which the agents are operating
        :attribute  financial_graph     graph of the environment
        :attribute  action_space        valid actions that the agent can take
        :attribute  number_of_agents    number of agents to control
        :attribute  number_of_actions   number of agents to control

        :attribute  policy              policy with which to make decisions
        """
        self.environment        = environment
        self.financial_graph    = environment.envs[0].financial_graph
        self.action_space       = environment.envs[0].action_space
        self.observation_space  = environment.envs[0].observation_space
        self.number_of_agents   = self.financial_graph.number_of_banks
        self.number_of_actions  = self.action_space.shape[0]
        self.input_shape        = self.observation_space.shape

        self.policy             = DQN()
        pass

    def get_observations(self, agent_identifier, observations):
        """
        :param  agent_identifier    agent which the observations should relate to
        :param  observations        observations available to all agents at the time
        """

        # Currently, every agent can see the entire network
        # Thus, return the whole observation set
        observations = observations[0]

        # The agent has to know who they are
        # Thus append a one-hot encoding of agent vector
        agent_vector = np.zeros(shape=(self.number_of_agents))
        agent_vector[agent_identifier] = 1

        # Concatenate to create the agent input vector
        observation = np.vstack((observations, agent_vector))

        return observation




    def predict(self, observations):
        """
        Decides on the next action to take given a observation vector
        :param  observation observation vector from the environment
        """
        next_state = None

        # Preallocate the multi-agent action space
        actions = np.zeros(shape=(self.number_of_agents,self.number_of_agents))

        # Compute the action for each agent
        for agent_identifier in range(self.number_of_agents):
            
            # Get the observation of the individual agent
            observation = self.get_observations(agent_identifier, observations)
            
            # Get the action of the individual agent
            action = self.policy.take_action(observation)

            # Update the collective action space
            actions[agent_identifier, :] = action
        
        # Flatten the actions vector for Gym
        actions = actions.reshape(1,-1)

        return actions, next_state


    def learn(self, steps):
        """
        :param  steps   number of steps to train the agent for
        """
        pass


    def save(self, save_directory):
        """
        :param save_directory    directory to save the learned models
        """
        #TODO: Write me!
        pass


if __name__ == "__main__":
    pass
