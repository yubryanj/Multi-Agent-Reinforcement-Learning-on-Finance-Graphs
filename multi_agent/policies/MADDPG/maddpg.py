from policies.MADDPG.actor_critic import Actor, Critic
import torch
import os

class MADDPG:
    def __init__(self, args, agent_identifier) -> None:
        self.args = args
        self.agent_identifer = agent_identifier
        self.train_step = 0

        # Create the network
        self.actor_network = Actor(args, agent_identifier)
        self.critic_network = Critic(args)

        # Create the target networks
        self.actor_target_network = Actor(args, agent_identifier)
        self.critic_target_network = Critic(args)

        # load the weights into the target network -- both networks initializes to same place
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())

        # Create the optimizer
        self.actor_optimizer = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)

        #Create the directory structure for storing models
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        
        # Create the directory for saving models
        self.model_path = f'{self.args.save_dir}/{self.args.scenario_name}'
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        # Create the directory for this agent
        self.agent_path = f'{self.model_path}/agent_{agent_identifier}'
        if not os.path.exists(self.agent_path):
            os.mkdir(self.agent_path)

        # If a trained actor model is available
        if os.path.exists(f'{self.model_path}/actor_params.pkl'):
            self.actor_network.load_state_dict(torch.load(f'{self.model_path}/actor_params.pkl'))
            print(f' Agent {agent_identifier} successfully loaded actor network!')

        # If a trained critic model is available
        if os.path.exists(f'{self.model_path}/critic_params.pkl'):
            self.critic_network.load_state_dict(torch.load(f'{self.model_path}/critic_params.pkl'))
            print(f' Agent {agent_identifier} successfully loaded critic network!')


    def _soft_update_target_network(self):
        """
        Applies the update rule (1-alpha) * old weights + alpha * new weights to the target network
        :param  None
        :output None
        """

        # Update the target network
        for target_param, param in zip(self.actor_target_network.parameters(), self.actor_network.parameters()):
            target_param.data.copy_((1-self.args.tau) * target_param.data + self.args.tau * param.data)

        # Update the critic network
        for target_param, param in zip(self.critic_target_network.parameters(), self.critic_network.parameters()):
            target_param.data.copy_((1-self.args.tau) * target_param.data + self.args.tau * param.data)


    def train(self, transitions, other_agents):
        """
        Trains the network
        :param  transitions     dictionary of transitions
        :param  other_agents    dictionary of the behavior of other agents in the transition set
        """

        # Convert transitions which is an np array into a tensor
        for key in transitions.keys():
            transitions[key] = torch.tensor(transitions[key], dtype=torch.float32)
        
        # Retrieve the rewards
        rewards = transitions[f'rewards_{self.agent_identifer}']
        
        # Allocate lists to store the observation, action, and next observation triplets
        observations, actions, next_observations = [], [], []

        # Store the triplet for each agent in the given transition
        for agent_identifier in range(self.args.n_agents):
            observations.append(transitions[f'observations_{agent_identifier}'])
            actions.append(transitions[f'actions_{agent_identifier}'])
            next_observations.append(transitions[f'next_observations_{agent_identifier}'])

        # Calculate the target Q Value
        next_action = []
        # Resets the gradients in pytorch
        with torch.no_grad():
            index = 0 
            # Calculate the next action for each agent
            for agent_identifier in range(self.args.n_agents):
                if agent_identifier == self.agent_identifer:
                    next_action.append(self.actor_target_network(next_observations[agent_identifier]))
                else:
                    next_action.append(other_agents[index].policy.actor_target_network(next_observations[agent_identifier]))
                    index += 1
            # Calculate the next Q Value and remove the graph from the tensor
            next_q_value = self.critic_target_network(next_observations, next_action).detach()

            # Calculate the target q value for the gradient update as [ immediate reward + discount * future_reward ]; remove graph from the tensor
            target_q = (rewards.unsqueeze(1) + self.args.gamma * next_q_value).detach()

        """ Compute the loss and update """
        # Compute the q value for the current observation, action pair
        current_q = self.critic_network(observations, actions)

        # Square the error of the value from this experience 
        critic_loss = (target_q - current_q).pow(2).mean()

        # Actor loss
        # Calculate the actions of the agent using the current observations
        actions[self.agent_identifer] = self.actor_network(observations[self.agent_identifer])
        
        # Calcu
        actor_loss = - self.critic_network(observations, actions).mean()

        # Clear the actor gradients for back propagation
        self.actor_optimizer.zero_grad()
        # Conduct the backwards pass
        actor_loss.backward()
        # Update the parameters
        self.actor_optimizer.step()

        # Clear the critic gradients for the backward propagation
        self.critic_optimizer.zero_grad()
        # Conduct the backward pass
        critic_loss.backward()
        # Update the critic parameters
        self.critic_optimizer.step()

        # Update both networks
        self._soft_update_target_network()

        # Save the model
        if self.train_step > 0 and self.train_step % self.args.save_rate == 0:
            self.save_model(self.train_step)
        
        # Update the current training step
        self.train_step += 1


    def save_model(self, train_step):
        """
        Save the model
        :param  train_step  current training step, used for naming
        """
        print("Saving model")

        num = str(train_step // self.args.save_rate)
        model_path = os.path.join(self.args.save_dir, self.args.scenario_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        agent_path = os.path.join(model_path, f'agent_{self.agent_identifer}')
        if not os.path.exists(agent_path):
            os.makedirs(agent_path)
        torch.save(self.actor_network.state_dict(), f'{agent_path}/{num}_actor_params.pkl')
        torch.save(self.critic_network.state_dict(), f'{agent_path}/{num}_critic_params.pkl')
