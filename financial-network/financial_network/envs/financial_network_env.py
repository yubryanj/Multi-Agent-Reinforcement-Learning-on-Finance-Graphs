import gym
import numpy as np
from gym import error, spaces, utils
from .financial_graph import Financial_Graph

DEBUG = False

class Financial_Network_Env(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self,  number_of_banks     = 3,   \
                      cash_in_circulation = 1000,\
                      haircut_multiplier  = 0.9,\
                      terminal_timestep   = 1):

    print("Initializing environment") if DEBUG else None

    self.timestep             = 0
    self.terminal_timestep    = terminal_timestep

    self.observation_space = spaces.Box(low   = 0,\
                                        high  = cash_in_circulation, \
                                        shape = (number_of_banks + 1, number_of_banks), \
                                        dtype = np.float32
                                        )

    # Defines all possible actions
    # NOTE: Assumes all-to-all connection between banks
    self.action_space = spaces.Box(   low   = 0,\
                                      high  = 1,\
                                      shape = ( number_of_banks * number_of_banks,), 
                                      dtype = np.float32
                                      )

    # Initialize the debt and cash positions of the banks
    self.financial_graph = Financial_Graph( number_of_banks=number_of_banks,\
                                            cash_in_circulation=cash_in_circulation,\
                                            haircut_multiplier = haircut_multiplier,\
                                          )

    print("Finished initializing environment") if DEBUG else None


  def step(self, action):
    """
    Takes one step in the environment
    :param  action      action submitted by the agent
    :output observation Vector representing updated environment state
    :output rewards     Reward vector for agent
    :output done        True/False depending if the episode is finished
    :output info        additional information
    """

    # Increment the timestep counter
    self.timestep += 1

    # Allocate the cash as the agents requested
    rewards = self.financial_graph.take_action(action)
    observations  = self.financial_graph.get_observation()
    done          = self._determine_if_episode_is_done()
    info          = {}

    return observations, rewards, done, info


  def reset(self):
    """
    Resets the environment to the initial state
    :param  None
    :output observations  np.matrix representing initial environment state
    """
    print("Resetting the environment") if DEBUG else None

    # Reset the timestep counter
    self.timestep = 0

    # Reset the environment
    self.financial_graph.reset()

    # Retrieve the observations of the resetted environment
    observations = self.financial_graph.get_observation()

    return observations


  def render(self, mode='human'):
    """
    Outputs a representation of the environment
    :param  mode  defines the representation to output
    :output None
    """

    if mode == 'human':
      print("Rendering the environment") if DEBUG else None
      print(f'At timestep {self.timestep}')
    else:
      pass


  def close(self):
    """
    Executed on closure of the program
    :param  None
    :output None
    """
    print("Closing the environment") if DEBUG else None
    pass


  def _determine_if_episode_is_done(self):
    """
    Returns a boolean determining whether this episode is completed
    :param    None
    :output   Boolean   True/False depending on whether the episode is finished
    """
    if self.timestep >= self.terminal_timestep: 
        return True
    
    return False


if __name__ == "__main__":
  environment = Financial_Network_Env(number_of_banks     = 3,   \
                                      cash_in_circulation = 1000,\
                                      haircut_multiplier  = 0.5,\
                                      terminal_timestep   = 1)

  observations = environment.reset()
  for _ in range(1):
    action = np.array([ [0.5,   0.0,  0.5],
                        [0.0,   1.0,  0.0],
                        [0.0,   0.0,  1.0]]
                        )
    observations, rewards, done, info = environment.step(action)
    
    print(f'Episode finished? {done}\nAgent rewards {rewards}')

    if done:
      observations = environment.reset()
