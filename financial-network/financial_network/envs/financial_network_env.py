import gym
import numpy as np
import pandas
from gym import error, spaces, utils

DEBUG = True

class Financial_Network_Env(gym.Env):
  metadata = {'render.modes': ['human']}
  def __init__(self,  number_of_banks=3, \
                      cash_in_circulation=1000,
                      haircut_multiplier = 0.9):
    print("Initializing environment") if DEBUG else None

    self.number_of_banks      = number_of_banks
    self.cash_in_circulation  = cash_in_circulation
    self.haircut_multiplier   = haircut_multiplier

    self.debts, self.cash = self._initialize_banks( number_of_banks=number_of_banks,\
                                                    cash_in_circulation=cash_in_circulation
                                                  )
  
    print("Finished initializing environment") if DEBUG else None

  def step(self, action):
    observations = None
    rewards = self.calculate_rewards()
    done = False
    info = {}

    return observations, rewards, done, info

  def reset(self):
    print("Resetting the environment") if DEBUG else None
    pass

  def render(self, mode='human'):
    print("Rendering the environment") if DEBUG else None

  def close(self):
    print("Closing the environment") if DEBUG else None
    pass


  """
  Calculates the distribution of cash and debt across the entities
  :param    scalar    number_of_banks       Number of banks in the network
  :param    scalar    cash_in_circulation   Amount of cash to be allocated across the banks
  :outputs  np.matrix debts                 A matrix showing the interbank debt network in notation: row owes column; 0s diagonal
  :outputs  np.array  cash                  A vector showing the amount of cash at each bank
  """
  def _initialize_banks(self,\
                        number_of_banks,\
                        cash_in_circulation
                        ):

    """
    Insert distribution
    """
    # debts = np.random.uniform(size=(number_of_banks,number_of_banks))
    # cash  = np.random.uniform(size=(number_of_banks))

    # # Normalize the amount of cash in circulation
    # cash /=np.sum(cash)
    # cash *= cash_in_circulation

    # print(f'Debts: \n{debts}\n\nCash:\n{cash}') if DEBUG else None

    """
    Test version of initialization
    """
    # Debts are organized as row owes column
    debts = np.array([[0,10,20],[0,0,0],[30,50,0]])

    # Note, bank 1 is in default, with 20 units more debt than cash
    cash = np.array([10,100,30])

    return debts, cash 

  """
  Calculate the rewards to each entity at the end of each round
  :param  None
  :output np.array  reward each agent receives as a function of their cash and debt profile
  """
  def _calculate_rewards(self):

    """
    # function to improve code readability
    # Returns true if the bank's net position is negative, else False
    """
    def bank_is_in_default(net_position):
      if net_position < 0:
        return True
      return False

    # pre allocate the rewards vector to be returned
    rewards = np.zeros(self.number_of_banks)

    # Calculate the total debt each bank owes
    per_bank_debt = np.sum(axis=1)

    # Obtain the net position of each bank
    per_bank_net_position = per_bank_debt - self.cash

    """
    We assume that a bank a cash balance less than a debt balance is subject to default
    at the end of the episode. 
    
    We assume that the default happens before the bank receives any incoming cash.

    A defaulting firm receives a haircut on its cash balance -- due to liquidation prices -- and
    redistributes its resultant balance uniformly across its creditors
    """

    for bank, net_position in enumerate(per_bank_net_position):
      if bank_is_in_default(net_position):
        


    return rewards