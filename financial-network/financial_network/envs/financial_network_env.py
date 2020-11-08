import gym
import numpy as np
import pandas
from gym import error, spaces, utils

DEBUG = True

class Financial_Network_Env(gym.Env):
  metadata = {'render.modes': ['human']}


  def __init__(self,  number_of_banks     = 3,   \
                      cash_in_circulation = 1000,\
                      haircut_multiplier  = 0.9,\
                      terminal_timestep   = 1):
    print("Initializing environment") if DEBUG else None

    self.timestep             = 0
    self.terminal_timestep    = 1
    self.number_of_banks      = number_of_banks
    self.cash_in_circulation  = cash_in_circulation
    self.haircut_multiplier   = haircut_multiplier

    # Initialize the debt and cash positions of the banks
    self.debts, self.cash_position = self._initialize_banks(  number_of_banks=number_of_banks,\
                                                              cash_in_circulation=cash_in_circulation
                                                            )

    # Define the observation space such that at each time step, the debt and cash matrix is shown
    # The observation matrix has (self.number_of_banks + 1) rows reflecting 
    # ...the debt matrix and the cash position matrix vertically stacked
    # The number of columns reflects the individual banks
    self.observation_space = spaces.Box(low   = 0,\
                                        high  = self.cash_in_circulation,\
                                        shape = ( self.number_of_banks + 1, # Vertically stacked debt matrix and cash position
                                                  self.number_of_banks),    # Columns of the debt matrix (individual banks)
                                        dtype = np.float32
                                        )

    # Action space: Every bank can send cash to every other bank
    # First item defines which banks to send money to
    # Second item: Percentage of current AUM to be allocated
    self.action_space = spaces.Box( low   = 0,\
                                    high  = 1,\
                                    shape = ( self.number_of_banks,   # "From" Bank
                                              self.number_of_banks),  # "To" Bank
                                    dtype = np.float32
                                    )
    # TODO:Should be able to accept a [number_of_banks,number_of_banks] 
    # matrix of values [0,1] and normalize again in the step function
  
    print("Finished initializing environment") if DEBUG else None


  def step(self, action):
    # Check that a valid action is passed
    assert self.action_space.contains(action)

    # Increment the timestep counter
    self.timestep += 1

    observations  = self._get_observation()
    rewards       = self._calculate_rewards()
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
    self.debts, self.cash_position = self._initialize_banks(  number_of_banks=self.number_of_banks,\
                                                              cash_in_circulation=self.cash_in_circulation
                                                            )
    
    # Rebuild the observations
    observations = self._get_observation()

    return observations


  def render(self, mode='human'):
    """
    Outputs a representation of the environment
    :param  mode  defines the representation to output
    :output None
    """
    print("Rendering the environment") if DEBUG else None
    
    print(f'self.debts:\n{self.debts}')
    print(f'self.cash_position:\n{self.cash_position}')


  def close(self):
    """
    Executed on closure of the program
    TODO: Write me?
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


  def _initialize_banks(self,\
                        number_of_banks,\
                        cash_in_circulation
                        ):
    """
    Calculates the distribution of cash and debt across the entities
    :param    scalar    number_of_banks       Number of banks in the network
    :param    scalar    cash_in_circulation   Amount of cash to be allocated across the banks
    :outputs  np.matrix debts                 A matrix showing the interbank debt 
                                              network in notation: row owes column; 0s diagonal
    :outputs  np.array  cash                  A vector showing the amount of cash at each bank
    """

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
    debts = np.array([[0,10,20],[0,0,50],[30,50,0]])

    # Note, bank 1 is in default, with 20 units more debt than cash
    cash = np.array([200,100,30])

    return debts, cash 


  def _calculate_rewards(self):
    """
    Calculate the rewards to each entity at the end of each round
    :param    None
    :output   np.array  reward each agent receives as a function 
                        of their cash and debt profile
    """

    """
    # function to improve code readability
    # Returns true if the bank's net position is negative, else False
    """
    def bank_is_in_default(net_position):
      if net_position < 0:
        return True
      return False

    """
    Returns a list of solvent and defaulting banks
    """
    def get_list_of_defaulting_and_solvent_banks():

      # Calculate the net position of each bank
      banks_net_position = self.cash_position - np.sum(self.debts,axis=1)

      # Retrieve a list of defaulting banks
      banks_in_default = [bank for bank in range(self.number_of_banks) if bank_is_in_default(banks_net_position[bank])]

      # Retrieve a list of solvent banks
      solvent_banks = [bank for bank in range(self.number_of_banks) if bank not in banks_in_default]

      return banks_in_default, solvent_banks
    
    """
    Proessing the debtor's positions
    """
    def process_debtors(defaulting_banks):
      
      # Initialize the rewards vector for this function
      rewards = np.zeros(self.number_of_banks)

      for bank in defaulting_banks:
      
        # The bank has defaulted and as such, has to liquidate its position at discounted prices
        self.cash_position[bank] *= self.haircut_multiplier

        # Retrieve the list of credits which this bank owes
        creditors = [_bank for _bank, amount in enumerate(self.debts[bank]) if amount > 0]

        # get the number of creditors
        number_of_creditors = len(creditors)

        # Calculate amount to distribute to each creditor
        amount_distributed_to_each_creditor = self.cash_position[bank] / number_of_creditors

        # Allocate the capital balance to each outstanding creditor
        rewards[creditors] += amount_distributed_to_each_creditor
    
      # Remove the outstanding debt balance from defaulted banks as creditor's share has been allocated
      self.debts[defaulting_banks,:] = 0

      # Remove any outstanding debt balance from solvent banks to defaulted banks.
      self.debts[:,defaulting_banks] = 0

      # After paying out the cash, update the cash position of defaulted banks
      self.cash_position[defaulting_banks] = 0

      return rewards


    """
    Process the creditor's position 
    """
    def process_creditors(solvent_banks):
      for debtor_bank in solvent_banks:
        for creditor_bank, amount in enumerate(self.debts[debtor_bank]):

          # Creditor receives the cash from the debtor - increase cash position of creditor by the full amount of the debt
          self.cash_position[creditor_bank] += amount

          # Debtor pays creditor - decrease cash position of the debtor by the full amount of the debt
          self.cash_position[debtor_bank] -= amount

          # Debt is paid, clear the debt record
          self.debts[debtor_bank, creditor_bank] = 0 

      return self.cash_position

    """
    We assume that a bank a cash balance less than a debt balance is subject to default
    at the end of the episode. 
    
    We assume that the default happens before the bank receives any incoming cash.

    A defaulting firm receives a haircut on its cash balance -- due to liquidation prices -- and
    redistributes its resultant balance uniformly across its creditors
    """

    # pre allocate the rewards vector to be returned
    rewards = np.zeros(self.number_of_banks)

    # Get the list of banks in default
    defaulting_banks, solvent_banks = get_list_of_defaulting_and_solvent_banks()
    
    # Process the cash allocation for defaulting banks
    rewards += process_debtors(defaulting_banks)

    # Process the cash allocation for solvent banks
    rewards += process_creditors(solvent_banks)

    return rewards


  def _get_observation(self):
    """
    Generates the observation matrix displayed to the agent
    :param    None
    :output   np.array  [self.number_of_banks + 1, self.number_of_banks] 
                        matrix stacking the debt and cash position of each agent
    """
    return np.vstack((self.debts,self.cash_position))

if __name__ == "__main__":
  environment = Financial_Network_Env()

  observations = environment.reset()
  for _ in range(1):
      action = [0]
      observations, rewards, done, info = environment.step(action)
      
      print(f'rewards:\n{rewards}')

      if done:
        observations = environment.reset()
