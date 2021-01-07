import numpy as np

EPSILON = 1e-10

class Financial_Graph():

    def __init__(   self,
                    number_of_banks         = 3,\
                    cash_in_circulation     = 100,\
                    haircut_multiplier      = 0.5,\
                    system_value_mode       = 'solvent_banks'
                ) -> None:

        """
        N banks in the system
        :attribute  debts                       [NxN] matrix Row->Col represents the row owes the column
        :attribute  cash_position               [N] vector with the cash position of each bank 
        :attribute  number_of_banks             number of banks in the system
        :attribute  cash_in_circulation         amount of cash initially in the system
        :attribute  number_of_banks_in_default  the number of banks in default
        :attribute  number_of_solvent_banks     the number of solvent banks
        :attribute  system_value_mode           mode with which to value the state of the system
        :attribute  haircut_multiplier          the amount of discount applied to any defaulting bank
        """
        print(f"Initializing financial graph!")

        self.number_of_banks            = number_of_banks
        self.cash_in_circulation        = cash_in_circulation
        self.haircut_multiplier         = haircut_multiplier
        self.system_value_mode          = system_value_mode
        self.debts, self.cash_position  = self._initialize_banks(   number_of_banks=number_of_banks,\
                                                                    cash_in_circulation=cash_in_circulation
                                                                )

        self.number_of_defaulting_banks = self.get_number_of_defaulting_banks()
        self.number_of_solvent_banks    = self.get_number_of_solvent_banks()
        self.system_net_position        = self.get_system_net_position()

        print(f"Finished initializing financial graph!")


    def get_observation(self):
        """
        Generates the observation matrix displayed to the agent
        :param    None
        :output   np.array  [self.number_of_banks + 1, self.number_of_banks] 
                            matrix stacking the debt and cash position of each agent
        """
        return np.vstack((self.debts, self.cash_position))

        
    def _initialize_banks(  self,\
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
        debts = np.array([  [00.0,  00.0, 00.0],
                            [00.0,  00.0, 50.0],
                            [00.0,  2500.0, 00.0]])

        # Note, bank 2 is in default, with 20 units more debt than cash
        # Only bank 0 can save bank 2 with a transfusion of >= 100 or 50 percent of its current asset base
        # Bank 1 will be fine without doing anything.
        cash = np.array(  [2000.0, 50.0, 1500.0] )

        return debts, cash 



    def compute_system_value(self):
        """
        Computes the value of the system as a whole either
        'solvent_banks'       Number of solvent banks
        'system_liquidity'    Amount of cash remaining in the system

        :param  mode            which method 
        :output system_value    valuation of the current state of the system
        """

        assert self.system_value_mode in ['solvent_banks','system_net_position']
        
        if self.system_value_mode =='solvent_banks':
            # Reward is contingent on the number of solvent banks in the system
            
            # Obtain the list of solvent and default banks
            _, solvent_banks  = self.get_list_of_defaulting_and_solvent_banks()

            # Calculate the number of defaulting banks  
            system_value = len(solvent_banks)

        elif self.system_value_mode =='system_net_position':
            # Reward is contingent on the liquidity in the system
            system_value = self.get_system_net_position()

        else:
            assert(False)

        return system_value


    def get_list_of_defaulting_and_solvent_banks(self):
        """
        Returns a list of solvent and defaulting banks
        :params   None
        :output   banks_in_default  list of banks in default
        :output   solvent_banks     list of solvent banks
        """

        # Calculate the net position of each bank
        banks_net_position = self.cash_position - np.sum(self.debts,axis=1)

        # Retrieve a list of defaulting banks
        banks_in_default = [bank for bank in range(self.number_of_banks) if self.bank_is_in_default(banks_net_position[bank])]

        # Retrieve a list of solvent banks
        solvent_banks = [bank for bank in range(self.number_of_banks) if bank not in banks_in_default]

        return banks_in_default, solvent_banks


    def bank_is_in_default(self, net_position):
        """
        # function to improve code readability
        # Returns true if the bank's net position is negative, else False
        :param  net_position  net_position of the bank (i.e. asset - liability)
        :output Boolean       True/False depending on computation

        """
        if net_position <= 0:
            return True
        return False


    def process_creditors(self):
        """
        Process the creditor's position 
        :param  None
        :output None
        """

        solvent_banks = self.get_list_of_defaulting_and_solvent_banks()[1]

        for debtor_bank in solvent_banks:
            for creditor_bank, amount in enumerate(self.debts[debtor_bank]):

                # Creditor receives the cash from the debtor - increase cash position of creditor by the full amount of the debt
                self.cash_position[creditor_bank] += amount

                # Debtor pays creditor - decrease cash position of the debtor by the full amount of the debt
                self.cash_position[debtor_bank] -= amount

                # Debt is paid, clear the debt record
                self.debts[debtor_bank, creditor_bank] = 0 


    def process_debtors(self):
        """
        Proessing the debtor's positions
        :param  None
        :output None
        """
        # Get the list of defaulting banks
        defaulting_banks = self.get_list_of_defaulting_and_solvent_banks()[0]
        
        for bank in defaulting_banks:

            # The bank has defaulted and as such, has to liquidate its position at discounted prices
            self.cash_position[bank] *= self.haircut_multiplier

            # Retrieve the list of credits which this bank owes
            creditors = [_bank for _bank, amount in enumerate(self.debts[bank]) if amount > 0]

            # get the number of creditors
            number_of_creditors = len(creditors)

            if number_of_creditors > 0:
                # Calculate amount to distribute to each creditor
                amount_distributed_to_each_creditor = self.cash_position[bank] / number_of_creditors
            else:
                amount_distributed_to_each_creditor = 0

            # Close the debt owed to each creditor
            for creditor in creditors:
                self.cash_position[creditor] += amount_distributed_to_each_creditor

        # Remove the outstanding debt balance from defaulted banks as creditor's share has been allocated
        self.debts[defaulting_banks,:] = 0

        # Remove any outstanding debt balance from solvent banks to defaulted banks.
        self.debts[:,defaulting_banks] = 0

        # After paying out the cash, update the cash position of defaulted banks
        self.cash_position[defaulting_banks] = 0


    def take_action(self, action):
        """
        Distributes cash as per the action requested by the agents
        :param  action  np.matrix where each cell is the percentage of the cash position to allocate
        :output reward  
        """

        old_system_net_position = self.get_system_net_position()

        action  = action.reshape(self.number_of_banks, self.number_of_banks)

        # Normalize the cash distribution to 100%
        action = self._normalize_cash_distribution(action)

        n_rows, n_cols = action.shape

        # Allocate cash as requested by the banks    
        for from_bank in range(n_rows):
            for to_bank in range(n_cols):
                percentage_to_allocate          = action[from_bank, to_bank]
                amount                          = self.cash_position[from_bank] * percentage_to_allocate
                self.cash_position[from_bank]   -= amount
                self.cash_position[to_bank]     += amount

        new_system_net_position = self.get_system_net_position()

        reward = new_system_net_position - old_system_net_position

        return reward


    def _normalize_cash_distribution(self, action):
        """
        In the case the agent attempts to distribute more than 100%
        of thier cash position, then the system will normalize the amount
        to be distribute 100%
        :param  action  action matrix to be normalized
        :output action  normalized action matrix
        """
        row_sums  = action.sum(axis=1, keepdims=True)
        action    = action / (row_sums + EPSILON)

        return action


    def reset(self):
        """
        Resets the environment
        :param  None
        :output None
        """
        self.debts, self.cash_position = self._initialize_banks(  number_of_banks=self.number_of_banks,\
                                                                  cash_in_circulation=self.cash_in_circulation
                                                                )

    
    def clear(self):
        """
        Clear the financial system by distributring debt and credit
        Clearing is completed when the system stabilizes (i.e. solvent banks no longer change)
        :params None
        :output None
        """
        old_defaulting_banks, old_solvent_banks = self.get_list_of_defaulting_and_solvent_banks()

        while True:
            
            self.process_debtors()
            self.process_creditors()

            defaulting_banks, solvent_banks = self.get_list_of_defaulting_and_solvent_banks()

            if  old_defaulting_banks == defaulting_banks and \
                old_solvent_banks == solvent_banks:
                break


    def get_number_of_solvent_banks(self):
        """
        Returns the number of solvent banks
        """
        solvent_banks = self.get_list_of_defaulting_and_solvent_banks()[1]

        return len(solvent_banks)
        

    def get_number_of_defaulting_banks(self):
        """
        Returns the number of solvent banks
        """
        defaulting_banks = self.get_list_of_defaulting_and_solvent_banks()[0]

        return len(defaulting_banks)
        

    def get_system_net_position(self):
        """
        Returns the net position of the system (i.e. total cash - total debt)
        Consider -- that the system should be cleared and debt should be equal to 0
        :params     None
        :outputs    system_net_position     net cash position of the system post clearing
        """

        # Get the current system configuration
        backup_cash_position = np.copy(self.cash_position)
        backup_debt_position = np.copy(self.debts)

        # Apply the recursive clearing system
        self.clear()

        # Calculate the net position of the system after clearing
        system_net_position = np.sum(self.cash_position) - np.sum(self.debts)
        
        # Restore the original system state
        self.cash_position  = backup_cash_position
        self.debts          = backup_debt_position
        
        return system_net_position

if __name__ == "__main__":
    fg = Financial_Graph()
    fg.clear()

    print("Done")