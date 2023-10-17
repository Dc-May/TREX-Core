from TREX_Core._agent.rewards.utils import process_ledger

class Reward:
    def __init__(self, timing=None, ledger=None, market_info=None, **kwargs):
        self.type = 'grid_equivalent'
        self.__timing = timing
        self.__ledger = ledger
        self.__market_info = market_info

    async def calculate(self, last_deliver=None, market_transactions=None, grid_transactions=None, financial_transactions=None):
        """
        Parameters:
            dict : settlements
            dict : grid_transactions
        """
        if not last_deliver:
            if 'last_deliver' not in self.__timing:
                return 0.0
            else:
                last_deliver = self.__timing['last_deliver']

        if (market_transactions==None and grid_transactions==None and financial_transactions==None):
            market_transactions, grid_transactions, financial_transactions = \
                await process_ledger(last_deliver, self.__ledger, self.__market_info)

        # market_cost = sum([t[1] * t[2] for t in market_transactions if t[0] == 'bid'])
        market_bought_quantity = sum([t[1] for t in market_transactions if t[0] == 'bid'])
        market_profit = sum([t[1] * t[2] for t in market_transactions if t[0] == 'ask'])
        market_sold_quantity = sum([t[1] for t in market_transactions if t[0] == 'ask'])

        # if market_sold_quantity > 0 or market_bought_quantity > 0:
        #    print(market_transactions, grid_transactions, financial_transactions)
        grid_bought_quantity = (grid_transactions[0] + market_bought_quantity)
        grid_cost = grid_bought_quantity * grid_transactions[1]
        grid_sold_quantity = (grid_transactions[2] + market_sold_quantity)
        grid_profit = grid_sold_quantity * grid_transactions[3]

        financial_cost =  0
        financial_profit =  0

        total_profit = grid_profit + financial_profit
        total_cost = grid_cost + financial_cost
        reward = float(total_profit - total_cost) / 1000
        # reward = float(total_profit - total_cost)
        # print(last_deliver, market_transactions, grid_transactions, financial_transactions, reward)
        return reward
