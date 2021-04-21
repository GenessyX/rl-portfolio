import numpy as np
from trading_env import PortfolioVector

class Agent:
    _id = "test"

    def __init__(self, **kwargs):
        raise NotImplementedError

    #######
    # API
    #######

    @property
    def name(self):
        return self._id

    def begin_episode(self, observation):
        pass

    def act(self, observation):
        raise NotImplementedError

    def observe(self, observation, action, reward, done, next_observation):
        pass

    def end_episode(self):
        pass

    # def fit(self, env, num_episodes=1, verbose=False):
    #     return run(env, self, num_episodes, True, verbose)
    
class RandomAgent(Agent):
    _id = "random"
    def __init__(self, tickers_count):
        self.tickers_count = tickers_count
    
    def act(self, observation):
        return PortfolioVector(self.tickers_count).sample()
    
class UniformAgent(Agent):
    _id = "uniform"
    def __init__(self, tickers_count):
        self.tickers_count = tickers_count
        
    def act(self, observation):
        return np.array([1/self.tickers_count for i in range(self.tickers_count)])
    
class WinnerAgent(Agent):
    _id = "winner"
    def __init__(self, tickers_count):
        self.tickers_count = tickers_count
        
    def act(self, observation):
        action = np.zeros(self.tickers_count)
        action[np.argmax(observation)] = 1
        return action
    
class LosserAgent(Agent):
    _id = "losser"
    def __init__(self, tickers_count):
        self.tickers_count = tickers_count
        
    def act(self, observation):
        action = np.zeros(self.tickers_count)
        action[np.argmin(observation)] = 1
        return action