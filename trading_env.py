import gym

import numpy as np
import pandas as pd

import quandl

import typing

import matplotlib.pyplot as plt

from collections import deque
import random
from datetime import timedelta

quandl.ApiConfig.api_key = "hgEYJRJUAoymNU_yDvbi"

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([-np.inf, np.inf], np.nan)
    return df.dropna()

class DataHandler:
    _col = "Adj. Close"
    
    @classmethod
    def read_from_csv(cls, root: str, tickers: typing.Union[str, typing.List[str]]):
        df = pd.read_csv(root, index_col="Date", parse_dates=True).sort_index(ascending=True)
        union = [ticker for ticker in tickers if ticker in df.columns]
        return df[union]
    
    @classmethod
    def get_price(cls, ticker: str, **kwargs):
        try:
            return quandl.get('WIKI/%s' % ticker, **kwargs)
        except:
            print("Failed to fetch data for {}".format(ticker))
            return None
    
    @classmethod
    def get_prices(cls, 
                   tickers: typing.List[str], 
                   start_date: str = None, 
                   end_date: str = None, 
                   freq: str = "B",
                   csv: str = None):
        if isinstance(csv, str):
            return cls.read_from_csv(csv, tickers).loc[start_date:end_date]
        else:
            data = {}
            for i, ticker in enumerate(tickers):
                tmp_df = cls.get_price(ticker, start_date=start_date, end_date=end_date)
                if tmp_df is not None:
                    data[ticker] = tmp_df[cls._col]
                
            df = pd.DataFrame(data)
            return df.sort_index(ascending=True).resample(freq).last()
    
    @classmethod
    def save_data(cls, df: pd.DataFrame, path: str):
        return df.to_csv(path)
                
class PortfolioVector:
    # Class for handling actions
    def __init__(self, tickers_count):
        # Lower bound
        self.low = -np.ones(tickers_count, dtype=float) * np.inf
        # Upper bound
        self.high = np.ones(tickers_count, dtype=float) * np.inf
        
    @property
    def shape(self):
        return self.low.shape
    
    def sample(self):
        # Random sample of PortfolioVector
        _vec = np.random.uniform(0, 1.0, self.shape[0])
        return _vec / np.sum(_vec) # sum(PortfolioVector) == 1
    
    def contains(self, x, tolerance=1e-5):
        x = np.array(x)
        shape_predicate = x.shape == self.shape
        range_predicate = (x >= self.low).all() and (x <= self.high).all()
        budget_constraint = np.abs(x.sum() - 1.0) < tolerance # sum ~<= 1
        return shape_predicate and range_predicate and budget_constraint
    
    def __repr__(self):
        return "PortfolioVector {}".format(self.shape)
    
    def __eq__(self, other):
        return np.allclose(self.low, other.low) and np.allclose(self.high, other.high)
        
class TradingEnv(gym.Env):
    class Record:
        def __init__(self, index, columns):
            self.actions = pd.DataFrame(columns=columns, index=index, dtype=float)
            self.actions.iloc[0] = np.zeros(len(columns))
            self.actions.iloc[0]["CASH"] = 1.0
            
            self.rewards = pd.DataFrame(columns=columns, index=index, dtype=float)
            self.rewards.iloc[0] = np.zeros(len(columns))
            
    # OpenAI gym like environment for trading
    def __init__(self, tickers=None, prices=None, trading_period="W-FRI", **kwargs):
        # tickers -> ["ticker_name"...]
        #self.tickers = tickers
        
        # trading_period -> "Period[D(day), W(week), M(month)...], WeekEnd[Fri, Sat, Sun]"
        self.trading_period = trading_period
        
        # prices [DataFrame] provided
        if prices is not None and isinstance(prices, pd.DataFrame):
            self._prices = clean(prices.resample(self.trading_period).last())
        
        # only tickers provided:
        # fetch prices
        elif tickers is not None and isinstance(tickers, list):
            self._prices = clean(self._get_prices(tickers, trading_period=self.trading_period, **kwargs))
        
        # number of tickers + cash
        tickers_count = len(self.tickers) + 1
        
        # Action space -> [ [-np.inf;np.inf] * tickers_count ] -> what agent can do (how much of each ticker in next timestep)
        self.action_space = PortfolioVector(tickers_count)
        
        # Observation space -> [ [-np.inf;np.inf] * tickers_count ] -> what agent sees (prices in concrete timestep)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (tickers_count, ), dtype=np.float32)
        
        # Cash column USD
        self._prices["CASH"] = 1.0
        
        # Return calculation -> [30,60] -> 2
        self._returns = self._prices.pct_change()
        
        # Current timeindex -> self.dates[self._counter] -> current timestamp
        self._counter = 0
        
        # Container for agents -> MultiAgent system
        self.agents = {}
        
        # DataFrame which tracks each agent wealth level
        self._pnl = pd.DataFrame(index=self.dates, columns=[agent.name for agent in self.agents])
        
        # Init fig, axes for plotting
        self._fig, self._axes = None, None
        
    @property
    def tickers(self):
        # get tickers
        return self._prices.columns.tolist()
    
    @property
    def dates(self):
        # get dates
        return self._prices.index
    
    @property
    def index(self):
        # current timestamp
        return self.dates[self._counter]
    
    @property
    def _max_episode_steps(self):
        # Number of timestamps available
        return len(self.dates)
            
    def _get_prices(self, tickers, trading_period, **kwargs):
        return DataHandler.get_prices(tickers, freq=trading_period, **kwargs)
        
    def _get_observation(self):
        ob = {}
        ob["prices"] = self._prices.loc[self.index, :]
        ob["returns"] = self._returns.loc[self.index, :]
        return ob
    
    def _get_reward(self, action):
        # [1.02, 0.99, 1.03] * [0;1] => reward
        return self._returns.loc[self.index] * action
    
    def _get_done(self):
        return self.index == self.dates[-1]
    
    def _get_info(self):
        return {}
        
    def register(self, agent):
        if not hasattr(agent, "name"):
            pass
        
        if agent.name not in self.agents:
            self.agents[agent.name] = self.Record(columns=self.tickers, index=self.dates)
        
    def unregister(self, agent):
        if agent is None:
            self.agents = {}
            return None
            
        if not hasattr(agent, "name"):
            raise ValueError
            
        if agent.name in self.agents:
            del self.agents[agent.name]
            
    def step(self, action):
        self._counter += 1
        observation = self._get_observation()
        done = self._get_done()
        info = self._get_info()
        
        if action.keys() != self.agents.keys():
            raise ValueError("Invalid agent name")
            
        reward = {}
        
        for name, A in action.items():
#             print("Action: ",A, type(A), type(A[0]), np.isnan(A))
            # if (np.isnan(A)[0]):
                # return np.array(observation["returns"]), {"DDPG":-100}, done, info
            if not self.action_space.contains(A):
                raise ValueError("Invalid action attempted {}".format(A))  
            self.agents[name].actions.loc[self.index] = A
            self.agents[name].rewards.loc[self.index] = self._get_reward(A)
            reward[name] = self.agents[name].rewards.loc[self.index].sum()
            # print(self.index)
            if (self._counter > 2):
                # print(self.agents[name].actions.loc[self.dates[self._counter - 1]])
                # print(self.agents[name].actions.loc[self.index])
                if ( (self.agents[name].actions.loc[self.dates[self._counter - 1]] != self.agents[name].actions.loc[self.index]).all() ):
                    reward[name] -= 0.0015
            
        return np.array(observation["returns"]), reward, done, info
    
    def reset(self):
        self._counter = 1
        ob = self._get_observation()
        return np.array(ob["returns"])
    
    def render(self):
        if self._fig is None or self._axes is None:
            self._fig, self._axes = plt.subplots(ncols=2, figsize=(19.2, 4.8))
        
        _pnl = pd.DataFrame(columns=self.agents.keys(), index=self.dates)
        
        for agent in self.agents:
            _pnl[agent] = (self.agents[agent].rewards.sum(axis=1) + 1).cumprod()
        
        self._axes[0].clear()
        self._axes[1].clear()
        
        self._prices.loc[:self.index].plot(ax=self._axes[0])
        _pnl.loc[:self.index].plot(ax=self._axes[1])
        
        self._axes[0].set_xlim(self.dates.min(),
                               self.dates.max())
        self._axes[0].set_title('Market Prices')
        self._axes[0].set_ylabel('Prices')
        self._axes[1].set_xlim(self._pnl.index.min(),
                               self._pnl.index.max())
        self._axes[1].set_title('PnL')
        self._axes[1].set_ylabel('Wealth Level')
        #plt.plot(self._axes[1])
        # draw throttled
        plt.pause(0.0001)
        self._fig.canvas.draw()
        self._prices.loc[:self.index].plot(figsize=(32.4,15.6))
        _pnl.loc[:self.index].plot(figsize=(32.4,15.6))
            
        
# start_date = "2015-01-01"
# trading_frequency = "W-FRI"
# tickers = ['AAPL', 'GE', 'JPM', 'MSFT', 'VOD', 'GS', 'TSLA', 'MMM']
# csv = "data.csv"

    
def run(env, agents):
    env.unregister(None)
    for agent in agents:
        env.register(agent)
    done = False
    observation = env.reset()
    while (not done):
        action = {agent.name:agent.act(observation) for agent in agents}
        next_observation, reward, done, info = env.step(action)
        observation = next_observation
        
# env = TradingEnv(tickers=tickers, csv="data.csv")
# random_agent = RandomAgent(9)
# uniform_agent = UniformAgent(9)
# winner_agent = WinnerAgent(9)
# losser_agent = LosserAgent(9)

# run(env, [random_agent, uniform_agent, winner_agent, losser_agent])
