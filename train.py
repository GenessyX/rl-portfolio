from trading_env import TradingEnv
from ddpg_agent import DDPGAgent
# from agents import RandomAgent, UniformAgent, LosserAgent, WinnerAgent
from dqn_agent import DQNAgent
from config import *
import numpy as np
from utils import cardinalities, run


def ddpg_train():
    env = TradingEnv(tickers=tickers, start_date=training_start_date, end_date=training_end_date)
    agent = DDPGAgent(alpha=0.0025, beta=0.025, input_dims=[len(tickers)+1], tau=0.001, env=env,
        batch_size=64, layer1_size=400, layer2_size=300, n_actions=len(tickers)+1)
    agent.load_models()
    env.register(agent)
    np.random.seed(42)
    score_history = []
    for i in range(1000):
        done = False
        score = 0
        obs = env.reset()
        while not done:
            act = agent.choose_action(obs)
            new_state, reward, done, info = env.step({agent.name: act})
            reward = reward["DDPG"]
            agent.remember(obs, act, reward, new_state, int(done))
            agent.learn()
            score += reward
            obs = new_state
        score_history.append(score)
        print("episode: {}; score: {}; 100 average: {}".format(i, score, np.mean(score_history[-100:])))
        if i % 25 == 0:
            agent.save_models()
    pass

def dqn_train():
    env = TradingEnv(tickers=tickers, start_date=training_start_date, end_date=training_end_date)
    observation_space, action_space = cardinalities(env)
    agent = DQNAgent(observation_space, action_space, binary=True, hidden_layer=50, capacity=3000)
    rewards, actions = run(env, agent, 250, True, False)

def main():
    ddpg_train()

if __name__ == "__main__":
    main()