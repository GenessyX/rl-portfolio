from trading_env import TradingEnv
from ddpg_agent import DDPGAgent
from agents import RandomAgent, UniformAgent, LosserAgent, WinnerAgent
from dqn_agent import DQNAgent
from config import *
from utils import cardinalities, run


def main():
    action_space = len(tickers)+1
    trading_env = TradingEnv(tickers=tickers, start_date=test_start_date, end_date=test_end_date)
    observation_space, action_space = cardinalities(trading_env)
    random_agent = RandomAgent(action_space)
    uniform_agent = UniformAgent(action_space)
    winner_agent = WinnerAgent(action_space)
    losser_agent = LosserAgent(action_space)
    ddpg_agent = DDPGAgent(alpha=0.0025, beta=0.025, input_dims=[9], tau=0.001, env=trading_env, 
                  batch_size=64, layer1_size=400, layer2_size=300, n_actions=9, training_mode=False)
    ddpg_agent.load_models()

    train_env = TradingEnv(tickers=tickers, start_date=training_start_date, end_date=training_end_date)
    dqn_agent = DQNAgent(observation_space, action_space, binary=True, hidden_layer= 50,capacity= 3000)
    rewards, actions = run(train_env, dqn_agent, 250, True, False)

    rewards = {'random': 0, 'uniform': 0, 'winner': 0, 'losser': 0, 'DDPG': 0, 'DQN': 0}

    trading_env.register(ddpg_agent)
    trading_env.register(dqn_agent)
    agents = [random_agent, uniform_agent, winner_agent, losser_agent]
    for agent in agents:
        trading_env.register(agent)
    done = False
    observation = trading_env.reset()
    while (not done):
        action = {agent.name:agent.act(observation) for agent in agents}
        ddpg_action = ddpg_agent.choose_action(observation)
        # print(ddpg_action)
        action[ddpg_agent.name] = ddpg_action
        dqn_action = dqn_agent.act(observation)
        action[dqn_agent.name] = dqn_action
        next_observation, reward, done, info = trading_env.step(action)
        observation = next_observation
        for r in reward:
            rewards[r] += reward[r]
    print(rewards)
    trading_env.render()
    input()


def ddpg_train():
    pass


if __name__ == "__main__":
    main()

