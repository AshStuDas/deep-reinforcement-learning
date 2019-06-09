from monitor import interact
import gym
import numpy as np

typeRL = 'Expected SARSA'

if typeRL == 'Expected SARSA':
    from agent import Agent_ExpSARSA
elif typeRL == 'SARSA':
    from agent import Agent_SARSA
elif typeRL == 'Max SARSA': # Q-learning
    from agent import Agent_MaxSARSA

env = gym.make('Taxi-v2')
agent = Agent_ExpSARSA()
avg_rewards, best_avg_reward = interact(env, agent)