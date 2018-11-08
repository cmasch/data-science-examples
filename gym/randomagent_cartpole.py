"""
Cartpole with RandomAgent

Cartpole has just two possible actions: going left or right. 
This implementation is using a RandomAgent to keep the pole upright 
for more than 180 steps. The parameters are random and fix for every episode.

@author: Christopher Masch
"""

import gym
import numpy as np

class RandomAgent:
    
    def __init__(self, env, steps):
        self.env   = env
        self.steps = steps
    
    def run(self, parameters):
        observation = self.env.reset()
        totalreward = 0
        for step in range(self.steps):
            action = 1 if np.matmul(parameters, observation) > 0 else 0
            observation, reward, done, info = self.env.step(action)
            totalreward += reward
            if done:
                break
            #env.render() # uncomment to visualize
        return totalreward

def train(agent, env, episodes=1000):
    for episode in range(episodes):
        parameters = np.random.rand(4)
        reward = agent.run(parameters)
        if reward >= 180:
            print('Winning with %i steps using parameters: %s'%(reward,parameters))
            break
        else:
            print('Losing after %i steps'%reward)
            
if __name__ == '__main__':
    env   = gym.make('CartPole-v0')
    
    print('########################')
    print('ENVIROMENT')
    print('Action space:',env.action_space)
    print('Observation space:',env.observation_space)
    print('########################\n')
    
    agent = RandomAgent(env, 250)
    
    for _ in range(5):
        print('#### ROUND %i'%(_+1))
        train(agent, env, episodes=50)
    env.close()