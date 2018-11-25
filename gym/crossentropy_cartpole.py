"""
Cartpole with Cross-Entropy Method (CEM)

Policy-based model that predicts the best action by an observation. The model 
is based on two hidden layers with a final softmax which gives the probabilty 
for both actions (going left / right). I implemented the neural network in Keras.
You can find a quite similiar implementation in PyTorch by Maxim Lapan [1].

# References
- [1](https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On/blob/master/Chapter04/01_cartpole.py)

@author: Christopher Masch
"""

import gym
import numpy as np
from keras import layers
from keras import optimizers
from keras import Sequential
from keras.utils import to_categorical

class NeuralNet():
    
    def __init__(self, env, hidden_size, batch_size):
        self.env         = env
        self.hidden_size = hidden_size
        self.batch_size  = batch_size
        self.build()
    
    def build(self):
        """Build model"""
        obs_space = self.env.observation_space.shape[0]
        act_space = self.env.action_space.n
        
        self.model = Sequential()
        self.model.add(layers.Dense(self.hidden_size, input_dim=obs_space, activation='relu'))
        self.model.add(layers.Dense(self.hidden_size, activation='relu'))
        self.model.add(layers.Dense(act_space, activation='softmax'))
        self.model.compile(loss="categorical_crossentropy", optimizer=optimizers.Nadam())
    
    def run(self):
        observation  = self.env.reset()
        totalreward  = 0.0
        batches = []
        steps   = []
        while len(batches) < self.batch_size:
            action_prob = self.model.predict(np.array([observation])).flatten()
            action      = np.random.choice(self.env.action_space.n, p=action_prob)
            next_observation, reward, done, info = self.env.step(action)
            totalreward += reward
            steps.append(np.array([observation, action]))
            if done:
                batches.append([totalreward, steps])
                totalreward = 0.0
                steps = []
                next_observation = self.env.reset()
                continue
            #env.render() # uncomment to visualize
            observation = next_observation
        reward_mean = self.train(batches)
        return reward_mean
    
    def train(self, batches):
        """Train the model with best 20% of rewards"""
        rewards = np.array(batches)[:,0]
        obs_act = np.array(batches)[:,1]
        reward_mean  = np.mean(rewards)
        X = []
        y = []
        for idx, example in enumerate(rewards):
            if example < np.percentile(rewards, 80):
                continue
            batch = np.array(obs_act[idx])
            X.extend(batch[:,0])           # Observations
            y.extend(batch[:,1])           # Actions
            
        y = to_categorical(y)
        value = self.model.fit(np.array(X), y, epochs=1, verbose=0)
        print("Reward: %0.2f"%reward_mean,
              "Loss: %0.4f\n"%value.history['loss'][0])
        return reward_mean

if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    
    print('###########################')
    print('ENVIROMENT')
    print('Action space:',env.action_space)
    print('Observation space:',env.observation_space)
    print('###########################\n')
    
    agent = NeuralNet(env, hidden_size=64, batch_size=24)
    for episode in range(40):
        print("Episode:", episode)
        reward = agent.run()
        if reward >= 200.0:
            print("WIN! After %i rounds"%episode)
            break
    env.close()