import gym
import numpy as np
import time

# building the fucking model
# http://adventuresinmachinelearning.com/reinforcement-learning-tutorial-python-keras/
from keras import Sequential
from keras.layers import InputLayer
from keras.layers import Dense

model = Sequential()
model.add(InputLayer(batch_input_shape=(1, 16)))
model.add(Dense(32, activation='sigmoid'))
model.add(Dense(4, activation="linear"))
model.compile(loss="mse", optimizer="adam", metrics=['mae'])


# paparing the fucking env
env = gym.make('FrozenLake-v0')

s = env.reset()
print("initial state: ", s)
print()

env.render()
print()

print(env.action_space)
print(env.observation_space)
print()

print("number of actions: ", env.action_space.n)
print("number of states: ", env.observation_space.n)
print()

epsilon = 0.3
lr = 0.5 # learning rate
y = 0.9 # discount factor lambda
eps = 10 #total episodes being 10000

identity = np.identity(env.observation_space.n) # for quickly get a hot vector, like 0001000000000000

for i in range(eps):
    state = env.reset() # update states
    t = False
    while (True): 
        # for new things, greedy algorithm
        p = np.random.uniform(low=0, high=1)
        if p > epsilon:
            action = np.argmax(model.predict(identity[state: state+1])) # 1 or 2 or 3 or 4
        else:
            action = env.action_space.sample()

        new_state, reward, done, _ = env.step(action) # take that action, get new states, reward
        if (reward == 0): # no reward has been got
            if done == True:
                reward = -5 # to give negative rewards when holes turn up
            else:
                reward = -1 # to give negative rewards to avoid long routes
        elif (reward == 1): # got reward
            reward = 100
        target = reward + y * np.max(model.predict(identity[new_state:new_state+1]))
        target_vec = model.predict(identity[state:state + 1])[0]
        target_vec[action] = target
        model.fit(identity[state: state+1], target_vec.reshape(-1, env.action_space.n), epochs=1, verbose=0)
        state = new_state
        if (done == True):  # it will run until it fall into hole, so t represent `hole`
            break

print("Q-network")
print(model.summary())
print()
exit()

print("outut after learning")
print()
# let's check how much our agent has learned

s = env.reset() # get states
env.render() # set up env
while(True):
    a = np.argmax(Q[s]) # get the best action from Q table (according to its value, we choose the max)
    s_, r, t, _ = env.step(a) # take action, get new state, reward, if fall into hole, _
    print("========================")
    env.render()
    s = s_ # old state = new state
    if (t == True):
        break
