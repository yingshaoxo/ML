import gym
import numpy as np
import time

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

print("number of actions: ", env.action_space.n) # 4
print("number of states: ", env.observation_space.n) # 16
print()


episodes = 100 # total episodes being 10000
epsilon = 0.9 # 90% chance to explore new stuff. why? because we only got 100 chances (episodes=100) to take adventure.

y = 0.5 # discount factor lambda, it's nothing but a constant, you can use any math symbol to represent it
lr = 0.5 # learning rate

identity = np.identity(env.observation_space.n) # for quickly get a hot vector, like 0001000000000000


# building the fucking model
# http://adventuresinmachinelearning.com/reinforcement-learning-tutorial-python-keras/
# https://keon.io/deep-q-learning/
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

model = Sequential()
model.add(Dense(env.observation_space.n, input_dim=env.observation_space.n, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(env.action_space.n, activation="linear"))
model.compile(loss="mse", optimizer=Adam(lr=lr), metrics=['mae'])


for i in range(episodes):
    state = env.reset() # update states
    while (True): 
        # for new things, greedy algorithm
        p = np.random.uniform(low=0, high=1)
        if p > epsilon:
            action = np.argmax(model.predict(identity[state: state+1])) # 0 or 1 or 2 or 3
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

        target = reward + y * np.amax(model.predict(identity[new_state:new_state+1]))
        target_old = model.predict(identity[state:state + 1])
        target_old[0][action] = target
        model.fit(identity[state: state+1], target_old, epochs=1, verbose=0)

        state = new_state
        if (done == True):  # it will run until it fall into hole, so t represent `hole`
            break

print("Q-network")
print(model.summary())
print()


print("outut after learning")
print()
# let's check how much our agent has learned

s = env.reset() # get states
env.render() # set up env
while(True):
    a = np.argmax(model.predict(identity[s: s+1])) # get the best action from Q table (according to its value, we choose the max), return 0 or 1 or 2 or 3
    s_, r, t, _ = env.step(a) # take action, get new state, reward, if fall into hole, _
    print("========================")
    env.render()
    s = s_ # old state = new state
    if (t == True):
        break
