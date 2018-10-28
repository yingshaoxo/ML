import gym
import numpy as np
import time

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

def epsilon_greedy(Q, s, na):
    epsilon = 0.3
    p = np.random.uniform(low=0, high=1)
    print(p)
    if p > epsilon:
        return np.argmax(Q[s,:])
    else:
        return env.action_space.sample()

Q = np.zeros([env.observation_space.n, env.action_space.n]) # the Q table
print(Q)

lr = 0.5 # learning rate
y = 0.9 # discount factor lambda
eps = 10000 #total episodes being 10000

for i in range(eps):
    s = env.reset() # update states
    t = False
    while (True): 
        a = epsilon_greedy(Q, s, env.action_space.n)
        s_, r, t, _ = env.step(a) # take that action, get new states, reward
        if (r == 0): # no reward has been got
            if t == True:
                r = -5 # to give negative rewards when holes turn up
                Q[s_] = np.ones(env.action_space.n) * r # set Q table for the last state, in terminal(last) state, Q value equals the reward
            else:
                r = -1 # to give negative rewards to avoid long routes
                #Q[s_] = np.ones(env.action_space.n) * r # set Q table for the last state, in terminal(last) state, Q value equals the reward
        elif (r == 1): # got reward
            r = 100
            Q[s_] = np.ones(env.action_space.n) * r # set Q table for the last state, in terminal(last) state, Q value equals the reward
        Q[s, a] = Q[s, a] + lr * (r + y*np.max(Q[s_, a]) - Q[s, a]) # use formula to change the old state according to the new state
        s = s_
        if (t == True):  # it will run until it fall into hole, so t represent `hole`
            break

print("Q-table")
print(Q)
print()

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
