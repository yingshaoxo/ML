import gym
import numpy as np
import tensorflow as tf
import random

# load the environment
env = gym.make('FrozenLake-v0')

# create neural network
tf.reset_default_graph()
## tensors for inputs, weights, biases, Qtarget
inputs = tf.placeholder(shape=[None, env.observation_space.n], dtype=tf.float32)
W = tf.get_variable(name="W", dtype=tf.float32, shape=[env.observation_space.n, env.action_space.n], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.zeros(shape=[env.action_space.n]), dtype=tf.float32)

qpred = tf.add(tf.matmul(inputs, W), b)
apred = tf.argmax(qpred, 1)

qtar = tf.placeholder(shape=[1, env.action_space.n], dtype=tf.float32)
loss = tf.reduce_sum(tf.square(qtar - qpred))

train = tf.train.AdamOptimizer(learning_rate=0.001)
minimizer = train.minimize(loss)

# train the network
init = tf.global_variables_initializer()
y = 0.5 # discount factor
e = 0.3 # epsilon value for epsilon-greedy task
episodes = 10000

with tf.Session() as sess:
    sess.run(init)
    for i in range(episodes):
        s = env.reset()
        t_total = 0 # to calculate the sum of rewards in the current episode
        while (True):
            a_pred, q_pred = sess.run([apred, qpred], feed_dict={inputs:np.identity(env.observation_space.n)[s:s+1]})
            if np.random.uniform(low=0, high=1) < e:
                a_pred[0] = env.action_space.sample()
            s_, r, t, _ = env.step(a_pred[0])
            if r == 0:
                if t == True:
                    r = -5
                else:
                    r = -1
            elif r == 1:
                r = 5
            q_pred_new = sess.run(qpred, feed_dict={inputs:np.identity(env.observation_space.n)[s_:s_+1]})
            targetQ = q_pred
            max_qpredn = np.max(q_pred_new)
            targetQ[0, a_pred[0]] = r + y*max_qpredn
            _ = sess.run(minimizer, feed_dict={inputs: np.identity(env.observation_space.n)[s:s+1], qtar:targetQ})
            s = s_
            if t == True:
                break

    print("output after learning")
    print()
    s = env.reset()
    env.render()
    while 1:
        a = sess.run(apred, feed_dict={inputs: np.identity(env.observation_space.n)[s:s+1]})
        s_, r, t, _ = env.step(a[0])
        print("=====================")
        env.render()
        s = s_
        if t == True:
            break
    s = env.reset()
