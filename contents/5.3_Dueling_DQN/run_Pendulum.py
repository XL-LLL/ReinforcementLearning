"""
Dueling DQN & Natural DQN comparison

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""

"""
原来我们会直接预估 Q值，现在我们需要预估两个值:S 值和 A 值。
S 值可以看成是该 state 下的 Q 值的平均数。A 值是有所限制的，
A 值的平均数为 0S 值与 A 值的和，就是原来的 Q值。
A+S=Q
在普通 DQN，当我们需要更新某个动作的 Q值，我们会直接更新 Q 网络
令这个动作的 Q 值提升。
Dueling DQN:在网络更新的时候，由于有 A 值之和必须为0的限制，
所以网络会优先更新S值。S 值是 Q 值的平均数，平均数的调整相当于一次性S下的所有 Q 值都更新一遍。
·所以网络在更新的时候，不但更新某个动作的Q值，而是把这个状态下，
所有动作的Q 值都调整一次。
目的是 这样，我们就可以在更少的次数让更多的值进行更新。
有同学可能会担心，这样调整最后的数值是对的吗?放心，在DuelingDQN，
我们只是优先调整S值。但最终我们的 target 目标是没有变的，所以我们最后更新出来也是对的。

A就像是我们之前的网络  但是在更行之前我们新添s值 优先更新 如果s不能满足条件在调整A
这个方法会比较早的达到目标
"""
import gym
from RL_brain import DuelingDQN
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


env = gym.make('Pendulum-v0')
env = env.unwrapped
env.seed(1)
MEMORY_SIZE = 3000
ACTION_SPACE = 25

sess = tf.Session()
with tf.variable_scope('natural'):
    natural_DQN = DuelingDQN(
        n_actions=ACTION_SPACE, n_features=3, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.001, sess=sess, dueling=False)

with tf.variable_scope('dueling'):
    dueling_DQN = DuelingDQN(
        n_actions=ACTION_SPACE, n_features=3, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.001, sess=sess, dueling=True, output_graph=True)

sess.run(tf.global_variables_initializer())


def train(RL):
    acc_r = [0]
    total_steps = 0
    observation = env.reset()
    while True:
        # if total_steps-MEMORY_SIZE > 9000: env.render()

        action = RL.choose_action(observation)

        f_action = (action-(ACTION_SPACE-1)/2)/((ACTION_SPACE-1)/4)   # [-2 ~ 2] float actions
        observation_, reward, done, info = env.step(np.array([f_action]))

        reward /= 10      # normalize to a range of (-1, 0)
        acc_r.append(reward + acc_r[-1])  # accumulated reward

        RL.store_transition(observation, action, reward, observation_)

        if total_steps > MEMORY_SIZE:
            RL.learn()

        if total_steps-MEMORY_SIZE > 15000:
            break

        observation = observation_
        total_steps += 1
    return RL.cost_his, acc_r

c_natural, r_natural = train(natural_DQN)
c_dueling, r_dueling = train(dueling_DQN)

plt.figure(1)
plt.plot(np.array(c_natural), c='r', label='natural')
plt.plot(np.array(c_dueling), c='b', label='dueling')
plt.legend(loc='best')
plt.ylabel('cost')
plt.xlabel('training steps')
plt.grid()

plt.figure(2)
plt.plot(np.array(r_natural), c='r', label='natural')
plt.plot(np.array(r_dueling), c='b', label='dueling')
plt.legend(loc='best')
plt.ylabel('accumulated reward')
plt.xlabel('training steps')
plt.grid()

plt.show()

