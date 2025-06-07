"""
Prioritized_Repla这个算法的目的是为解决困难样本   使得困难样本被多学习几次
multi-step 算法就是将更多的状态存在buffer里面   就是网络的输入更多了  计算真实值和预测值时候就需要回溯了累加了
Noisy Net 和我们之前让选择具有探索性 我们这里直接在在网络的参数上加参数
Distributional Q-function因为 Q 值是期望值，是 mean，所以不同的分布或许有同样的mean，
两个分布比较选择了期望大的，有可能其实方差也恰巧5很大，那么选择的这个 action 行为就会显得风险很高
意思就是Q值是期望值  但是有时候两个动作的的Q值也就是期望差不太多 但是方差却差的很多 方法越大 不就是不稳定
所以我们更倾向于方差小的 但按照以前的单纯看Q那就不好了  所以我们以这个想法1我们就想找到各个Q值的分布
以此求出方差 这样就可以跟据期望和方差共同去选择动作  但是这个方法我i们只是在训练时候使用 在真正的时候还是选最大的
Rainbow  方法就是将上面这些所有的优化方法都使用了
Continuous Actions 现实中往往不仅仅是离散值 而是连续值 解决方法就是将连续值采样  然后选择使得Q值最大的
还有一个方法就是梯度提升 就是用梯度 不断逼近使得Q值最大的动作 
第三种就是设置一个网络 输入一个状态 返回最优的行为
"""


import gym
from RL_brain import DQNPrioritizedReplay
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

env = gym.make('MountainCar-v0')
env = env.unwrapped
env.seed(21)
MEMORY_SIZE = 10000

sess = tf.Session()
with tf.variable_scope('natural_DQN'):
    RL_natural = DQNPrioritizedReplay(
        n_actions=3, n_features=2, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.00005, sess=sess, prioritized=False,
    )

with tf.variable_scope('DQN_with_prioritized_replay'):
    RL_prio = DQNPrioritizedReplay(
        n_actions=3, n_features=2, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.00005, sess=sess, prioritized=True, output_graph=True,
    )
sess.run(tf.global_variables_initializer())


def train(RL):
    total_steps = 0
    steps = []
    episodes = []
    for i_episode in range(20):
        observation = env.reset()
        while True:
            # env.render()

            action = RL.choose_action(observation)

            observation_, reward, done, info = env.step(action)

            if done: reward = 10

            RL.store_transition(observation, action, reward, observation_)

            if total_steps > MEMORY_SIZE:
                RL.learn()

            if done:
                print('episode ', i_episode, ' finished')
                steps.append(total_steps)
                episodes.append(i_episode)
                break

            observation = observation_
            total_steps += 1
    return np.vstack((episodes, steps))

his_natural = train(RL_natural)
his_prio = train(RL_prio)

# compare based on first success
plt.plot(his_natural[0, :], his_natural[1, :] - his_natural[1, 0], c='b', label='natural DQN')
plt.plot(his_prio[0, :], his_prio[1, :] - his_prio[1, 0], c='r', label='DQN with prioritized replay')
plt.legend(loc='best')
plt.ylabel('total training time')
plt.xlabel('episode')
plt.grid()
plt.show()


