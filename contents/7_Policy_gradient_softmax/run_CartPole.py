"""
Policy Gradient, Reinforcement Learning.

The cart pole example

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import gym
from RL_brain import PolicyGradient
import matplotlib.pyplot as plt

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()



DISPLAY_REWARD_THRESHOLD = 10  #  多少个回合后显示
RENDER = False  # rendering wastes time

env = gym.make('CartPole-v1', render_mode="human")
#env.seed(1)     # reproducible, general Policy gradient has high variance
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)
tf.reset_default_graph()#清空所有计算图防止 下次运行时，变量重复
RL = PolicyGradient(
    n_actions=env.action_space.n,
    n_features=env.observation_space.shape[0],
    learning_rate=0.02,
    reward_decay=0.99,
    # output_graph=True,
)
#这里让计算机跑完一个回合才更斯一次，因为我们要计算总的奖励啊 ，所以要整个轨迹
#之前的QLearn等在四合中每一步部可用定新参数
for i_episode in range(3000):

    observation = env.reset(seed = 1)
    observation = observation[0]

    while True:
        if RENDER: env.render()#显示会使得程序变慢 所以先不显示  练的擦不多在显示

        action = RL.choose_action(observation)

        observation_, reward, done, info ,_= env.step(action)

        RL.store_transition(observation, action, reward)#存储轨迹中去

        if done:
            ep_rs_sum = sum(RL.ep_rs)#奖励加和
            
            #求平均
            if 'running_reward' not in globals():#第一个回合
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
                
                
            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True     # rendering  显示
            print("episode:", i_episode, "  reward:", int(running_reward))

            vt = RL.learn()

            if i_episode == 0:
                plt.plot(vt)    # plot the episode vt
                plt.xlabel('episode steps')
                plt.ylabel('normalized state-action value')
                plt.show()
            break

        observation = observation_
