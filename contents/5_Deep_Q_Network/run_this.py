
from maze_env import Maze
from RL_brain import DeepQNetwork
import tensorflow as tf

def run_maze():
    step = 0
    for episode in range(300):
        # 初始化机器人位置  使之回到初始位置1，1
        observation = env.reset()

        while True:
            # 刷新
            env.render()

            #选择动作
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)
            #更新replay buffer
            RL.store_transition(observation, action, reward, observation_)
            #首先要保证有200个数据，才可以开始学习  之后就是没5步骤学习一次
            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
            step += 1

    # end of game
    print('game over')
    env.destroy()


if __name__ == "__main__":
    tf.reset_default_graph()#清空所有计算图防止 下次运行时，变量重复
    # maze game
    env = Maze()
    
    """
    env.n_actions   动作空间  就是有几个动作 上下左右
    env.n_features  网络的输入特征数 比如是图片 那就是图片的像素数 那这里应该是坐标  那就是2 因为是横纵坐标
                     实际上我们获取到的是四个角的坐标 所以那就是4 但是这里把它变成中心点的坐标 所以就是2
    learning_rate=0.01 学习率     
    reward_decay=0.9  衰减率
    e_greedy=0.9        探索率
    replace_target_iter=200, 200次后将q网络参数复制到  target网络
    memory_size=2000,   存储数据的大小，主要作用在当我们要训练数据时候，需要样本  我们使用批量梯度下降
                        所以要存一下数据作为样本 这也有关到网络的形状
    output_graph=True 是否输出计算图
    """
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )
    env.after(100, run_maze)
    env.mainloop()
    RL.plot_cost()