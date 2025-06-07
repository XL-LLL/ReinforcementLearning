"""
This part of code is the DQN brain, which is a brain of the agent.
All decisions are made in here.
Using Tensorflow to build the neural network.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.7.3
"""

import numpy as np
import pandas as pd
import tensorflow as tf


np.random.seed(1)
tf.set_random_seed(1)


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        #n_features是2 这代表着输出是两个特征  至于为什么buffer要2*2+2
        #首先这个横坐标指定是memory_size没什么 纵坐标篇我们要记录[s, a, r, s_]
        #其中[s, s_]这两个代表的是机器人的状态实际上就是坐标，也就是特征数也就是2
        #代表着x与y  那么我们将这四个变量堆叠到一起后就变成了[s(x),s(y),a, r,s_(x),s_(y)]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        self._build_net()
        #tf.get_collection用于根据集合名称获取相关的全部变量引用列表
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        
        #将eval网络的参数复制到target网络的参数中去，这也是需要sess.run的，可不是一直执行的
        #初始化的时候执行一下，保证两个参数一致
        # tf.assign 函数可以对 tensor 进行整体的赋值
        #zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
        # a = [1,2,3] b = [4,5,6] zipped = zip(a,b)     # 打包为元组的列表
        #输出[(1, 4), (2, 5), (3, 6)]
        #range(0, 30, 5)  # 步长为 5
        #输出[0, 5, 10, 15, 20, 25]
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)
        #全局变量初始化
        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        # ------------------ build evaluate_net ------------------
        #placeholder()函数是在神经网络构建graph的时候在模型中的占位，此时并没有把要输入的数据传入模型，它只会分配必要的内存。
        #等建立session，在会话中，运行模型的时候通过feed_dict()函数向占位符喂入数据。
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        #tf.variable_scope('eval_net')指定作用域  这样两个网络的参数不会重复
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            # first layer. collections is used later when assign to target net
           
            #tf.get_variable()作用：用于获取已存在的变量(要求不仅名字，而且初始化方法等各个参数都一样)
            #如果不存在，就新建一个。可以用各种初始化方法，不用明确指定值。

            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l1, w2) + b2

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2

    def store_transition(self, s, a, r, s_):
        #hasattr() 函数用于判断对象是否包含对应的属性。
        #memory_counter记录有多少条数据
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        #np.hstack将参数元组的元素数组按水平方向进行叠加,
      
        transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]
        #为了探索
        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            #q_eval就是正向传播的，这是一个列表 对应的是每一个动作对应的q值
            #输入是机器人当前的状态
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # check to replace target parameters
        #将eval网络的参数复制到target网络的参数中去，这也是需要sess.run的，可不是一直执行的
        #初始化的时候执行一下，保证两个参数一致，每隔一段时间在复制
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        # 随机在memory中抽取batch_size个数据
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        #batch_memory[:, -self.n_features:]  是[:,-2:] 所有行 倒数第二列到最后就是s_ 
        #batch_memory[:, :self.n_features]同理
        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],  # fixed params
                self.s: batch_memory[:, :self.n_features],  # newest params
            })

        # 防止后续修改q_target就是各个状态对应的动作的值
        q_target = q_eval.copy()
        #返回一个有终点和起点的固定步长的排列（可理解为一个等差数组）。 0~batch_size 
        #生成batch_size的序号 因为每一个batch都存起来 要是一个batch有20条数据 
        #动作空间是2 那么网络的输出20*2  这不是也是深度学习时，矩阵的作用 一次训练好多样本
        #输入是样本数*特征数  输出是分类数*样本数  中间网络就是特征数*分类数
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        #获取r和act
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        #训练
        # train eval network
        #训练需要真实值和预测值   预测值可以通过不变q_target网络前向传播 然后经过公式变成预测值
        #真实值可以通过时变q_eval网络正向传播 之后获得真实值 二者都有了就可以梯度下降了
        #训练是为了减小误差 改变参数  误差是真实值与预测值的差 二者都要有样本 是输入是replaybuffer存的实际也就是 s和s_
        #都是一样的输入一样的网络，为啥会有误差呢 因为两个网络并不是每一此都复制的  需要过几次才会将网络参数复制
        #
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

    def display_memory(self):
        print(self.memory)
