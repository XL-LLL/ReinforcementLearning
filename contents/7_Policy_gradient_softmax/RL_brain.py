"""
PG的基本思想:直接训练神经网络来输入state 输出 action，这个过程中不去计算 Q
相当于直接学习策略 不是像之前那样还要计算Q值 在选择最大的Q，这里直接改变了策略

如果说 DQN 是一个TD+神经网络的算法,那么PG 是一个蒙地卡罗+神经网络的算法。
利用 reward 奖励直接对选择行为的可能性进行增强和减弱，好的行为会被增加下一次被选中的概率，
不好的行为会被减弱下次被选中的概率。换句话说，如果智能体的动作是对的，
那么就让这个动作获得更多被选择的几率;相反，如果这个动作是错的，
那么这个动作被选择的几率将会减少。

一个回合的所有轨迹 所获得的所有奖励 的平均 就可以衡量动作的好坏
最大化奖励的平均就可以得出最优姐 也就是神经网络的参数 这个参数决定机器人的行为
一个回合中的所有行为构成的轨迹获得的所有奖励的平均可以衡量动作的好坏
之后就又绕回来了 最大化这个奖励 就可以得到网络的参数 这不就更新了网络的参数
这里涉及最大化的奖励  那就是梯度上升 经过一些列推到
原函数就是  将所有奖励求和  导数是

推导到这里，其实最后理解起来是非常容易的，就是说处在t回合中的state 下，
采取我们就希望更新模型使得几率p(&Ist,@越个action，对应的 total reward 是正的话，大越大，
反之对应的 total reward 是负的话我们就希望更新模型使得几率越小越好注意这里一定是
 total reward，如果是某一t时刻的 reward，那么只会让那些得分的动作反复学到
 ，比如开火会得分，那么就只学会了一直开火，而不会学会移动。
                                                        
"""

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

np.random.seed(1)
tf.set_random_seed(1)
tf.set_random_seed(1)

class PolicyGradient:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.95,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        self._build_net()

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # http://0.0.0.0:6006/
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        with tf.name_scope('inputs'):
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations")
            
            self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")
        # fc1
        layer = tf.layers.dense(
            inputs=self.tf_obs,
            units=10,
            activation=tf.nn.tanh,  # tanh activation
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc1'
        )
        # fc2
        all_act = tf.layers.dense(
            inputs=layer,
            units=self.n_actions,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc2'
        )

        self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')  # use softmax to convert to probability
    
        with tf.name_scope('loss'):
            # to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss)
            # neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.tf_acts)   # this is negative log of chosen action
            # or in this way:
            neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
            loss = tf.reduce_mean(neg_log_prob * self.tf_vt)  # reward guided loss

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def choose_action(self, observation):
        prob_weights = self.sess.run(self.all_act_prob, 
                                     feed_dict={self.tf_obs: observation[np.newaxis, :]})
       #给了P概率就把最大概率额抽取出来  没给就按照相同概率抽取
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        # discount and normalize episode reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards()

        # train on episode
        self.sess.run(self.train_op, feed_dict={
             self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
             self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
             self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
        })

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []    # empty episode data
        return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs



