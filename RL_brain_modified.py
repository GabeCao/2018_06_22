import numpy as np
import tensorflow as tf
import random
import os
from sklearn.preprocessing import LabelBinarizer
import sys

np.random.seed(1)
tf.set_random_seed(1)
random.seed(1)


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions=1,
            n_features=195,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=1000,
            batch_size=50,
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
        self.memory = np.zeros((self.memory_size, 152 * 2 + 2 + 43 + 1 + 1 + 1))

        # consist of [target_net, evaluate_net]
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

        # ################从这开始都是新添加的属性###############################
        # 保存训练结果
        self.saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=0.3)
        # 对 42个hotspot 进行独热编码
        self.hotspots_num = [str(i) for i in range(1, 43)]
        self.hotspot_num_one_hot_encoded_label_binarizer = LabelBinarizer()
        self.hotspot_num_one_hot_encoded = self.hotspot_num_one_hot_encoded_label_binarizer.fit_transform(self.hotspots_num)
        # 对action 独热编码,用字典表示,key: action(string), value: action的编码结果(ndarray)
        self.action_one_hot_encoded_1 = {}
        self.action_one_hot_encoded_2 = {}
        self.action_one_hot_encoded_3 = {}
        self.action_one_hot_encoded_4 = {}
        self.action_one_hot_encoded_5 = {}
        self.action_one_hot_encoded_6 = {}
        self.action_one_hot_encoded_7 = {}
        self.action_one_hot_encoded_8 = {}
        self.action_one_hot_encoded_9 = {}
        self.action_one_hot_encoded_10 = {}
        self.action_one_hot_encoded_11 = {}
        self.action_one_hot_encoded_12 = {}
        self.action_one_hot_encoded_13 = {}
        self.action_one_hot_encoded_14 = {}
        self.action_one_hot_encoded_15 = {}
        self.action_one_hot_encoded_16 = {}
        self.action_one_hot_encoded_17 = {}
        self.action_one_hot_encoded_18 = {}
        self.action_one_hot_encoded_19 = {}
        self.action_one_hot_encoded_20 = {}
        self.action_one_hot_encoded_21 = {}
        self.action_one_hot_encoded_22 = {}
        self.action_one_hot_encoded_23 = {}
        self.action_one_hot_encoded_24 = {}
        self.action_one_hot_encoded_25 = {}
        self.action_one_hot_encoded_26 = {}
        self.action_one_hot_encoded_27 = {}
        self.action_one_hot_encoded_28 = {}
        self.action_one_hot_encoded_29 = {}
        self.action_one_hot_encoded_30 = {}
        self.action_one_hot_encoded_31 = {}
        self.action_one_hot_encoded_32 = {}
        self.action_one_hot_encoded_33 = {}
        self.action_one_hot_encoded_34 = {}
        self.action_one_hot_encoded_35 = {}
        self.action_one_hot_encoded_36 = {}
        self.action_one_hot_encoded_37 = {}
        self.action_one_hot_encoded_38 = {}
        self.action_one_hot_encoded_39 = {}
        self.action_one_hot_encoded_40 = {}
        self.action_one_hot_encoded_41 = {}
        self.action_one_hot_encoded_42 = {}
        self.set_action_one_hot_encoded()

    def _build_net(self):
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, 106], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, 106], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [106, 106], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, 106], initializer=b_initializer, collections=c_names)
                l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l3'):
                w3 = tf.get_variable('w3', [106, self.n_actions], initializer=w_initializer, collections=c_names)
                b3 = tf.get_variable('b3', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l2, w3) + b3

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
                w1 = tf.get_variable('w1', [self.n_features, 106], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, 106], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [106, 106], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, 106], initializer=b_initializer, collections=c_names)
                l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l3'):
                w3 = tf.get_variable('w3', [106, self.n_actions], initializer=w_initializer, collections=c_names)
                b3 = tf.get_variable('b3', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l2, w3) + b3

    def store_transition(self, s, a, r, done, phase, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        # action_hotspot = a[0]
        # action_staying_time = a[1]
        # transition = np.hstack((s, [action_hotspot, action_staying_time, r], s_))
        transition = np.hstack((s, a, r, done, phase, s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    # 从这里开始都是新添加的函数
    # 传入一个时间段，返回该时间段的对action 独热编码
    def get_current_action_one_hot_encoded(self, hour):
        if hour == 1:
            return self.action_one_hot_encoded_1
        elif hour == 2:
            return self.action_one_hot_encoded_2
        elif hour == 3:
            return self.action_one_hot_encoded_3
        elif hour == 4:
            return self.action_one_hot_encoded_4
        elif hour == 5:
            return self.action_one_hot_encoded_5
        elif hour == 6:
            return self.action_one_hot_encoded_6
        elif hour == 7:
            return self.action_one_hot_encoded_7
        elif hour == 8:
            return self.action_one_hot_encoded_8
        elif hour == 9:
            return self.action_one_hot_encoded_9
        elif hour == 10:
            return self.action_one_hot_encoded_10
        elif hour == 11:
            return self.action_one_hot_encoded_11
        elif hour == 12:
            return self.action_one_hot_encoded_12
        elif hour == 13:
            return self.action_one_hot_encoded_13
        elif hour == 14:
            return self.action_one_hot_encoded_14
        elif hour == 15:
            return self.action_one_hot_encoded_15
        elif hour == 16:
            return self.action_one_hot_encoded_16
        elif hour == 17:
            return self.action_one_hot_encoded_17
        elif hour == 18:
            return self.action_one_hot_encoded_18
        elif hour == 19:
            return self.action_one_hot_encoded_19
        elif hour == 20:
            return self.action_one_hot_encoded_20
        elif hour == 21:
            return self.action_one_hot_encoded_21
        elif hour == 22:
            return self.action_one_hot_encoded_22
        elif hour == 23:
            return self.action_one_hot_encoded_23
        elif hour == 24:
            return self.action_one_hot_encoded_24
        elif hour == 25:
            return self.action_one_hot_encoded_25
        elif hour == 26:
            return self.action_one_hot_encoded_26
        elif hour == 27:
            return self.action_one_hot_encoded_27
        elif hour == 28:
            return self.action_one_hot_encoded_28
        elif hour == 29:
            return self.action_one_hot_encoded_29
        elif hour == 30:
            return self.action_one_hot_encoded_30
        elif hour == 31:
            return self.action_one_hot_encoded_31
        elif hour == 32:
            return self.action_one_hot_encoded_32
        elif hour == 33:
            return self.action_one_hot_encoded_33
        elif hour == 34:
            return self.action_one_hot_encoded_34
        elif hour == 35:
            return self.action_one_hot_encoded_35
        elif hour == 36:
            return self.action_one_hot_encoded_36
        elif hour == 37:
            return self.action_one_hot_encoded_37
        elif hour == 38:
            return self.action_one_hot_encoded_38
        elif hour == 39:
            return self.action_one_hot_encoded_39
        elif hour == 40:
            return self.action_one_hot_encoded_40
        elif hour == 41:
            return self.action_one_hot_encoded_41
        elif hour == 42:
            return self.action_one_hot_encoded_42
    ###########################################################
    # 新添加了一个传入参数，当前环境的 秒数
    def choose_action(self, observation, seconds):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            # actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            # action = np.argmax(actions_value)

            ###########################################################
            # 获得当前的时间段，通过当前时间段获得 对应的action 独热编码
            hour = int(seconds / 1200) + 1
            current_phase = self.get_current_action_one_hot_encoded(hour)
            chose_value = -sys.maxsize-1
            chose_action = None
            chose_action_encoded = None
            # 遍历所有action 找到神经网络返回最大的action_value 的 action
            for key, value in current_phase.items():
                value = value[np.newaxis, :]
                action_observation = np.c_[value, observation]
                action_value = self.sess.run(self.q_eval, feed_dict={self.s: action_observation})
                if action_value[0][0] > chose_value:
                    chose_value = action_value
                    chose_action = key
                    chose_action_encoded = value
            action = chose_action
            # 对action 处理，因为得到的action 形如 '23,4' ，表示在第23个hotspot 待4个t,转换成ndarray [23 4]
            action = list(action.split(','))
            action = list(map(int, action))
            for i in chose_action_encoded[0]:
                action.append(i)
            action = np.array(action)
            ###########################################################

        else:
            # action = np.random.randint(0, self.n_actions)
            ###########################################################
            hour = int(seconds / 1200) + 1
            current_phase = self.get_current_action_one_hot_encoded(hour)
            # 随机从current_phase 选出一个action，代码其他的地方好像用的是伪随机数
            action, action_encoded = random.choice(list(current_phase.items()))
            # 对action 处理，因为得到的action 形如 '23,4' ，表示在第23个hotspot 待4个t,转换成ndarray [23 4]
            action = list(action.split(','))
            action = list(map(int, action))
            for i in action_encoded:
                action.append(i)
            action = np.array(action)
            ###########################################################
        return action

    # 在八点时刻，随机选一个action。这个函数供环境在初始化的时候使用
    def random_action(self):
        action, action_encoded = random.choice(list(self.action_one_hot_encoded_1.items()))
        action = list(action.split(','))
        action = list(map(int, action))
        for i in action_encoded:
            action.append(i)
        action = np.array(action)
        return action

    def action_to_one_hot_encoded(self, path, file, action_one_hot_encoded):
        # 存放每个文件中的所有hotspot
        hotspots = []
        # 存放每个hotspot 对应的最大等待时间
        hotspot_max_staying_time = {}
        with open(path + file) as f:
            for line in f:
                data = line.strip().split(',')
                hotspots.append(data[0])
                hotspot_max_staying_time[data[0]] = data[1]

        for hotspot_num, max_staying_time in hotspot_max_staying_time.items():
            # 根据hotspot_num 找到 对应的热编码后的结果
            hotspot_num_encoded = self.hotspot_num_one_hot_encoded_label_binarizer.transform([hotspot_num])[0]
            max_staying_time = int(max_staying_time)
            for i in range(1, max_staying_time + 1):
                tmp = np.r_[hotspot_num_encoded, i]
                action_one_hot_encoded[hotspot_num + ',' + str(i)] = tmp

    def set_action_one_hot_encoded(self):
        # 保存最大等待时间的文件夹
        path = '20minutes/'
        files = os.listdir(path)
        # 对于每个文件,读取文件的内容, 进行热编码放入相应的字典中。例如8.txt代表8时间段的内容，
        # 放入到action_one_hot_encoded_8字典中
        for file in files:
            if file == '1.txt':
                self.action_to_one_hot_encoded(path, file, self.action_one_hot_encoded_1)
            elif file == '2.txt':
                self.action_to_one_hot_encoded(path, file, self.action_one_hot_encoded_2)
            elif file == '3.txt':
                self.action_to_one_hot_encoded(path, file, self.action_one_hot_encoded_3)
            elif file == '4.txt':
                self.action_to_one_hot_encoded(path, file, self.action_one_hot_encoded_4)
            elif file == '5.txt':
                self.action_to_one_hot_encoded(path, file, self.action_one_hot_encoded_5)
            elif file == '6.txt':
                self.action_to_one_hot_encoded(path, file, self.action_one_hot_encoded_6)
            elif file == '7.txt':
                self.action_to_one_hot_encoded(path, file, self.action_one_hot_encoded_7)
            elif file == '8.txt':
                self.action_to_one_hot_encoded(path, file, self.action_one_hot_encoded_8)
            elif file == '9.txt':
                self.action_to_one_hot_encoded(path, file, self.action_one_hot_encoded_9)
            elif file == '10.txt':
                self.action_to_one_hot_encoded(path, file, self.action_one_hot_encoded_10)
            elif file == '11.txt':
                self.action_to_one_hot_encoded(path, file, self.action_one_hot_encoded_11)
            elif file == '12.txt':
                self.action_to_one_hot_encoded(path, file, self.action_one_hot_encoded_12)
            elif file == '13.txt':
                self.action_to_one_hot_encoded(path, file, self.action_one_hot_encoded_13)
            elif file == '14.txt':
                self.action_to_one_hot_encoded(path, file, self.action_one_hot_encoded_14)
            elif file == '15.txt':
                self.action_to_one_hot_encoded(path, file, self.action_one_hot_encoded_15)
            elif file == '16.txt':
                self.action_to_one_hot_encoded(path, file, self.action_one_hot_encoded_16)
            elif file == '17.txt':
                self.action_to_one_hot_encoded(path, file, self.action_one_hot_encoded_17)
            elif file == '18.txt':
                self.action_to_one_hot_encoded(path, file, self.action_one_hot_encoded_18)
            elif file == '19.txt':
                self.action_to_one_hot_encoded(path, file, self.action_one_hot_encoded_19)
            elif file == '20.txt':
                self.action_to_one_hot_encoded(path, file, self.action_one_hot_encoded_20)
            elif file == '21.txt':
                self.action_to_one_hot_encoded(path, file, self.action_one_hot_encoded_21)
            elif file == '22.txt':
                self.action_to_one_hot_encoded(path, file, self.action_one_hot_encoded_22)
            elif file == '23.txt':
                self.action_to_one_hot_encoded(path, file, self.action_one_hot_encoded_23)
            elif file == '24.txt':
                self.action_to_one_hot_encoded(path, file, self.action_one_hot_encoded_24)
            elif file == '25.txt':
                self.action_to_one_hot_encoded(path, file, self.action_one_hot_encoded_25)
            elif file == '26.txt':
                self.action_to_one_hot_encoded(path, file, self.action_one_hot_encoded_26)
            elif file == '27.txt':
                self.action_to_one_hot_encoded(path, file, self.action_one_hot_encoded_27)
            elif file == '28.txt':
                self.action_to_one_hot_encoded(path, file, self.action_one_hot_encoded_28)
            elif file == '29.txt':
                self.action_to_one_hot_encoded(path, file, self.action_one_hot_encoded_29)
            elif file == '30.txt':
                self.action_to_one_hot_encoded(path, file, self.action_one_hot_encoded_30)
            elif file == '31.txt':
                self.action_to_one_hot_encoded(path, file, self.action_one_hot_encoded_31)
            elif file == '32.txt':
                self.action_to_one_hot_encoded(path, file, self.action_one_hot_encoded_32)
            elif file == '33.txt':
                self.action_to_one_hot_encoded(path, file, self.action_one_hot_encoded_33)
            elif file == '34.txt':
                self.action_to_one_hot_encoded(path, file, self.action_one_hot_encoded_34)
            elif file == '35.txt':
                self.action_to_one_hot_encoded(path, file, self.action_one_hot_encoded_35)
            elif file == '36.txt':
                self.action_to_one_hot_encoded(path, file, self.action_one_hot_encoded_36)
            elif file == '37.txt':
                self.action_to_one_hot_encoded(path, file, self.action_one_hot_encoded_37)
            elif file == '38.txt':
                self.action_to_one_hot_encoded(path, file, self.action_one_hot_encoded_38)
            elif file == '39.txt':
                self.action_to_one_hot_encoded(path, file, self.action_one_hot_encoded_39)
            elif file == '40.txt':
                self.action_to_one_hot_encoded(path, file, self.action_one_hot_encoded_40)
            elif file == '41.txt':
                self.action_to_one_hot_encoded(path, file, self.action_one_hot_encoded_41)
            elif file == '42.txt':
                self.action_to_one_hot_encoded(path, file, self.action_one_hot_encoded_42)

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        # ################################################################
        # 获得batch_memory 的行数和列数
        rows, cols = batch_memory.shape
        for row in range(rows):
            batch_memory_row = batch_memory[row]
            # 获得state_i
            state_i = batch_memory_row[: 152][np.newaxis, :]
            # 获得action_i
            action_i = batch_memory_row[154: 197][np.newaxis, :]
            # 将state_i 和 action_i 拼接在一起，放在q_eval_all_action_state，以后传入eval_net
            q_eval_action_state = np.c_[action_i, state_i]
            if row == 0:
                q_eval_all_action_state = q_eval_action_state
            else:
                q_eval_all_action_state = np.vstack((q_eval_all_action_state, q_eval_action_state))

            reward_i = batch_memory_row[-155]
            done = batch_memory_row[-154]

            # done == 1 表示 state_i+1 is terminal
            if row == 0 and done == 1:
                q_target = np.array([reward_i])[np.newaxis, :]
            elif row == 0 and done == 0:
                choose_value = -sys.maxsize - 1
                # 得到state_i+1 的时间段
                phase = batch_memory_row[-153]
                # 得到state_i+1
                state_i_1 = batch_memory_row[-152:][np.newaxis, :]
                current_phase = self.get_current_action_one_hot_encoded(phase)
                # 遍历当前时间段的所有action，得到返回最大的reward
                for _, action_i_1 in current_phase.items():
                    action_i_1 = action_i_1[np.newaxis, :]
                    action_state = np.c_[action_i_1, state_i_1]
                    state_i_1_reward = self.sess.run(self.q_next, feed_dict={self.s_: action_state})
                    if state_i_1_reward[0][0] > choose_value:
                        choose_value = state_i_1_reward[0][0]

                y_i = reward_i + self.gamma * choose_value
                q_target = np.array([y_i])[np.newaxis, :]
            elif row != 0 and done == 1:
                q_target = np.vstack((q_target, np.array([reward_i])[np.newaxis, :]))
            elif row != 0 and done == 0:
                choose_value = -sys.maxsize - 1
                # 得到state_i+1 的时间段
                phase = batch_memory_row[-153]
                # 得到state_i+1
                state_i_1 = batch_memory_row[-152:][np.newaxis, :]
                current_phase = self.get_current_action_one_hot_encoded(phase)
                # 遍历当前时间段的所有action，得到返回最大的reward
                for _, action_i_1 in current_phase.items():
                    action_i_1 = action_i_1[np.newaxis, :]
                    action_state = np.c_[action_i_1, state_i_1]
                    state_i_1_reward = self.sess.run(self.q_next, feed_dict={self.s_: action_state})
                    if state_i_1_reward[0][0] > choose_value:
                        choose_value = state_i_1_reward[0][0]

                y_i = reward_i + self.gamma * choose_value
                q_target = np.vstack((q_target, np.array([y_i])[np.newaxis, :]))

        # print(q_target.shape)

        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: q_eval_all_action_state,
                                                self.q_target: q_target})
        self.saver.save(self.sess, './checkpoint_dir/MyModel')
        # self.cost_his.append(self.cost)
        # print('self.cost', self.cost)
        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
        print('learning ......', self.learn_step_counter, '    times')

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()