#-*- coding: UTF-8 -*- 
import tensorflow as tf
import numpy as np
from collections import deque
import random
import networkx as nx
import copy
from pypower.api import rundcpf
from pypower.api import dcpf 
from pypower.case300 import case300
from pypower.case9 import case9
from pypower.case14 import case14
from pypower.case57 import case57
from pypower.case24_ieee_rts import case24_ieee_rts
import DCCFS3 as dc

class DeepQNetwork:

    # 状态数。
    state_num = 9

    # 动作数。        
    action_num = 9

    # 执行步数。
    step_index = 0

    # 训练之前观察多少步。
    OBSERVE = 1000.

    # 选取的小批量训练样本数。
    BATCH = 100

    # epsilon 的最小值，当 epsilon 小于该值时，将不在随机选择行为。
    FINAL_EPSILON = 0.0001

    # epsilon 的初始值，epsilon 逐渐减小。
    INITIAL_EPSILON = 0.1

    # epsilon 衰减的总步数。
    EXPLORE = 3000000.

    # 探索模式计数。
    epsilon = 0

    # 训练步数统计。
    learn_step_counter = 0

    # 学习率。
    learning_rate = 0.001

    # γ经验折损率。
    gamma = 0.9

    # 记忆上限。
    memory_size = 5000

    # 当前记忆数。
    memory_counter = 0

    # 保存观察到的执行过的行动的存储器，即：曾经经历过的记忆。
    replay_memory_store = deque()

    # 生成一个状态矩阵 一行代表一个状态
    state_list = None

    # 生成一个动作矩阵。
    action_list = None

    # q_eval 网络。
    q_eval_input = None
    action_input = None
    q_target = None
    q_eval = None
    predict = None
    loss = None
    train_op = None
    cost_his = None
    reward_action = None

    # tensorflow 会话。
    session = None

    # 保存和调用 tensorflow 神经网络参数 
    saver = None

    def __init__(self, basecase, learning_rate=0.001, gamma=0.9, memory_size=5000):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.memory_size = memory_size
        # 状态数。
        self.state_num = len(basecase['branch'])
        # 动作数。        
        self.action_num = len(basecase['branch'])

        # 初始化状态矩阵。
        # self.state_list = np.identity(self.state_num)

        # 感觉状态矩阵不需要初始化 状态多
        
        # 初始化动作矩阵。
        # self.action_list = np.identity(self.action_num)

        # 创建神经网络。
        self.create_network()

        # 初始化 tensorflow 会话。
        self.session = tf.InteractiveSession()

        # 初始化 tensorflow 参数。
        self.session.run(tf.initialize_all_variables())

        # 初始化 tensorflow 保存调用对象
        self.saver = tf.train.Saver()

        # 记录所有 loss 变化。
        self.cost_his = []

    def create_network(self):
        """
        创建神经网络。
        :return:
        """
        self.q_eval_input = tf.placeholder(shape=[None, self.state_num], dtype=tf.float32)
        self.action_input = tf.placeholder(shape=[None, self.action_num], dtype=tf.float32)
        self.q_target = tf.placeholder(shape=[None], dtype=tf.float32)

        neuro_layer_1 = 3
        w1 = tf.Variable(tf.random_normal([self.state_num, neuro_layer_1]))
        b1 = tf.Variable(tf.zeros([1, neuro_layer_1]) + 0.1)
        l1 = tf.nn.relu(tf.matmul(self.q_eval_input, w1) + b1)

        w2 = tf.Variable(tf.random_normal([neuro_layer_1, self.action_num]))
        b2 = tf.Variable(tf.zeros([1, self.action_num]) + 0.1)
        self.q_eval = tf.matmul(l1, w2) + b2

        # 假设只有一层
        # w1 = tf.Variable(tf.random_normal([self.state_num, self.action_num]))
        # b1 = tf.Variable(tf.zeros([1, self.action_num]) + 0.1)
        # self.q_eval = tf.matmul(self.q_eval_input, w1) + b1

        # 取出当前动作的得分。
        # print(self)
        self.reward_action = tf.reduce_sum(tf.multiply(self.q_eval, self.action_input), reduction_indices=1)
        # self.reward_action = self.q_eval[self.action_input]
        self.loss = tf.reduce_mean(tf.square((self.q_target - self.reward_action)))
        # self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

        self.predict = tf.argmax(self.q_eval, 1)

    def save_network(self, path):
        save_path = self.saver.save(self.session, path)
        return save_path

    def restore_network(self, path):
        load_path = self.saver.restore(self.session,path)
        return load_path
          
    def select_action(self, current_state_list):
        """
        根据策略选择动作。
        :param state_index: 当前状态。
        :return:
        """
        # current_state = self.state_list[state_index:state_index + 1]
        current_action_index = 0
        if np.random.uniform() < self.epsilon:
            # current_action_index = np.random.randint(0, self.action_num)
            while(True):
                choose_to_attack=random.randint(0,len(current_state_list)-1)
                # print choose_to_attack, state_string
                if current_state_list[choose_to_attack]==1:
                    current_action_index = choose_to_attack
                    # print(current_state_list,current_action_index)
                    break

        else:
            # print(current_state_list)
            
            # print(temp_list)
            actions_value = self.session.run(self.q_eval, feed_dict={self.q_eval_input: [current_state_list]})
            # print (current_state_list)
            # print(actions_value)
            # action = np.argmax(actions_value)
            # current_action_index = action
            temp_max_action_value = -10000
            choose_to_attack = []
            for i in range(0,len(current_state_list)):
                if current_state_list[i] == 1:
                    if temp_max_action_value<actions_value[0][i]:
                        choose_to_attack=[]
                        choose_to_attack.append(i)
                        temp_max_action_value = actions_value[0][i]
                    elif temp_max_action_value == actions_value[0][i]:
                        choose_to_attack.append(i)
            current_action_index =  choose_to_attack[random.randint(0,len(choose_to_attack)-1)]
            # print(current_state_list,current_action_index)

        # 开始训练后，在 epsilon 小于一定的值之前，将逐步减小 epsilon。
        if self.step_index > self.OBSERVE and self.epsilon > self.FINAL_EPSILON:
            self.epsilon -= (self.INITIAL_EPSILON - self.FINAL_EPSILON) / self.EXPLORE
        
        return current_action_index

    def save_store(self, current_state, current_action, current_reward, next_state, done):
        """
        保存记忆。
        :param current_state_index: 当前状态 index。
        :param current_action_index: 动作 index。
        :param current_reward: 奖励。
        :param next_state_index: 下一个状态 index。
        :param done: 是否结束。
        :return:
        """
        # print(current_state)
        # print(current_action)  
        # 记忆动作(当前状态， 当前执行的动作， 当前动作的得分，下一个状态)。
        self.replay_memory_store.append((
            [current_state],
            [current_action],
            current_reward,
            [next_state],
            done))

        # 如果超过记忆的容量，则将最久远的记忆移除。
        if len(self.replay_memory_store) > self.memory_size:
            self.replay_memory_store.popleft()

        self.memory_counter += 1

    def convert_statelist_to_staterun(self, basecase,state_list):

        # get four attributes
        branch = basecase['branch']
        time = basecase['time']
        bus = basecase['bus']
        gen = basecase['gen']
        subbus = []
        subbranch = []
        subtime = []
        subgen = []

        G = nx.Graph()
        for i in range(0,len(state_list)):
            if state_list[i] == 1:
                G.add_edge(branch[i][0],branch[i][1])

        set_subbus = list(nx.connected_components(G))

        for i in range(0,len(set_subbus)):
            temp_subbus = []
            temp_subbranch = []
            temp_subtime = []
            temp_gen = []
            for s in set_subbus[i]:
                for b in bus:
                    if s == b[0]:
                        temp_subbus.append(b) 
                        break
                for g in gen:
                    if g[0] == s:
                        temp_gen.append(g)
                        break       
            for k in range(0, len(branch)):
                if (branch[k][0] in set_subbus[i] and branch[k][1] in set_subbus[i]):
                    temp_subbranch.append(branch[k])
                    temp_subtime.append(time[k])
                
            if len(temp_gen) == 0:
                continue
            subbus.append(temp_subbus)
            subbranch.append(temp_subbranch)
            subtime.append(temp_subtime)
            subgen.append(temp_gen)

        return subbus,subbranch,subtime,subgen

    def convert_staterun_to_statelist(self, basecase,allcase):
        state_list=[]
        branch=[]
        for a in allcase:
            for b in a['branch']:
                branch.append(b)
        for b in basecase['branch']:
            if b in branch:
                state_list.append(1)
            else:
                state_list.append(0)
        return state_list

#------------------------------------------------------------------
# all the same in DCFS3.py    
    # def CFS():
    #     pass

    # def re_dispatch():
    #     pass
    # def result_sort():
    #     pass
#------------------------------------------------------------------
    def get_blackout_size(self, state_list):
        blackout_size = 0
        for i in state_list:
            if i == 0:
                blackout_size+=1

        return blackout_size

    def get_reward(self,blackout_size, threshold, attack_num):
        r=0 
        if blackout_size>=threshold and attack_num<threshold:
            r=1
        if blackout_size>=threshold and attack_num>=threshold:
            r=-1
        
        # if r==1:
            # print r
        return r

    def step(self, basecase, state_list, action, threshold, attack_num ):
        """
        执行动作。
        :param state: 当前状态。
        :param action: 执行的动作。
        :return:
        """
        # action means attack
        state_list[action] = 0

        subbus,subbranch,subtime,subgen = self.convert_statelist_to_staterun(basecase,state_list)
        subcasedata = copy.deepcopy(basecase)
        allcase=[]
        for i in range(0,len(subbus)):
            # print( len(subbranch[i])
            subcasedata['bus'] = subbus[i]
            subcasedata['branch'] = subbranch[i]
            subcasedata['time'] = subtime[i]
            subcasedata['gen'] = subgen[i]
            # print(subcasedata)
            next_subcasedata = copy.deepcopy(dc.re_dispatch(subcasedata, subbus[i], subbranch[i],subtime[i],subgen[i], 1000))
            # at here gen use to deside whether to run rundcpf
            # no load or no gen anymore
            if len(next_subcasedata['gen']) == 0:
                # print (next_subcasedata)
                next_state_list = current_state_list = [0]*self.action_num
                reward = self.get_reward(self.state_num, threshold, attack_num)
                done = True
                return next_state_list,reward, done
            # print( subbus[i],subbranch[i],subtime[i],subgen[i]
            # print( casedata['branch']
            dc.result_sort(basecase,next_subcasedata)
            # print( next_subcasedata['branch']
            # print(next_state)
            dc.CFS(next_subcasedata,allcase)

        next_state_list = self.convert_staterun_to_statelist(basecase,allcase)
        blackout_size = self.get_blackout_size(next_state_list)
        reward = self.get_reward(blackout_size,threshold,attack_num)
        
        
        # 到达阈值终止
        done =False
        if blackout_size>=threshold:
            done = True
        
        return next_state_list, reward, done

    def experience_replay(self):
        """
        记忆回放。
        :return:
        """
        # 随机选择一小批记忆样本。
        batch = self.BATCH if self.memory_counter > self.BATCH else self.memory_counter
        minibatch = random.sample(self.replay_memory_store, batch)

        batch_state = None
        batch_action = None
        batch_reward = None
        batch_next_state = None
        batch_done = None

        for index in range(len(minibatch)):
            if batch_state is None:
                batch_state = minibatch[index][0]
            elif batch_state is not None:
                batch_state = np.vstack((batch_state, minibatch[index][0]))

            if batch_action is None:
                batch_action = minibatch[index][1]
            elif batch_action is not None:
                batch_action = np.vstack((batch_action, minibatch[index][1]))

            if batch_reward is None:
                batch_reward = minibatch[index][2]
            elif batch_reward is not None:
                batch_reward = np.vstack((batch_reward, minibatch[index][2]))

            if batch_next_state is None:
                batch_next_state = minibatch[index][3]
            elif batch_next_state is not None:
                batch_next_state = np.vstack((batch_next_state, minibatch[index][3]))

            if batch_done is None:
                batch_done = minibatch[index][4]
            elif batch_done is not None:
                batch_done = np.vstack((batch_done, minibatch[index][4]))

        # q_next：下一个状态的 Q 值。
        q_next = self.session.run([self.q_eval], feed_dict={self.q_eval_input: batch_next_state})

        q_target = []
        for i in range(len(minibatch)):
            # 当前即时得分。
            current_reward = batch_reward[i][0]

            # # 游戏是否结束。
            # current_done = batch_done[i][0]

            # 更新 Q 值。
            q_value = current_reward + self.gamma * np.max(q_next[0][i])

            # 当得分小于 0 时，表示走了不可走的位置。
            # if current_reward < 0:
            #     q_target.append(current_reward)
            # else:
            #     q_target.append(q_value)
            q_target.append(q_value)
        # print(q_target)
        _, cost, reward = self.session.run([self.train_op, self.loss, self.reward_action],
                                           feed_dict={self.q_eval_input: batch_state,
                                                      self.action_input: batch_action,
                                                      self.q_target: q_target})

        self.cost_his.append(cost)

        if self.step_index % 1000 == 0:
            print("loss:", cost)

        self.learn_step_counter += 1

    def train(self,basecase):
        """
        训练。
        :return:
        """

        threshold = 4
        # k = 2

        # 初始化当前状态。
        # current_state = np.random.randint(0, self.action_num - 1)
        # current_state_list = list(np.random.randint(2,size=(self.action_num,)))
        current_state_list = [1]*self.action_num
        attack_num = 0
        # print(current_state_list)
        # for i in range(0,len(basecase['branch'])):
        #     current_state_list.append(1)

        self.epsilon = self.INITIAL_EPSILON

        while True:
            # 选择动作。
            # print("just testing if it dies")
            action = self.select_action(current_state_list)
            attack_num +=1
            # print(action)
            # 执行动作，得到：下一个状态，执行动作的得分，是否结束。
            
            next_state_list, reward, done = self.step(basecase, current_state_list, action, threshold, attack_num)

            # 保存记忆。
            action_list = [0]*self.action_num
            action_list[action] = 1
            self.save_store(current_state_list, action_list, reward, next_state_list, done)

            # 先观察一段时间累积足够的记忆再进行训练。
            if self.step_index > self.OBSERVE:
                self.experience_replay()

            if self.step_index > 10000:
                break

            if done:
                # current_state_list = list(np.random.randint(2,size=(self.action_num,)))
                current_state_list = [1]*self.action_num
                attack_num =0 
            else:
                current_state_list = next_state_list

            self.step_index += 1
            # if self.step_index %1000 ==0:
            #     print(self.step_index)
                # print(self.session.run(self.w1))
            # if self.step_index >3700 :
            #     print (self.step_index)
        path = 'D:\\Code\\DC\\DQN\\network\\case9-1\\'
        save_path = self.save_network(path)
        print(save_path)

if __name__ == "__main__":
    basecase = case9()
    dc.CFS_int(basecase)

    q_network = DeepQNetwork(basecase)
    # print(len(basecase['branch']))
    # print(q_network.q_target)
    q_network.train(basecase)
    # print(q_network.q_target)