#-*- coding: UTF-8 -*- 

# DeepQNetwork说明


def create_network(self):
	# 暂时用了一层的线性网络
	# 基本保留输入的格式之类的
	# state_list =[]

	# action = [] 不是单个边的index
	# 因为方便计算 # 取出当前动作的得分
	pass

      
def select_action(self, current_state_list):
	# if 
	# 随机选择一个action
	# else
	# 选择Q最大的一个action 多个的话再随机从里边选取
	# if
	# 更新上述的判定概率
	pass
    
def save_store(self, current_state, current_action, current_reward, next_state, done):
	# 记录进行过的训练状态
	# state 和 action 要保存为[]
	# current_reward 保存为某个值    
    pass

def convert_statelist_to_staterun(self, basecase,state_list):
	# 把state_list转成可以跑的形式 branch bus 。。。。

	# 可能会有多个子图
    pass

def convert_staterun_to_statelist(self, basecase,allcase):
	# 把跑出来的结果重新转回state_list  []形式 
	# 只看branch    
	pass

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
	# 根据state_list得到blackout_size
	pass

def get_reward(self,blackout_size, threshold, attack_num):
	# 判定的规则感觉和论文不大一样
	# 不断更新attack_num
	# attack_num较小的时候可以达到目标
	pass

def step(self, basecase, state_list, action, threshold, attack_num ):
 #    执行一次action
	# 先把state状态改成执行了action后的

	# 然后转成可以跑的state

	# 分析有多少个联通子图 
	# 每个联通子图独立跑

	# 因为去掉了某些边所以要重新分配 gen 和 load

	# 把所有子图的运行结果合到一个allcase里边
	pass 

def experience_replay(self):
	# 记忆回放 也就是选取训练集数据 用于训练神经网络 
	# 根据要求选取相应的格式和数据

	# q_target 同理按照要求求出
	pass

def train(self,basecase):

	# 初始化状态以及相关参数

	# while True：
	# 执行动作 

	# 更新保存相关action后的结果 先跑一定的次数 类似冷启动 
	# 然后再抽数据出来训练

	# 该状态到达尽头 
	# 1. 没有gen了
	# 2. 级联失效bus个数大于阈值

	# 然后就重新开始状态继续跑

	# 否则进入下一个状态跑
	pass



# 最终结果 神经网络里面的参数应该保存下来