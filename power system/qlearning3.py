#-*- coding: UTF-8 -*- 
# import pypower as pw
import numpy as np
import networkx as nx
import random
import copy
from pypower.api import rundcpf
from pypower.api import dcpf 
from pypower.case300 import case300
from pypower.case9 import case9
from pypower.case14 import case14
from pypower.case57 import case57
from pypower.case24_ieee_rts import case24_ieee_rts
import DCCFS3
import matplotlib.pyplot as plt

def get_state(basedata, nowdata):
	s=''
	nowdata_index = 0
	for i in range(0,len(basedata['branch'])):
		if nowdata_index == len(nowdata['branch']):
			s+='0'
			continue
		if basedata['branch'][i][0] == nowdata['branch'][nowdata_index][0] and basedata['branch'][i][1] == nowdata['branch'][nowdata_index][1]:
			s+='1'
			nowdata_index+=1		
		else:
			s+='0'
	return s

def get_Q_value(Q,state_string,choose_to_attack):
	if state_string in Q:
		if choose_to_attack in Q[state_string]:
			# return Q[state_string][choose_to_attack]
			pass
		else:
			Q[state_string][choose_to_attack]=1
			# return Q[state_string][choose_to_attack]
	else:
		Q[state_string]={}
		Q[state_string][choose_to_attack]=1
		# return Q[state_string][choose_to_attack]

def result_sort(basedata,subcasedata,state_string):
	# adjust result branch  keep the same order as before
	# branch_index = []
	# print("in resort len(subcasedata['branch']:",len(subcasedata['branch']),choose_to_attack)
	# print("in resort subtring:",get_state(basedata,subcasedata),state_string)

	adjust_branch = []
	adjust_time = []
	for i in range(0,len(basedata['branch'])):
		if state_string[i]=='0':
			continue
		for k in range(0,len(subcasedata['branch'])):
			if subcasedata['branch'][k][0] == basedata['branch'][i][0] and subcasedata['branch'][k][1] == basedata['branch'][i][1]:
				adjust_time.append(subcasedata['time'][k])
				adjust_branch.append(subcasedata['branch'][k])
				# branch_index.append(i)
				break

	subcasedata['time'] = adjust_time
	subcasedata['branch'] = np.array(adjust_branch)

def state_to_graph(basedata,state_string):
	# print state_string
	casedata = copy.deepcopy(basedata)
	branch =[]
	for i in range(0,len(basedata['branch'])):
		if state_string[i]=='1':
			branch.append(basedata['branch'][i])

	# print ("len(branch):",len(branch))
	casedata['branch'] = np.array(branch)
	subbus,subbranch,subtime,subgen = DCCFS3.subgraph(casedata)
	# print len(subgen)
	subcasedata_list =[]

	for i in range(0,len(subbus)):
			# print len(subbranch[i])
			subcasedata=copy.deepcopy(casedata)
			subcasedata['bus'] = np.array(subbus[i])
			subcasedata['branch'] = np.array(subbranch[i])
			subcasedata['time'] = subtime[i]
			subcasedata['gen'] = np.array(subgen[i])

			# print("len(subcasedata['branch']) before sort:" ,len(subcasedata['branch']))
			# print("st at_state_string:",get_state(basedata,subcasedata))
			# next_subcasedata = copy.deepcopy(re_dispatch(subcasedata, subbus[i], subbranch[i],subtime[i],subgen[i], timeslot))
			# at here gen use to deside whether to run rundcpf
			# no load or no gen anymore
			if len(subcasedata['gen']) != 0 and len(subcasedata['branch'])!=0:
				result_sort(basedata,subcasedata,state_string)
				# print("len(subcasedata['branch']) after sort:" ,len(subcasedata['branch']))
				# print('get after sort string:', get_state(basedata,subcasedata))
				subcasedata_list.append(subcasedata)

	return subcasedata_list

def num_to_bin(num,size):
	s=bin(num)
	r=''
	for i in range(0,size-len(s)+2):
		r+='0'
	for i in range(0,len(s)):
		if i>1:
			r+=s[i]
	return r

def ql_init(basedata):
	action_size = len(basedata['branch'])
	# print action_size
	Q = {}
	per_state={}
	for k in range(0,action_size):
		per_state[k]=0

	state_size=2**action_size
	for i in range(0,state_size):
		p_state=copy.deepcopy(per_state)
		Q[num_to_bin(i,action_size)]=p_state
	# print bin(state_size)
	return Q

def get_blackout_size(next_string):
	blackout_size = 0
	for n in next_string:
		if n=='0':
			blackout_size+=1
	return blackout_size
	# return len(basedata['branch'])-len(state['branch'])

def get_reward(blackout_size, threshold, k):
	r=0 
	if blackout_size>=threshold and k<threshold:
		r=1
	if blackout_size>=threshold and k>=threshold:
		r=-1
	# if r==1:
		# print r
	return r

def choose_attack(Q, state_string,pro):
	r=random.random()
	if r<pro:
		while(True):
			choose_to_attack=random.randint(0,len(state_string)-1)
			# print choose_to_attack, state_string
			if state_string[choose_to_attack]=='1':
				return choose_to_attack
	else:
		choose_to_attack,state_q_max = get_max_choice(Q,state_string)
		return choose_to_attack

def get_max_choice(Q, state_string):
	allzero = True
	choose_to_attack = []
	state_q_max = -100
	for i in range(0,len(state_string)):
		# print state_q_max,Q[state_string][i]
		if state_string[i]=='1':
			allzero = False 
			get_Q_value(Q,state_string,i)
			
			if state_q_max == Q[state_string][i]:
				choose_to_attack.append(i)
			if state_q_max < Q[state_string][i]:
				state_q_max=Q[state_string][i]
				choose_to_attack=[]
				choose_to_attack.append(i)
			

	# print state_string,choose_to_attack
	if allzero:
		return -1,0
	
	# print choose_to_attack
	else:
		return choose_to_attack[random.randint(0,len(choose_to_attack)-1)],state_q_max

def replace_string(str,index,c):
	re = ''
	for i in range(0,index):
		re+=str[i]
	re +=c
	for i in range(index+1,len(str)):
		re+=str[i]
	return re

def ql(basedata, threshold, trials):
	#basedata means donot change, threshold is blacksize's threshold that cannot recover
	# trials mean total trials, k is the number of attack sequence
	Q_value={}
	store_total_num = []
	alf=0.1
	gam=0.9 
	total_attack_num = 0
	pro=0.3
	pro_down=0.005
	sub_pro=0.005
	for cur_trials in range(0,trials):
		# initialize
		blackout_size=0 
		state = copy.deepcopy(basedata)
		state_string = get_state(basedata, state)
		if cur_trials ==0:
			for i in range(0,len(state_string)):
				get_Q_value(Q_value,state_string,i)
				
		while(True):
			# print("----------------------------------")
			# print ("\nblackout_size:",blackout_size)
			if blackout_size>=threshold:
				# print('--------------------------------------','\n'
				print ("\nblackout_size:",blackout_size)
				print('total_attack_num:',total_attack_num)
				
				store_total_num.append(total_attack_num)
				total_attack_num = 0
				break
			
			total_attack_num +=1
			#1 is state['branch']
			#2 is below
			
			choose_to_attack = choose_attack(Q_value,state_string,pro)
	
			at_state_string=replace_string(state_string,choose_to_attack,'0')
			# print("state_string:    " ,state_string , choose_to_attack)
			# print('at_state_string: ', at_state_string)
			#3
			#here run cfs
			subcasedata_list=state_to_graph(basedata, at_state_string)
			# print subcasedata_list
			# print len(subcasedata_list)
			next_state_list = []
			for s in subcasedata_list:
				# print("len(s['branch']):",len(s['branch']))
				# print('get before run string:', get_state(basedata, s))
				temp_next_state_list = DCCFS3.re_dispatch(s,s['bus'],s['branch'],s['time'],s['gen'],100000)
				for t in temp_next_state_list:
					DCCFS3.CFS(t,next_state_list)
			#then get each sub string
			# print next_state_list
			# print len(next_state_list)
			next_string_list=[]
			# print('len(next_state_list):', len(next_state_list))
			for n in next_state_list:
				next_string_list.append(get_state(basedata, n))
			#combine all sub string
			# print next_string_list

			#没有state了 把branch设为0，证明已经blackout
			if len(next_state_list)==0:
				temp_state = copy.deepcopy(basedata)
				temp_branch = []
				temp_state['branch']= np.array(temp_branch)
				next_string_list.append(get_state(basedata, temp_state))

			# 现存的所有branch合起来
			next_string=next_string_list[0]
			for i in range(1,len(next_string_list)):
				for j in range(0,len(next_string)):
					if next_string_list[i][j]=='1':
						# next_string[j]='1' 
						next_string=replace_string(next_string,j,'1')
			# print('next_string',next_string)
			# for n in next_string_list:
				# print('next_string_list:', n)
			#4
			#get reward from state n+1
			blackout_size=get_blackout_size(next_string)
			reward = get_reward(blackout_size,threshold,total_attack_num)
			
			# r[state_string][choose_to_attack] = reward
			#5
			# update Q

			next_choose_to_attack,next_state_q_max = get_max_choice(Q_value,next_string)

			# Q[state_string][choose_to_attack]=(1-alf)*Q[state_string][choose_to_attack]+alf*(r[state_string][choose_to_attack]+gam*next_state_q_max)
			Q_value[state_string][choose_to_attack]=(1-alf)*Q_value[state_string][choose_to_attack]+alf*(reward+gam*next_state_q_max)
			# if reward == -1:
			# 	print(Q_value[state_string][choose_to_attack])

			state_string = next_string

			pro-=sub_pro
			if pro<pro_down:
				pro=pro_down
			# print('pro:',pro)
	# plt.plot(store_total_num)
	# plt.show()
	return Q_value

if __name__ == '__main__':

	basedata=case24_ieee_rts()
	DCCFS3.CFS_int(basedata)
	threshold = 8
	
	experiment = 1
	for i in range(0,experiment):
		trials = 200
		Q_value=ql(basedata,threshold,trials)
		max_state_string = None
		max_Q = -1
		max_in_state_action = None
		for state_string in Q_value:
			for choose_to_attack in Q_value[state_string]:
				if Q_value[state_string][choose_to_attack]!=1 and max_Q < Q_value[state_string][choose_to_attack]:
					# print (state_string,choose_to_attack,Q[state_string][choose_to_attack])
					max_Q = Q_value[state_string][choose_to_attack]
					max_state_string = state_string
					max_in_state_action = choose_to_attack
					# pass
		print(max_Q,max_state_string,max_in_state_action)
	# 1.8912720364319122 11111111111111111111111111111111111111 6
	# 1.8929303495098488 11111111111111111111111111111111111111 26