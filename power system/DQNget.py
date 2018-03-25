#-*- coding: UTF-8 -*- 
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
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
from pypower.case118_C import case118
import DCCFS3 as dc
import DQNtest 

def result_sort(basedata,subcasedata,state_list):
	# adjust result branch  keep the same order as before
	adjust_branch = []
	adjust_time = []
	for i in range(0,len(basedata['branch'])):
		if state_list[i]== 0 :
			continue
		for k in range(0,len(subcasedata['branch'])):
			if subcasedata['branch'][k][0] == basedata['branch'][i][0] and subcasedata['branch'][k][1] == basedata['branch'][i][1]:
				adjust_time.append(subcasedata['time'][k])
				adjust_branch.append(subcasedata['branch'][k])
				# branch_index.append(i)
				break

	subcasedata['time'] = adjust_time
	subcasedata['branch'] = np.array(adjust_branch)

def get_staterun(basecase,state_list,attack_num):
	casedata = copy.deepcopy(basecase)
	branch =[]
	for i in range(0,len(basecase['branch'])):
		if state_list[i] == 1:
			branch.append(basecase['branch'][i])

	casedata['branch'] = np.array(branch)
	subbus,subbranch,subtime,subgen = dc.subgraph(casedata)
	# print len(subgen)
	subcasedata_list =[]

	for i in range(0,len(subbus)):
			# print len(subbranch[i])
			subcasedata=copy.deepcopy(casedata)
			subcasedata['bus'] = np.array(subbus[i])
			subcasedata['branch'] = np.array(subbranch[i])
			subcasedata['time'] = subtime[i]
			subcasedata['gen'] = np.array(subgen[i])

			if len(subcasedata['gen']) != 0 and len(subcasedata['branch'])!=0:
				result_sort(basecase,subcasedata,state_list)
				subcasedata_list.append(subcasedata)

	return subcasedata_list

if __name__ == "__main__":
	# basecase =case24_ieee_rts()
	basecase = case9()
	# path = 'D:\\Code\\DC\\DQN\\network\\case24_ieee_rts\\'
	path = 'D:\\Code\\DC\\DQN\\network\\case9\\'

	dc.CFS_int(basecase)
	state_num = len(basecase['branch'])
	q_network = DQNtest.DeepQNetwork(basecase)

	q_network.restore_network(path)

	# q_next = q_network.session.run(q_network.q_eval, feed_dict={self.q_eval_input: batch_next_state})
	blackout_size = 0
	threshold = 5
	trials = 0
	subcasedata = copy.deepcopy(basecase)
	allcase=[]
	allcase.append(subcasedata)
	total_branch_num = dc.get_branch_num(allcase)
	attack_num = -1
	state_list = [1]*state_num

	while True:
		# print(1)
		if blackout_size >= threshold:
			break
		trials +=1
		# print('len(allcase):', len(allcase))
		# print(allcase)
		# print(state_list)
		state_list = [state_list]
		out_result = q_network.session.run(q_network.q_eval, feed_dict={
	                    q_network.q_eval_input: state_list})
		# make sure attack success
		# print(state_list[0])
		while True:
			attack_num = np.argmax(out_result[0])
			print(state_list[0],attack_num)
			if state_list[0][attack_num] != 0:
				# dc.num_attack(attack_num,subcasedata,allcase)
				state_list[0][attack_num] = 0

				subbus,subbranch,subtime,subgen = q_network.convert_statelist_to_staterun(basecase,state_list[0])
				# print(subbus)
				subcasedata = copy.deepcopy(basecase)
				allcase=[]
				for i in range(0,len(subbus)):
					# print( len(subbranch[i])
					subcasedata['bus'] = subbus[i]
					subcasedata['branch'] = subbranch[i]
					subcasedata['time'] = subtime[i]
					subcasedata['gen'] = subgen[i]
					# print(subcasedata)
					next_subcasedata_list = copy.deepcopy(dc.re_dispatch(subcasedata, subbus[i], subbranch[i],subtime[i],subgen[i], 100000))
		            # at here gen use to deside whether to run rundcpf
		            # no load or no gen anymore
					if len(next_subcasedata_list) == 0:
						next_state_list = [0]*state_num
						break

					for next_subcasedata in next_subcasedata_list:
						dc.result_sort(basecase,next_subcasedata)
		                # print( next_subcasedata['branch']
		                # print(next_state)
						dc.CFS(next_subcasedata,allcase)

					# print(allcase)
				break

			else:
				out_result[0][attack_num]= -10000
			print('in else')
			# break
		# allcase store information about after attack
		
		state_list = q_network.convert_staterun_to_statelist(basecase,allcase,state_list[0])
		# print(state_list)
		blackout_size = q_network.get_blackout_size(state_list)
		print("a while ",blackout_size)
		# if trials> 5:
		# 	break
	print(trials)