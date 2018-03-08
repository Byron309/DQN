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

def result_sort(basedata,subcasedata):
	# adjust result branch  keep the same order as before
	# branch_index = []
	adjust_branch = []
	adjust_time = []
	for i in range(0,len(basedata['branch'])):
		# print( result_branch[k][1]
		for k in range(0,len(subcasedata['branch'])):
			# print( casedata['branch'][i]
			if subcasedata['branch'][k][0] == basedata['branch'][i][0] and subcasedata['branch'][k][1] == basedata['branch'][i][1]:
				adjust_time.append(subcasedata['time'][k])
				adjust_branch.append(subcasedata['branch'][k])
				# branch_index.append(i)
				break

	subcasedata['time'] = adjust_time
	subcasedata['branch'] = np.array(adjust_branch)

def CFS_int(casedata):
	# print( casedata
	casedata['time'] = branch_time(casedata)
	if 'areas' in casedata:
		del casedata['areas']
	if 'gencost' in casedata:
		del casedata['gencost']
	# result = rundcpf(casedata)

def branch_time(casedata):    # to get time proportity 
	time = [] # t[0] is accalulate  t[1] is slot add    t[2] is limit   t[3] is state: 0 is out  1 is not
	for b in casedata['branch']:
		# time_temp = []
		# time_temp.append(0)
		# time_temp.append(b[5])
		# time_temp.append(5 * 0.5 * b[5])
		# time_temp.append(1)
 	# 	time.append(time_temp)
 		time.append(0)
 	# casedata['time'] = time	
	return time

def decide_out(casedata):  # 5s out of limit will dowm  then time-solt is 1s
	totalslot = 0
	out_num = 0

	while(True):
		cal_num = 0	
		for i in range(0,len(casedata['branch'])):
			if(casedata['branch'][i][5] < abs(casedata['branch'][i][13])):  # this branch is out-of-branch
				casedata['time'][i] += (abs(casedata['branch'][i][13]) - abs(casedata['branch'][i][5]) )
				if casedata['time'][i] >= 5*0.5* casedata['branch'][i][5]:
					out_num +=1
				cal_num -=1
			cal_num += 1
		if cal_num == len(casedata['branch']):
			return -1, 0
		if out_num > 0:
			break
		totalslot += 1

	slot = 2 
	index = 0
	key_index = index
	for i in range(0,len(casedata['branch'])):#find min solt and key_index
		if(casedata['branch'][i][5] < abs(casedata['branch'][i][13])):  # this branch is out-of-branch
			casedata['time'][i] -= (abs(casedata['branch'][i][13]) - abs(casedata['branch'][i][5]) )
			if slot > (5*0.5* casedata['branch'][i][5] - casedata['time'][i] ) / (abs(casedata['branch'][i][13]) - abs(casedata['branch'][i][5]) ):
				slot = (5*0.5* casedata['branch'][i][5] - casedata['time'][i] ) / (abs(casedata['branch'][i][13]) - abs(casedata['branch'][i][5]) )
				key_index = i

	# print( slot,index,key_index
	# for t in casedata['time']:  # t[0] is accalulate  t[1] is slot add    t[2] is limit 
		# t[0] += t[1] * slot

	for i in range(0,len(casedata['branch'])):
		if(casedata['branch'][i][5] < abs(casedata['branch'][i][13])):
			casedata['time'][i] += slot* (abs(casedata['branch'][i][13]) - abs(casedata['branch'][i][5]) )
			# print( casedata['branch'][i]
	totalslot +=slot
	return key_index , totalslot

def branch_attack(casedata, b_start, b_end):
	for i in range(0,len(casedata['branch'])):
		if casedata['branch'][i][0] == b_start and casedata['branch'][i][1] == b_end:
			# delete or set 0
			casedata['branch']= np.delete(casedata['branch'], i, 0)
			casedata['time']= np.delete(casedata['time'], i, 0)
			break
	return 

def random_attack(casedata,allcase):

	r= random.randint(0,len(casedata['branch'])-1)
	# r = 7
	# print( r,len(casedata['branch'])
	casedata['branch'] = np.array(np.delete(casedata['branch'],r,0))
	casedata['time'] = np.delete(casedata['time'],r,0)
	# return casedata

	subbus,subbranch,subtime,subgen = subgraph(casedata)


	for i in range(0,len(subbus)):
		# print( len(subbranch[i])
		subcasedata = copy.deepcopy(casedata)
		subcasedata['bus'] = np.array(subbus[i])
		subcasedata['branch'] = np.array(subbranch[i])
		subcasedata['time'] = np.array(subtime[i])
		subcasedata['gen'] = np.array(subgen[i])
		if len(subgen[i]) != 0:
			result_sort(casedata,subcasedata)
			allcase.append(subcasedata)
			# print( subcasedata['gen']
	# for a in allcase:
	# 	print( "a:",a['gen']

	# do not process if subbus is null

def judge_failure(branch):
	#count failure number or something
	pass

def show(casedata):
	print( 'bus: ',len(casedata['bus']))
	for b in casedata['branch']:
		print( b[0],'load: ', b[2])

	print( 'gen: ',len(casedata['gen']))
	for g in casedata['gen']:
		print( g[0], 'generator: ', b[1])

	print( 'branch: ',len(casedata['branch']) )
	for b in casedata['branch']:
		print( b[0],b[1])

def CFS(casedata,allcase):

	result = rundcpf(casedata)

	subcasedata = copy.deepcopy(casedata)
	subcasedata['bus'] = result[0]['bus']
	subcasedata['branch'] = result[0]['branch']
	subcasedata['gen'] = result[0]['gen']
	# print( casedata['branch']
	# print( subcasedata['branch']
	result_sort(casedata,subcasedata)
	# print( len(subcasedata['branch'])
	# print( subcasedata['branch']

	del_index, timeslot = decide_out(subcasedata)     # timeslot use to caculate adjust of generator
	
	# print( del_index,timeslot
	if del_index != -1:  # means exisit branch to delete
		subcasedata['branch']= np.delete(subcasedata['branch'], del_index, 0)
		subcasedata['time']= np.delete(subcasedata['time'], del_index, 0)
		# show(casedata)
		# print( len(casedata['time']),len(casedata['branch'])
		# print( casedata['branch']
		subbus,subbranch,subtime,subgen = subgraph(subcasedata)  # whether generate subgraph 
		# print( len(subbus)
		# if len(subbus)>1: # generate subgraph already
		for i in range(0,len(subbus)):
			# print( len(subbranch[i])
			subcasedata['bus'] = subbus[i]
			subcasedata['branch'] = subbranch[i]
			subcasedata['time'] = subtime[i]
			subcasedata['gen'] = subgen[i]
			next_subcasedata = copy.deepcopy(re_dispatch(subcasedata, subbus[i], subbranch[i],subtime[i],subgen[i], timeslot))
			# at here gen use to deside whether to run rundcpf
			# no load or no gen anymore
			if len(next_subcasedata['gen']) == 0:
				return
			# print( subbus[i],subbranch[i],subtime[i],subgen[i]
			# print( casedata['branch']
			result_sort(casedata,next_subcasedata)
			# print( next_subcasedata['branch']
			CFS(next_subcasedata,allcase)
	else:
		allcase.append(subcasedata)

def subgraph(casedata):
	branch = casedata['branch']
	# print((len(branch))
	time = casedata['time']
	bus = casedata['bus']
	gen = casedata['gen']
	subbus = []
	subbranch = []
	subtime = []
	subgen = []

	g=nx.Graph()
	for b in branch:
		g.add_edge(b[0],b[1])

	set_subbus = list(nx.connected_components(g))
	# print( len(set_subbus)
	# print((set_subbus)
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

	# print( len(subbranch)
	return subbus,subbranch,subtime,subgen

def re_dispatch(casedata, bus, branch, time, gen, timeslot):
	# calculate power of generator and load first
	# adjust input and out put that is gen and bus  
	# 
	adjust_rate = 0.3  # per slot
	adjust_rate*= timeslot
	subcasedata = copy.deepcopy(casedata)
	if len(gen) == 0:
		t=[]
		subcasedata['gen']=np.array(t)  # use gen to control no need condition
		return subcasedata 

	# find power of L
	# l_bus = []
	l_sum = 0
	for b in bus:
		l_sum += b[2]

	# if no load   \\  make sure exist not zero load
	if l_sum == 0:
		t=[]
		subcasedata['gen']=np.array(t)  # use gen to control no need condition
		return subcasedata
	#find power of G and L bus 
	g_sum = 0
	g_up_limit = []
	g_down_limit = []
	g_up_sum = 0
	g_dowm_sum = 0

	#get the generator propority
	for g in gen:  #g[0] is number g[1] is output power
		g_sum += g[1]   
		if g[1]*adjust_rate < g[8]:
			g_up_limit.append(g[1]*adjust_rate)
			g_up_sum += g[1]*adjust_rate
		else:
			g_up_limit.append(g[8])
			g_up_sum += g[8]

		if g[1]*(1-adjust_rate) > g[9]:
			g_down_limit.append(g[1]*(1-adjust_rate))
			g_dowm_sum += g[1]*(1-adjust_rate)
		else:
			g_down_limit.append(g[9])
			g_dowm_sum += g[9]

	# print( bus
	# print( casedata['gen']

	# redispatch G and L
	# first adjust G 
	while True :  # here make sure no any need for out of generator  
		isin = False
		if g_sum > l_sum and g_dowm_sum > l_sum:  # means at high to low but not enough 
			isin = True
			index = 0    # the delete the min g first  
			g_min = 1000
			for i in range(0,len(gen)):
				if gen[i][1] < g_min:
					index = i
					g_min = gen[i][1]

			g_delete_num = gen[index][0]
			# delete the smallest generator
			g_sum -= gen[index][1]
			gen = np.delete(gen, index, 0)
			g_dowm_sum -= g_down_limit[index]
			g_down_limit = np.delete(g_down_limit, index, 0)
			g_up_sum -= g_up_limit[index]
			g_up_limit = np.delete(g_up_limit, index, 0)

			# if delete generator  also have to delete branch 
			for i in range(0,len(bus)):
				if bus[i][0] == g_delete_num:
					bus = np.delete(bus,i,0)
					break 

			branch_delete_list =[]
			for i in range(0,len(branch)):
				if branch[i][0] ==g_delete_num or branch[i][1]==g_delete_num:
					branch_delete_list.append(i)

			branch = np.delete(branch, branch_delete_list, 0)
			time = np.delete(time, branch_delete_list, 0)
			# print( len(branch_delete_list)
		if isin == False:
			break

	# make sure there are some generator
	if g_sum == 0:  
		t=[]
		subcasedata['gen']=np.array(t)  # use gen to control no need condition
		return subcasedata

	# subdivide into different condition
	if g_sum > l_sum:	
		rate = l_sum / g_sum
		g_sum = 0
		for i in range(len(gen)):
			if gen[i][1] * rate < g_down_limit[i]:
				gen[i][1] = g_down_limit[i]
			else:
				gen[i][1] = gen[i][1]*rate
			g_sum+=gen[i][1]
	else:
		rate = l_sum / g_sum
		g_sum = 0
		for i in range(len(gen)):
			if gen[i][1] * rate > g_up_limit[i]:
				gen[i][1] = g_up_limit[i]
			else:
				gen[i][1] = gen[i][1]*rate
			g_sum+=gen[i][1]

	# if no load or no generator then just return
	if g_sum ==0 or l_sum ==0:
		t=[]
		subcasedata['gen']=np.array(t)  # use gen to control no need condition
		return subcasedata

	# if adjust G not enough the adjust L
	if g_sum != l_sum:
		rate = g_sum / l_sum
		for b in bus:
			b[2] *= rate

	subcasedata['bus'] = np.array(bus) 
	subcasedata['gen'] = np.array(gen)
	subcasedata['time'] = time
	subcasedata['branch'] = np.array(branch)
	# print( subcasedata['gen']
	return subcasedata

if __name__ == '__main__':

	# casedata = case9()
	# allcase =[]
	# k = np.delete(casedata['branch'],[0,6],0)
	# casedata['branch'] = k
	# k = np.delete(casedata['bus'],[0,1],0)
	# casedata['bus'] = k
	# k = np.delete(casedata['gen'],[0,1],0)
	# casedata['gen'] = k
	# CFS_int(casedata)
	# show (casedata)
	# print( casedata['gen'].shape[1]
	# print( casedata['gen']
	# CFS(casedata,allcase)
	# print( casedata
	# print( casedata['branch']
	# print( len(allcase)
	# print(("aaa")
	casedata = case9()
	# print( casedata
	CFS_int(casedata)
	# print( len(casedata['branch'])
	allcase = []
	allcase.append(casedata)
	for i in range(0,20):
		ra = random.randint(0,len(allcase)-1)
		at_casedata = allcase[ra]
		# print( len(allcase),ra 
		# print( allcase
		allcase.remove(at_casedata)

		temp_allcase = []

		random_attack(at_casedata, allcase)
		# print( len(allcase)
		for a in allcase:
			CFS(a,temp_allcase)
		allcase = copy.deepcopy(temp_allcase)
		# print( temp_allcase
		sum = 0
		for i in range (0,len(allcase)):
			sum += len(allcase[i]['branch'])
			# pass
		print( sum)
		if len(allcase) == 0:
			break
		# print( len(get_allcase) )

	# print( len(get_allcase)
	# for i in range (0,len(get_allcase)):
	# 	print( i,': ',get_allcase[i]['gen']

	# print(  get_allcase[1]['bus'],get_allcase[0]['bus']