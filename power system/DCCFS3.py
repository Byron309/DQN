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

def result_sort(basedata,subcasedata):
	# adjust result branch  keep the same order as before
	# branch_index = []
	
	adjust_branch = []
	adjust_time = []
	for i in range(0,len(basedata['branch'])):
		for k in range(0,len(subcasedata['branch'])):
			if subcasedata['branch'][k][0] == basedata['branch'][i][0] and subcasedata['branch'][k][1] == basedata['branch'][i][1]:
				adjust_time.append(subcasedata['time'][k])
				adjust_branch.append(subcasedata['branch'][k])
				# branch_index.append(i)
				break

	subcasedata['time'] = adjust_time
	subcasedata['branch'] = np.array(adjust_branch)

def CFS_int(casedata):
	casedata['time'] = branch_time(casedata)
	if 'areas' in casedata:
		del casedata['areas']
	if 'gencost' in casedata:
		del casedata['gencost']
	# result = rundcpf(casedata)

def branch_time(casedata):    # to get time proportity 
	time = [] # t[0] is accalulate  t[1] is slot add    t[2] is gap   t[3] is state: 0 is out  1 is not
	for b in casedata['branch']:
 		time.append(0)
 	# casedata['time'] = time	
	return time

def decide_out(casedata):  # 5s out of limit will dowm  then time-solt is 1s
	out_slot = []
	no_overload =True
	del_index = 0
	minsolt = 100000
	# print(casedata['time'])
	for i in range(0,len(casedata['branch'])):

		if(round(casedata['branch'][i][5],2) < round(abs(casedata['branch'][i][13]),2)):  # this branch is out-of-branch
			# if overload , calculate the slot : ( 5*0.5*limit - already accumulate ) / gap of out 
			no_overload=False
			temp_slot = ((5*0.5*casedata['branch'][i][5])-casedata['time'][i])/(abs(casedata['branch'][i][13])-abs(casedata['branch'][i][5])) 
			out_slot.append(temp_slot) 
			# if i == 20:
			# 	print((5*0.5*casedata['branch'][i][5]),casedata['time'][i],(abs(casedata['branch'][i][13])-abs(casedata['branch'][i][5])))
			if temp_slot < minsolt:
				minsolt = temp_slot
				del_index = i
		else:
			out_slot.append(-1)

	if no_overload:
		return -1,0
	# print("out_slot:",out_slot)
	for i in range(0,len(casedata['branch'])):
		if(round(casedata['branch'][i][5],2) < round(abs(casedata['branch'][i][13]),2)):  # this branch is out-of-branch
			casedata['time'][i] += minsolt* (abs(casedata['branch'][i][13])-abs(casedata['branch'][i][5]))
	# print (5*0.5*casedata['branch'][del_index][5],'---------------',abs(casedata['branch'][del_index][13])-casedata['branch'][del_index][5])
	# print('after:',casedata['time'])
	return del_index , minsolt

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
	# print(r,len(casedata['branch']))
	casedata['branch'] = np.array(np.delete(casedata['branch'],r,0))
	casedata['time'] = np.delete(casedata['time'],r,0)
	# return casedata
	# print('random')
	subbus,subbranch,subtime,subgen = subgraph(casedata)


	for i in range(0,len(subbus)):
		# print(len(subbranch[i]))
		subcasedata_list = copy.deepcopy(re_dispatch(casedata, subbus[i], subbranch[i], subtime[i], subgen[i], 100000) )# timesolt=1000表示可以瞬间调整
		# print(len(subcasedata_list))
		for subcasedata in subcasedata_list:
			# print(len(subcasedata['gen']))
			result_sort(casedata,subcasedata)
			allcase.append(subcasedata)
			# print( subcasedata['gen']
	# for a in allcase:
	# 	print( "a:",a['gen']

	# do not process if subbus is null

def num_attack(r,casedata,allcase):

	# r= random.randint(0,len(casedata['branch'])-1)
	print("mun_attack:",r,len(casedata['branch']))
	casedata['branch'] = np.array(np.delete(casedata['branch'],r,0))
	casedata['time'] = np.delete(casedata['time'],r,0)
	# return casedata
	# print('random')
	subbus,subbranch,subtime,subgen = subgraph(casedata)


	for i in range(0,len(subbus)):
		# print(len(subbranch[i]))
		subcasedata_list = copy.deepcopy(re_dispatch(casedata, subbus[i], subbranch[i], subtime[i], subgen[i], 100000) )# timesolt=1000表示可以瞬间调整
		# print(len(subcasedata_list))
		for subcasedata in subcasedata_list:
			# print(len(subcasedata['gen']))
			result_sort(casedata,subcasedata)
			allcase.append(subcasedata)

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

	# print('into-cfs',len(casedata['gen']))

	result = rundcpf(casedata)
	
	subcasedata = copy.deepcopy(casedata)
	subcasedata['bus'] = result[0]['bus']
	subcasedata['branch'] = result[0]['branch']
	subcasedata['gen'] = result[0]['gen']

	result_sort(casedata,subcasedata)

	# print("len(subcasedata['branch'] in CFS:",len(subcasedata['branch']))

	del_index, timeslot = decide_out(subcasedata)     # timeslot use to caculate adjust of generator
	# print(del_index,timeslot,"------------------")
	# print( del_index,timeslot)
	# print('del_index:',del_index)
	if del_index != -1:  # means exisit branch to delete
		subcasedata['branch']= np.delete(subcasedata['branch'], del_index, 0)
		subcasedata['time']= np.delete(subcasedata['time'], del_index, 0)

		# print('cfs',len(subcasedata['gen']))
		subbus,subbranch,subtime,subgen = subgraph(subcasedata)  # whether generate subgraph 
		# print(len(subcasedata['branch']), len(subbus),'\n')
		# if len(subbus)>1: # generate subgraph already
		for i in range(0,len(subbus)):
			next_subcasedata_list = copy.deepcopy(re_dispatch(subcasedata, subbus[i], subbranch[i],subtime[i],subgen[i], timeslot))
			# at here gen use to deside whether to run rundcpf
			# no load or no gen anymore
			# print(len(next_subcasedata_list))
			if len(next_subcasedata_list) == 0:
				return
			for next_subcasedata in next_subcasedata_list:
				result_sort(casedata,next_subcasedata)
				CFS(next_subcasedata,allcase)
	else:
		allcase.append(subcasedata)
	# print(len(allcase))

def subgraph(casedata):
	# print(casedata['gen'])
	branch = casedata['branch']
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
	# print(set_subbus)
	# for g in gen:
	# 	print(g[0])
	for i in range(0,len(set_subbus)):
            temp_subbus = []
            temp_subbranch = []
            temp_subtime = []
            temp_gen = []
            for s in set_subbus[i]:
                for b in bus:
                    if s == b[0]:
                        temp_subbus.append(b) 
                        
                for g in gen:
                    if g[0] == s:
                        temp_gen.append(g)
                        # print(g)
                               
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

            # print('len(temp_gen):',len(temp_gen))
	# print(len(casedata['gen']),len(subgen))
	# print( len(subbranch))
	return subbus,subbranch,subtime,subgen

def re_dispatch(casedata, bus, branch, time, gen, timeslot):
	# calculate power of generator and load first
	# adjust input and out put that is gen and bus  
	# 
	adjust_rate = 100000  # per slot  数字越大 瞬调 0.3可能比较合适
	adjust_rate *= timeslot
	
	if len(gen) == 0:
		return []
	# find power of L
	# l_bus = []
	l_sum = 0
	for b in bus:
		l_sum += b[2]

	# if no load   \\  make sure exist not zero load
	if l_sum <= 0:
		return []
	#find power of G and L bus 
	g_sum = 0
	g_up_gap = []
	g_down_gap = []
	g_up_sum = 0
	g_down_sum = 0

	#get the generator propority
	for g in gen:  #g[0] is number g[1] is output power
		# print(g)
		g_sum += g[1]   
		if g[1]*(1+adjust_rate) < g[8]:
			g_up_gap.append(g[1]*adjust_rate)
			g_up_sum += (g[1]*adjust_rate)
		else:
			g_up_gap.append(g[8]-g[1])
			g_up_sum += (g[8]-g[1])

		if g[1]*(1-adjust_rate) > g[9]:
			g_down_gap.append(g[1]*adjust_rate)
			g_down_sum += (g[1]*adjust_rate)
		else:
			g_down_gap.append(g[1]-g[9])
			g_down_sum += (g[1]-g[9])
		# print(g_up_sum)
	# print(g_sum,l_sum,g_down_sum,g_up_sum)

	while True :  # here make sure no any need for out of generator and load 
		# print( 'is here dead?')
		while(True):
			# print('at first dead?')
			# isin = False
			# g_sum=round(g_sum,2)
			# l_sum=round(l_sum,2)
			# print('in redispatch  g_sum:',g_sum, ' l_sum:',l_sum, ' gap g-l:',g_sum-l_sum)
			if round(g_sum,2) > round(l_sum,2) and g_sum-g_down_sum > l_sum:  # means at high to low but not enough  so delete gen
				# isin = True
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
				g_down_sum -= g_down_gap[index]
				g_down_gap = np.delete(g_down_gap, index, 0)
				g_up_sum -= g_up_gap[index]
				g_up_gap = np.delete(g_up_gap, index, 0)

				# if delete generator  also have to delete bus and branch 
				# print ('redispatch')
				bus_delete_list =[]
				for i in range(0,len(bus)):
					if bus[i][0] == g_delete_num:
						l_sum -= bus[i][2]
						bus_delete_list.append(i)
				bus = np.delete(bus,bus_delete_list,0)

				branch_delete_list =[]
				for i in range(0,len(branch)):
					if branch[i][0] ==g_delete_num or branch[i][1]==g_delete_num:
						branch_delete_list.append(i)

				branch = np.delete(branch, branch_delete_list, 0)
				time = np.delete(time, branch_delete_list, 0)
				# print(len(branch_delete_list)
			# if isin == False:
			# 	break
			else:
				break
		while(True):
			# print('at second dead?',l_sum,g_sum,g_down_sum,g_up_sum)
			# isin = False
			# g_sum=round(g_sum,2)
			# l_sum=round(l_sum,2)
			if round(g_sum,2) < round(l_sum,2) and g_sum+g_up_sum < l_sum:  # means at low to high but not enough so delete load 
				# print('Am I in?')
				# isin = True
				index = 0
				l_max = 0 #这就保证 删不到 load = 0的bus
				for i in range(0,len(bus)):
					if bus[i][2] >l_max:
						index = i
						l_max = bus[i][2] 

				# 看最大的load差值是不是够删除
				if l_max > l_sum-(g_sum+g_up_sum): # 削减完还是够的
					bus[index][2]-= l_sum-(g_sum+g_up_sum)
					l_sum = g_sum+g_up_sum

				else:   #把这个bus处理下
					# 是gen 不删 不是的话删掉
					if len(bus) ==0:
						print(l_sum,g_sum,g_down_sum,g_up_sum)
					l_sum -= bus[index][2]

					isgen = False
					for g in gen:
						if g[0] == bus[index][0]:
							isgen = True
					if isgen:
						bus[index][2] = 0
						
					else:  # 删掉 这个bus 那就连带branch删掉
						l_delete_num = bus[index][0]
						bus = np.delete(bus,index,0)
						branch_delete_list =[]
						for i in range(0,len(branch)):
							if branch[i][0] ==l_delete_num or branch[i][1]==l_delete_num:
								branch_delete_list.append(i)
						branch = np.delete(branch, branch_delete_list, 0)
						time = np.delete(time, branch_delete_list, 0)
			else:
				break
			# if isin == False:
			# 	break
		
		if (round(g_sum,2) > round(l_sum,2) and g_sum-g_down_sum > l_sum) or (round(g_sum,2) < round(l_sum,2) and g_sum+g_up_sum < l_sum):
			continue
		else:
			break

	# print(g_sum,l_sum,g_down_sum,g_up_sum)

	temp_gen_sum = g_sum
	if g_sum > l_sum: # g 大的
		if (g_sum-g_down_sum)<=l_sum: # 降低gen 可调 
			for i in range(0,len(gen)):
				gen[i][1] -= ((temp_gen_sum-l_sum)/g_down_sum)*g_down_gap[i]
				g_sum -=((temp_gen_sum-l_sum)/g_down_sum)*g_down_gap[i]
				# print(g_sum)
		else:	
			# 删 gen 降幅不够 前面删过了
			pass

	elif g_sum < l_sum: # g 小的
		if (g_sum+g_up_sum)>=l_sum: # 提高gen 可调 
			for i in range(0,len(gen)):
				gen[i][1] += ((l_sum-temp_gen_sum)/g_up_sum)*g_up_gap[i]
				g_sum += ((l_sum-temp_gen_sum)/g_up_sum)*g_up_gap[i]
		else:
			#不可调 删掉load 前面删掉了
			pass 

	
	# if no load or no generator then just return
	if g_sum ==0 or l_sum ==0:
		return []
	
	subcasedata = copy.deepcopy(casedata)
	subcasedata['bus'] = np.array(bus) 
	subcasedata['gen'] = np.array(gen)
	subcasedata['time'] = time
	subcasedata['branch'] = np.array(branch)
	# print('redispatch')
	subbus,subbranch,subtime,subgen = subgraph(subcasedata)

	# print(subgen)
	# print(len(branch),len(subbranch))
	subcasedata_list =[]
	for i in range(0,len(subbus)):
		subcasedata = copy.deepcopy(casedata)
		subcasedata['bus'] = np.array(subbus[i]) 
		subcasedata['gen'] = np.array(subgen[i])
		subcasedata['time'] = time
		subcasedata['branch'] = np.array(subbranch[i])
		subcasedata_list.append(subcasedata)
		# print(subgen[i])

	# print( subcasedata['gen']
	# print(g_sum,l_sum)
	# print(len(subcasedata_list))
	return subcasedata_list

def get_branch_num(allcase):
	num = 0
	for a in allcase:
		num+=len(a['branch'])
	return num

if __name__ == '__main__':

	casedata = case24_ieee_rts()
	CFS_int(casedata)

	n = len(casedata['branch'])
	c=0
	# print (n)
# ----------------------------------------------
	# for i in range(0,n):
	# 	testdata = copy.deepcopy(casedata)
	# 	allcase = []
		
	# 	temp_allcase = []
	# 	random_attack(i,testdata, allcase)
			
	# 	for a in allcase:
				
	# 		CFS(a,temp_allcase)
	# 	allcase = copy.deepcopy(temp_allcase)

	# 	if len(allcase)!= 1:
	# 		c+=1
	# 		# print(casedata['branch'][i])
	# 		# break
	# 	print(get_branch_num(allcase))
	# print(c)

# -----------------------------------------------
	testdata = copy.deepcopy(casedata)
	allcase = []
	temp_allcase = []
	num_attack(26,testdata, allcase)  # 8 11 12
	# print( len(allcase))
	for a in allcase:
		CFS(a,temp_allcase)
	allcase = copy.deepcopy(temp_allcase)
	# print(len(allcase))

	t =allcase[0]
	# print(len(t['branch']))
	allcase= [] 
	temp_allcase =[]
	num_attack(21,t,allcase)
	for a in allcase:
		CFS(a,temp_allcase)
	allcase = copy.deepcopy(temp_allcase)

	# print(len(allcase))

	num = 0
	for a in temp_allcase:
		num += len(a['branch'])
		# print(len(a['branch']))
	print (38-num)

	# print(round(2.888,0))