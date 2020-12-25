code = compile("""
#This one is just for the proof that luck plays an important role in the leaderboard
#only 200 matches but there are more than 650 ai score > 5000 
import random

num_predictors =27
num_meta= 18

if input =="":
	len_rfind = [20]
	limit = [10,20,60]
	beat = { "P":"S" , "R":"P" , "S":"R" }
	not_lose = { "R":"PR", "P":"SP", "S":"RS" } 
	your_his =""
	my_his = ""
	both_his=""
	both_his2=""
	length =0
	score1=[3]*num_predictors
	score2=[3]*num_predictors
	score3=[3]*num_predictors
	score4=[3]*num_predictors
	score5=[3]*num_predictors
	score6=[3]*num_predictors
	metascore=[3]*num_meta
	temp1 = { "PP":"1","PR":"2","PS":"3",
			  "RP":"4","RR":"5","RS":"6",
			  "SP":"7","SR":"8","SS":"9"}
	temp2 = { "1":"PP","2":"PR","3":"PS",
				"4":"RP","5":"RR","6":"RS",
				"7":"SP","8":"SR","9":"SS"} 
	who_win = { "PP": 0, "PR":1 , "PS":-1,
				"RP": -1,"RR":0, "RS":1,
				"SP": 1, "SR":-1, "SS":0}
	index = { "P":0, "R":1, "S":2 }
	chance =[0]*num_predictors
	chance2 =[0]*num_predictors
	output = random.choice("RPS")
	predictors = [output]*num_predictors
	metapredictors = [output]*num_meta
else:
	#calculate score
	for i in range(num_predictors):
		#meta 1
		score1[i]*=0.8
		if input==predictors[i]:
			score1[i]+=3
		else:
			score1[i]-=3
		#meta 2
		if input==predictors[i]:
			score2[i]+=3
		else:
			score2[i]=0
		#meta 3
		score3[i]*=0.8
		if output==predictors[i]:
			score3[i]+=3
		else:
		   score3[i]-=3
		#meta 4
		if output==predictors[i]:
			score4[i]+=3
		else:
			score4[i]=0
		#meta 5
		score5[i]*=0.8
		if input==predictors[i]:
			score5[i]+=3
		else:
			if chance[i]==1:
				chance[i]=0
				score5[i]-=3
			else:
				chance[i]=1
				score5[i]=0
		#meta 6
		score6[i]*=0.8
		if output==predictors[i]:
			score6[i]+=3
		else:
			if chance2[i]==1:
				chance2[i]=0
				score6[i]-=3
			else:
				chance2[i]=1
				score6[i]=0
	#calculate metascore
	for i in range(num_meta):
		metascore[i]*=0.9
		if input==metapredictors[i]:
			metascore[i]+=3
		else:
			metascore[i]=0
	#Predictors
	#if length>1:
	#    output=beat[predict]
	your_his+=input
	my_his+=output
	both_his+=temp1[(input+output)]
	both_his2+=temp1[(output+input)]
	length+=1

	#history matching
	for i in range(1):
		len_size = min(length,len_rfind[i])
		j=len_size
		#both_his
		while j>=1 and not both_his[length-j:length] in both_his[0:length-1]:
			j-=1
		if j>=1:
			k = both_his.rfind(both_his[length-j:length],0,length-1)
			predictors[0+6*i] = your_his[j+k]
			predictors[1+6*i] = beat[my_his[j+k]]
		else:
			predictors[0+6*i] = random.choice("RPS")
			predictors[1+6*i] = random.choice("RPS")
		j=len_size
		#your_his
		while j>=1 and not your_his[length-j:length] in your_his[0:length-1]:
			j-=1
		if j>=1:
			k = your_his.rfind(your_his[length-j:length],0,length-1)
			predictors[2+6*i] = your_his[j+k]
			predictors[3+6*i] = beat[my_his[j+k]]
		else:
			predictors[2+6*i] = random.choice("RPS")
			predictors[3+6*i] = random.choice("RPS")
		j=len_size
		#my_his
		while j>=1 and not my_his[length-j:length] in my_his[0:length-1]:
			j-=1
		if j>=1:
			k = my_his.rfind(my_his[length-j:length],0,length-1)
			predictors[4+6*i] = your_his[j+k]
			predictors[5+6*i] = beat[my_his[j+k]]
		else:
			predictors[4+6*i] = random.choice("RPS")
			predictors[5+6*i] = random.choice("RPS")
	
	#Reverse
	for i in range(3):
		temp =""
		search = temp1[(output+input)] #last round
		for start in range(2, min(limit[i],length) ):
			if search == both_his2[length-start]:
				temp+=both_his2[length-start+1]
		if(temp==""):
			predictors[6+i] = random.choice("RPS")
		else:
			collectR = {"P":0,"R":0,"S":0} #take win/lose from opponent into account
			for sdf in temp:
				next_move = temp2[sdf]
				if(who_win[next_move]==-1):
					collectR[temp2[sdf][1]]+=3
				elif(who_win[next_move]==0):
					collectR[temp2[sdf][1]]+=1
				elif(who_win[next_move]==1):
					collectR[beat[temp2[sdf][0]]]+=1
			max1 = -1
			p1 =""
			for key in collectR:
				if(collectR[key]>max1):
					max1 = collectR[key]
					p1 += key
			predictors[6+i] = random.choice(p1)
	
	for i in range(9,27):
		predictors[i]=beat[beat[predictors[i-9]]]
	#find prediction for each meta
	metapredictors[0]=predictors[score1.index(max(score1))]
	metapredictors[1]=predictors[score2.index(max(score2))]
	metapredictors[2]=beat[predictors[score3.index(max(score3))]]
	metapredictors[3]=beat[predictors[score4.index(max(score4))]]
	metapredictors[4]=predictors[score5.index(max(score5))]
	metapredictors[5]=beat[predictors[score6.index(max(score6))]]
	for i in range(6,18):
		metapredictors[i] = beat[metapredictors[i-6]]
	
	predict = metapredictors[metascore.index(max(metascore))]
	# output = beat[predict]
	output = random.choice(not_lose[predict])
""", '<string>', 'exec')

runner = {}

def agent(obs, config):
	
	global runner
	global code

	inp = ''
	if obs.step:
		inp = 'RPS'[obs.lastOpponentAction]
	runner['input'] = inp
	exec(code, runner)

	action = {'R': 0, 'P': 1, 'S': 2}[runner['output']]
	return action
