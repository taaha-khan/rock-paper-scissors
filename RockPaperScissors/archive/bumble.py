
bumble_code = compile("""
#see also www.dllu.net/rps.html
import random
numPre = 54
numMeta = 24
if not input:
	limits = [50,20,10]
	beat={'R':'P','P':'S','S':'R'}
	moves=['','','']
	pScore=[[3]*numPre,[3]*numPre,[3]*numPre,[3]*numPre,[3]*numPre,[3]*numPre,[3]*numPre,[3]*numPre]
	centrifuge={'RP':'a','PS':'b','SR':'c','PR':'d','SP':'e','RS':'f','RR':'g','PP':'h','SS':'i'}
	length=0
	p=[random.choice("RPS")]*numPre
	m=[random.choice("RPS")]*numMeta
	mScore=[3]*numMeta
	threat = [0,0,0]
	outcome = 0
else:
	oldoutcome = outcome
	outcome = (beat[input]==output2) - (input==beat[output2])
	threat[oldoutcome + 1] *= 0.957
	threat[oldoutcome + 1] -= 0.042*outcome
	for i in range(numPre):
		pScore[0][i]=0.8*pScore[0][i]+((input==p[i])-(input==beat[beat[p[i]]]))*3
		pScore[1][i]=0.8*pScore[1][i]+((output==p[i])-(output==beat[beat[p[i]]]))*3
		pScore[2][i]=0.87*pScore[2][i]+(input==p[i])*3.3-(input==beat[p[i]])*0.9-(input==beat[beat[p[i]]])*3
		pScore[3][i]=0.87*pScore[3][i]+(output==p[i])*3.3-(output==beat[p[i]])*0.9-(output==beat[beat[p[i]]])*3
		pScore[4][i]=(pScore[4][i]+(input==p[i])*3)*(1-(input==beat[beat[p[i]]]))
		pScore[5][i]=(pScore[5][i]+(output==p[i])*3)*(1-(output==beat[beat[p[i]]]))
		pScore[6][i]=(pScore[6][i]+(input==p[i])*3)*(1-((input==beat[beat[p[i]]]) or (input==beat[p[i]])))
		pScore[7][i]=(pScore[7][i]+(output==p[i])*3)*(1-((output==beat[beat[p[i]]]) or (output==beat[p[i]])))
	for i in range(numMeta):
		mScore[i]=0.94*mScore[i]+(input==m[i])-(input==beat[beat[m[i]]])
		if input==beat[beat[m[i]]] and random.random()<0.87 or mScore[i]<0:
			mScore[i]=0
	moves[0]+=centrifuge[input+output]
	moves[1]+=input		
	moves[2]+=output
	length+=1
	for z in range(3):
		limit = min([length,limits[z]])
		for y in range(3):
			j=limit
			while j>=1 and not moves[y][length-j:length] in moves[y][0:length-1]:
				j-=1
			if j>=1:
				if random.random()<0.6:
					i = moves[y].rfind(moves[y][length-j:length],0,length-1)
				elif random.random()<0.5:
					i = moves[y].rfind(moves[y][length-j:length],0,length-1)
					i2 = moves[y].rfind(moves[y][length-j:length],0,i)
					if i2!=-1:
						i=i2
				else:
					i = moves[y].find(moves[y][length-j:length],0,length-1)
				p[0+2*y+6*z] = moves[1][j+i] 
				p[1+2*y+6*z] = beat[moves[2][j+i]] 
	
	for i in range(18,18*3):
		p[i]=beat[beat[p[i-18]]]
		
	for i in range(0,8,2):
		m[i]=       p[pScore[i  ].index(max(pScore[i  ]))]
		m[i+1]=beat[p[pScore[i+1].index(max(pScore[i+1]))]]
	for i in range(8,24):
		m[i]=beat[beat[m[i-8]]]
output2 = output = beat[m[mScore.index(max(mScore))]]
if random.random()<0.1 or random.randint(3,40)>length:
	output=beat[random.choice("RPS")]
""", '<string>', 'exec')

bumble_runner = {}

def bumble_agent(obs, config):
	
	global bumble_runner
	global bumble_code

	inp = ''
	if obs.step:
		inp = 'RPS'[obs.lastOpponentAction]
	bumble_runner['input'] = inp
	exec(bumble_code, bumble_runner)

	action = {'R': 0, 'P': 1, 'S': 2}[bumble_runner['output']]
	return action
