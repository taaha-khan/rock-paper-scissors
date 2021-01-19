
bumble_code = compile("""
#see also www.dllu.net/rps.html
import random
numPre = 18
numMeta = 18
if not input:
	limit = 8
	beat={'R':'P','P':'S','S':'R'}
	moves=['','','']
	pScore=[[3]*numPre,[3]*numPre,[3]*numPre,[3]*numPre,[3]*numPre,[3]*numPre]
	centrifuge={'RP':'a','PS':'b','SR':'c','PR':'d','SP':'e','RS':'f','RR':'g','PP':'h','SS':'i'}
	length=0
	p=[random.choice("RPS")]*numPre
	m=[random.choice("RPS")]*numMeta
	mScore=[3]*numMeta
else:
	for i in range(numPre):
		pScore[0][i]=0.8*pScore[0][i]+((input==p[i])-(input==beat[beat[p[i]]]))*3
		pScore[1][i]=0.8*pScore[1][i]+((output==p[i])-(output==beat[beat[p[i]]]))*3
		pScore[2][i]=0.87*pScore[2][i]+(input==p[i])*3.3-(input==beat[p[i]])*0.9-(input==beat[beat[p[i]]])*3
		pScore[3][i]=0.87*pScore[3][i]+(output==p[i])*3.3-(output==beat[p[i]])*0.9-(output==beat[beat[p[i]]])*3
		pScore[4][i]=(pScore[4][i]+(input==p[i])*3)*(1-(input==beat[beat[p[i]]]))
		pScore[5][i]=(pScore[5][i]+(output==p[i])*3)*(1-(output==beat[beat[p[i]]]))
	for i in range(numMeta):
		mScore[i]=(mScore[i]+(input==m[i]))*(1-(input==beat[beat[m[i]]]))
	moves[0]+=centrifuge[input+output]
	moves[1]+=input		
	moves[2]+=output
	length+=1
	for y in range(3):
		j=min([length,limit])
		while j>=1 and not moves[y][length-j:length] in moves[y][0:length-1]:
			j-=1
		i = moves[y].rfind(moves[y][length-j:length],0,length-1)
		p[0+2*y] = moves[1][j+i] 
		p[1+2*y] = beat[moves[2][j+i]] 
	
	for i in range(6,6*3):
		p[i]=beat[beat[p[i-6]]]
		
	for i in range(0,6,2):
		m[i]=       p[pScore[i  ].index(max(pScore[i  ]))]
		m[i+1]=beat[p[pScore[i+1].index(max(pScore[i+1]))]]
	for i in range(6,18):
		m[i]=beat[beat[m[i-6]]]
output = beat[m[mScore.index(max(mScore))]]
if max(mScore)<0.13 or random.randint(3,40)>length:
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
