dllu1_code = compile(
	"""
import random
numPre = 30
numMeta = 6
if not input:
	limit = 8
	beat={'R':'P','P':'S','S':'R'}
	moves=['','','','']
	pScore=[[5]*numPre,[5]*numPre,[5]*numPre,[5]*numPre,[5]*numPre,[5]*numPre]
	centrifuge={'RP':0,'PS':1,'SR':2,'PR':3,'SP':4,'RS':5,'RR':6,'PP':7,'SS':8}
	centripete={'R':0,'P':1,'S':2}
	soma = [0,0,0,0,0,0,0,0,0];
	rps = [1,1,1];
	a="RPS"
	best = [0,0,0];
	length=0
	p=[random.choice("RPS")]*numPre
	m=[random.choice("RPS")]*numMeta
	mScore=[5,2,5,2,4,2]
else:
	for i in range(numPre):
		pp = p[i]
		bpp = beat[pp]
		bbpp = beat[bpp]
		pScore[0][i]=0.9*pScore[0][i]+((input==pp)-(input==bbpp))*3
		pScore[1][i]=0.9*pScore[1][i]+((output==pp)-(output==bbpp))*3
		pScore[2][i]=0.87*pScore[2][i]+(input==pp)*3.3-(input==bpp)*1.2-(input==bbpp)*2.3
		pScore[3][i]=0.87*pScore[3][i]+(output==pp)*3.3-(output==bpp)*1.2-(output==bbpp)*2.3
		pScore[4][i]=(pScore[4][i]+(input==pp)*3)*(1-(input==bbpp))
		pScore[5][i]=(pScore[5][i]+(output==pp)*3)*(1-(output==bbpp))
	for i in range(numMeta):
		mScore[i]=0.96*(mScore[i]+(input==m[i])-(input==beat[beat[m[i]]]))
	soma[centrifuge[input+output]] +=1;
	rps[centripete[input]] +=1;
	moves[0]+=str(centrifuge[input+output])
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
	j=min([length,limit])
	while j>=2 and not moves[0][length-j:length-1] in moves[0][0:length-2]:
		j-=1
	i = moves[0].rfind(moves[0][length-j:length-1],0,length-2)
	if j+i>=length:
		p[6] = p[7] = random.choice("RPS")
	else:
		p[6] = moves[1][j+i] 
		p[7] = beat[moves[2][j+i]]
		
	best[0] = soma[centrifuge[output+'R']]*rps[0]/rps[centripete[output]]
	best[1] = soma[centrifuge[output+'P']]*rps[1]/rps[centripete[output]]
	best[2] = soma[centrifuge[output+'S']]*rps[2]/rps[centripete[output]]
	p[8] = p[9] = a[best.index(max(best))]
	
	for i in range(10,numPre):
		p[i]=beat[beat[p[i-10]]]
		
	for i in range(0,numMeta,2):
		m[i]=       p[pScore[i  ].index(max(pScore[i  ]))]
		m[i+1]=beat[p[pScore[i+1].index(max(pScore[i+1]))]]
output = beat[m[mScore.index(max(mScore))]]
if max(mScore)<0.07 or random.randint(3,40)>length:
	output=beat[random.choice("RPS")]
""", '<string>', 'exec')

dllu1_runner = {}

def dllu1_agent(obs, config):
	
	global dllu1_runner
	global dllu1_code

	inp = ''
	if obs.step:
		inp = 'RPS'[obs.lastOpponentAction]
	dllu1_runner['input'] = inp
	exec(dllu1_code, dllu1_runner)

	action = {'R': 0, 'P': 1, 'S': 2}[dllu1_runner['output']]
	return action
