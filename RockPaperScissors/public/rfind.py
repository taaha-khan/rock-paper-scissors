# = * = * = * = * = * = * 

max_limit = 23  # can be modified
add_rotations = True

# number of predictors
numPre = 6
if add_rotations:
	numPre *= 3

# number of meta-predictors
numMeta = 4
if add_rotations:
	numMeta *= 3

# saves history
moves = ['', '', '']

beat = {'R':'P', 'P':'S', 'S':'R'}
dna =  {'RP':0, 'PS':1, 'SR':2,
		'PR':3, 'SP':4, 'RS':5,
		'RR':6, 'PP':7, 'SS':8}

p = ["P"]*numPre
m = ["P"]*numMeta
pScore = [[0]*numPre for i in range(8)]
mScore = [0]*numMeta

length = 0
threat = 0
output = "P"


def myagent(observation, configuration):    
	global max_limit, add_rotations, \
		numPre, numMeta, moves, beat, dna, \
		p, m, pScore, mScore, length, threat, output

	if observation.step < 2:
		output = beat[output]
		return {'R':0, 'P':1, 'S':2}[output]

	# - - - -

	input = "RPS"[observation.lastOpponentAction]

	# threat of opponent
	outcome = (beat[input]==output) - (input==beat[output])
	threat = 0.9*threat - 0.1*outcome
	
	# refresh pScore
	for i in range(numPre):
		pp = p[i]
		bpp = beat[pp]
		bbpp = beat[bpp]
		pScore[0][i] = 0.9*pScore[0][i] + 0.1*((input==pp)-(input==bbpp))
		pScore[1][i] = 0.9*pScore[1][i] + 0.1*((output==pp)-(output==bbpp))
		pScore[2][i] = 0.8*pScore[2][i] + 0.3*((input==pp)-(input==bbpp)) + \
						0.1*(length % 3 - 1)
		pScore[3][i] = 0.8*pScore[3][i] + 0.3*((output==pp)-(output==bbpp)) + \
						0.1*(length % 3 - 1)

	# refresh mScore
	for i in range(numMeta):
		mScore[i] = 0.9*mScore[i] + 0.1*((input==m[i])-(input==beat[beat[m[i]]])) + \
					0.05*(length % 5 - 2)

	# refresh moves
	moves[0] += str(dna[input+output])
	moves[1] += input
	moves[2] += output

	# refresh length
	length += 1

	# new predictors
	limit = min([length,max_limit])
	for y in range(3):	# my moves, his, and both
		j = limit
		while j>=1 and not moves[y][length-j:length] in moves[y][0:length-1]:
			j-=1
		if j>=1:
			i = moves[y].rfind(moves[y][length-j:length],0,length-1)
			p[0+2*y] = moves[1][j+i] 
			p[1+2*y] = beat[moves[2][j+i]]

	# rotations of predictors
	if add_rotations:
		for i in range(int(numPre/3),numPre):
			p[i]=beat[beat[p[i-int(numPre/3)]]]

	# new meta
	for i in range(0,4,2):
		m[i] = p[pScore[i].index(max(pScore[i]))]
		m[i+1] = beat[p[pScore[i+1].index(max(pScore[i+1]))]]

	# rotations of meta
	if add_rotations:
		for i in range(4,12):
			m[i]=beat[beat[m[i-4]]]
	
	# - - -
	
	output = beat[m[mScore.index(max(mScore))]]

	if threat > 0.4:
		# ah take this!
		output = beat[beat[output]]

	return {'R':0, 'P':1, 'S':2}[output]