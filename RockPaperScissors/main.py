
import concurrent.futures as processor
from kaggle_environments import make
from prettytable import from_csv

agents = [

	# Public Bots
	'public/decision_tree.py',
	'public/decision_tree_2.py',
	'public/memory.py',
	'public/rfind.py',
	'public/geometry.py',

	# Previous Contest Winners
	'archive/iocaine.py',
	'archive/greenberg.py',

	# Previous Contest Archive
	'archive/testing.py',
	'archive/IO2.py',
	'archive/dllu1.py',
	'archive/bumble.py',
	'archive/bumble2.py',
	'archive/meta_fix.py',
	'archive/lucker.py',

	# Flagship
	'hydra.py'

]

archive = [agent for agent in agents if agent[:8] == 'archive/']
public  = [agent for agent in agents if agent[:7] == 'public/']

def new_game(player1, player2):

	env = make('rps')

	# Deterministic or easily beatable agents (skip these rounds to save time)
	ez_dubs = ['archive/greenberg.py', 'archive/meta_fix.py', 'archive/testing.py', 'public/rfind.py', 'archive/lucker.py', 'archive/IO2.py']
	
	if player1 == 'hydra.py' and player2 in ez_dubs:
		rewards = [1, 0]
	elif player2 == 'hydra.py' and player1 in ez_dubs:
		rewards = [0, 1]
	else:
		
		env.reset()
		env.run([player1, player2])
		
		json = env.toJSON()
		rewards = json['rewards']

	output = {
		'winner': None,
		'loser': None,
		'draw': False,
		'players': [player1, player2]
	}

	if None in rewards:
		error_index = rewards.index(None)
		rewards[error_index] = -1

	if rewards[0] > rewards[1]:
		output['winner'] = player1
		output['loser'] = player2
	elif rewards[1] > rewards[0]:
		output['winner'] = player2
		output['loser'] = player1
	else:
		output['draw'] = True

	return output

def main(pool, n, evaluate = 'hydra.py'):

	data = { name: {
		'agent': name, 'score': 0, 'win%': 0,
		'wins': 0, 'draws': 0, 'losses': 0,
		'games': 0
	} for name in pool }

	with processor.ProcessPoolExecutor() as executor:

		outputs = []
		games = []

		for player1 in pool:
			for player2 in pool:
				if player1 != player2:
					for _ in range(n):
						games.append(executor.submit(new_game, player1, player2))

		print(f'{len(games)} games scheduled')

		for game in processor.as_completed(games):
			output = game.result(); outputs.append(output)
			print(f"{len(outputs)} games completed: {output['players'][0]} vs {output['players'][1]}")

	for output in outputs:
		for player in output['players']:
			if player == output['winner']:
				data[player]['wins'] += 1
			elif player == output['loser']:
				data[player]['losses'] += 1
				if player == evaluate:
					print(f'{evaluate} was beat by {output["winner"]}')
			elif output['draw']:
				data[player]['draws'] += 1
			data[player]['games'] += 1
			data[player]['win%'] = round(data[player]['wins'] / data[player]['games'], 3)
	
	for player in pool:
		data[player]['score'] = data[player]['wins'] - data[player]['losses'] # + (data[player]['draws'] / 2)
		data[player]['score'] = round(data[player]['score'], 3)

	pool.sort(key = lambda name: data[name]['score'], reverse = True)
	with open('leaderboard/leaderboard.csv', 'w') as file:
		file.write(f"{','.join(data[pool[0]].keys())}")
		for name in pool:
			output = f'\n'#{rank + 1}'
			for item in data[name]:
				output += f'{data[name][item]},'
			output = output[:-1]
			file.write(output)

	with open('leaderboard/leaderboard_table.txt', 'w') as file:
		table = from_csv(open('leaderboard/leaderboard.csv'))
		table.align = 'l'
		string = table.get_string().replace('-', '_').replace('+', '|')
		file.write(string)

def play(agent1, agent2):

	env = make('rps', debug = True)
	env.run([agent1, agent2])

	json = env.toJSON()
	rewards = json['rewards']

	if None not in rewards:
		print(f'{agent1}: {int(rewards[0])} vs {agent2}: {int(rewards[1])}')

if __name__ == '__main__':
	play('hydra.py', 'public/rfind.py')
	# main(agents, 2)
