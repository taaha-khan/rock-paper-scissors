
from kaggle_environments.envs.rps.agents import *
from kaggle_environments import make

import concurrent.futures as processor
from prettytable import from_csv
import random, time

agents = [

	# Public Bots
	'public/matrix.py',
	'public/markov.py',
	'public/memory.py',
	'public/decision_tree.py',
	'public/decision_tree_2.py',

	# Old Contest Archived Bots
	'archive/iocaine.py',
	'archive/greenberg.py',
	'archive/testing.py',
	'archive/rank1.py',

	# Flagship
	'hydra.py'

]

def new_game(player1, player2):

	env = make('rps')

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

def main(pool, n = 2):

	data = { name: {
		'name': name, 'score': 0, 'win%': 0,
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
			outputs.append(game.result())
			print(f"{len(outputs)} games completed")

	for output in outputs:
		for player in output['players']:
			if player == output['winner']:
				data[player]['wins'] += 1
			elif player == output['loser']:
				data[player]['losses'] += 1
			elif output['draw']:
				data[player]['draws'] += 1
			data[player]['games'] += 1
			data[player]['win%'] = round(data[player]['wins'] / data[player]['games'], 3)
	
	for player in pool:
		data[player]['score'] = data[player]['wins'] + (data[player]['draws'] / 2)
		data[player]['score'] -= data[player]['losses']
		data[player]['score'] = round(data[player]['score'], 3)

	pool.sort(key = lambda name: data[name]['score'], reverse = True)
	with open('leaderboard/leaderboard.csv', 'w') as file:
		file.write(f"rank,{','.join(data[list(data.keys())[0]].keys())}")
		for rank, name in enumerate(pool):
			info = data[name]
			output = f'\n{rank + 1}'
			for item in info:
				output += f',{info[item]}'
			file.write(output)

	with open('leaderboard/leaderboard_table.txt', 'w') as file:
		file.write(str(from_csv(open('leaderboard/leaderboard.csv'))))

def play(agent1, agent2):
	
	env = make('rps', debug = True)
	env.run([agent1, agent2])

	json = env.toJSON()
	rewards = json['rewards']

	print(f'{agent1}: {rewards[0]}  vs. {agent2}: {rewards[1]}')

if __name__ == '__main__':
	# play('hydra.py', 'public/memory.py')
	main(agents)