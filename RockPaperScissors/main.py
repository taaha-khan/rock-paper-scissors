
from kaggle_environments.envs.rps.agents import *
from kaggle_environments import make, evaluate

from prettytable import from_csv
import random

pool = [

	# Built-In Bots
	'reactionary',
	'counter_reactionary',
	'statistical',

	# Public Bots
	'public/matrix.py',
	'public/markov.py',
	'public/memory.py',
	'public/decision_tree.py',

	# Random Agent
	'random_agent.py',

	# Old Contest Archived Bots
	'archive/iocaine.py',
	'archive/greenberg.py',
	'archive/testing.py',
	'archive/rank1.py',

	# Flagship
	'hydra.py'

]

def main(pool, total_rounds = 200):

	data = {name: {
		'games': 0,
		'wins': 0,
		'losses': 0,
		'draws': 0,
		'score': 0
	} for name in pool}

	env = make('rps', debug = True)

	with open('leaderboard/leaderboard.csv', 'w') as file:
		file.write('rank,name,score,wins,draws,losses,games\n')

	for i in range(total_rounds):

		player1 = random.choice(pool)

		player2 = random.choice(pool)
		while player2 == player1:
			player2 = random.choice(pool)

		print(f'\nGame {i + 1} - {player1} vs {player2}')

		env.reset()
		env.run([player1, player2])
		
		json = env.toJSON()
		rewards = json['rewards']

		if rewards[0] > rewards[1]:
			data[player1]['wins'] += 1
			data[player2]['losses'] += 1
			print(f'{player1} wins: {rewards[0]}')
		elif rewards[1] > rewards[0]:
			data[player2]['wins'] += 1
			data[player1]['losses'] += 1
			print(f'{player2} wins: {rewards[1]}')
		else:
			data[player1]['draws'] += 1
			data[player2]['draws'] += 1
			print(f'Draw Game')
		
		data[player1]['games'] += 1
		data[player2]['games'] += 1

		data[player1]['score'] = data[player1]['wins'] / data[player1]['games']
		data[player2]['score'] = data[player2]['wins'] / data[player2]['games']

	pool = list(set(pool))
	pool.sort(key = lambda name: data[name]['score'], reverse = True)
	with open('leaderboard/leaderboard.csv', 'a') as file:
		for rank, name in enumerate(pool):
			info = data[name]
			file.write(f"{rank + 1},{name},{round(info['score'], 3)},{info['wins']},{info['draws']},{info['losses']},{info['games']}\n")

	with open('leaderboard/leaderboard_table.txt', 'w') as file:
		file.write(str(from_csv(open('leaderboard.csv'))))

if __name__ == '__main__':
	main(pool)