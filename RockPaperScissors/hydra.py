
'''
 __ __ __ __ __ __ __ __ __ __ __ __ __ __ __
|                                            |
|   Copyright (C) 2020 Taaha Khan @taahakhan |
|   "Kaggle RPS Hydra Agent" - V.7.0         |
|   Rock Paper Scissors Algorithm with hydra |
|   net of strong agents and meta-strategy   |
|   selectors to pick the best agent and     |
|   return the best predicted action.        |
|__ __ __ __ __ __ __ __ __ __ __ __ __ __ __|

'''

from sklearn.tree import DecisionTreeClassifier
from collections import defaultdict, namedtuple
from typing import List, Dict
import numpy as np
import operator
import getpass
import random
import time

LOCAL_MODE = getpass.getuser() == 'taaha'
PRINT_OUTPUT = True

class Agent():
	''' Base class for all agents '''

	def initial_step(self, obs, config):
		''' Move to play on initial step '''
		return self.step(obs, config)
	
	def step(self, obs, config):
		''' Agent actions per step '''
		return random.randrange(3)
	
	def get_action(self, obs, config):
		''' Universal Constant Get Action '''
		if obs.step == 0:
			return int(self.initial_step(obs, config))
		return int(self.step(obs, config))

	def set_last_action(self, action):
		''' Overwriting Personal Agent History '''
		return None
	

# Decision Tree Classifier: https://www.kaggle.com/alexandersamarin/decision-tree-classifier
class DecisionTree(Agent):

	def __init__(self, noise = False):

		# Globals
		self.rollouts_hist = {'steps': [], 'actions': [], 'opp-actions': []}
		self.last_move = {'step': 0, 'action': random.randrange(3)}
		self.data = {'x': [], 'y': []}
		self.test_sample = None
		self.noise = noise

		# Hyperparameters
		self.min_samples = 25
		self.k = 5
		
	def construct_local_features(self, rollouts):
		features = np.array([[step % k for step in rollouts['steps']] for k in (2, 3, 5)])
		features = np.append(features, rollouts['steps'])
		features = np.append(features, rollouts['actions'])
		features = np.append(features, rollouts['opp-actions'])
		return features

	def construct_global_features(self, rollouts):
		features = []
		for key in ['actions', 'opp-actions']:
			for i in range(3):
				actions_count = np.mean([r == i for r in rollouts[key]])
				features.append(actions_count)
		return np.array(features)

	def construct_features(self, short_stat_rollouts, long_stat_rollouts):
		lf = self.construct_local_features(short_stat_rollouts)
		gf = self.construct_global_features(long_stat_rollouts)
		features = np.concatenate([lf, gf])
		return features

	def predict_opponent_move(self, train_data, test_sample):
		classifier = DecisionTreeClassifier(random_state = 42)
		classifier.fit(train_data['x'], train_data['y'])
		return classifier.predict(test_sample)

	def update_rollouts_hist(self, rollouts_hist, last_move, opp_last_action):
		rollouts_hist['steps'].append(last_move['step'])
		rollouts_hist['actions'].append(last_move['action'])
		rollouts_hist['opp-actions'].append(opp_last_action)
		return rollouts_hist

	def init_training_data(self, rollouts_hist, k, data):
		for i in range(len(rollouts_hist['steps']) - k + 1):
			short_stat_rollouts = {key: rollouts_hist[key][i : i + k] for key in rollouts_hist}
			long_stat_rollouts = {key: rollouts_hist[key][:i + k] for key in rollouts_hist}
			features = self.construct_features(short_stat_rollouts, long_stat_rollouts)        
			data['x'].append(features)
		test_sample = data['x'][-1].reshape(1, -1)
		data['x'] = data['x'][:-1]
		data['y'] = rollouts_hist['opp-actions'][k:]
		return (data, test_sample)

	def warmup_strategy(self, observation, configuration):
		action = random.randrange(3)
		self.rollouts_hist = self.update_rollouts_hist(self.rollouts_hist, self.last_move, observation.lastOpponentAction)
		self.last_move = {'step': observation.step, 'action': action}
		return action

	def initial_step(self, observation, configuration):
		self.last_move = {'step': 0, 'action': 0}
		self.rollouts_hist = {'steps': [], 'actions': [], 'opp-actions': []}
		return 0

	def step(self, observation, configuration):

		k = self.k

		if observation.step <= self.min_samples + k:
			return self.warmup_strategy(observation, configuration)
	
		# update statistics
		self.rollouts_hist = self.update_rollouts_hist(self.rollouts_hist, self.last_move, observation.lastOpponentAction)
		
		# update training data
		if len(self.data['x']) == 0:
			self.data, self.test_sample = self.init_training_data(self.rollouts_hist, k, self.data)
		else:
			short_stat_rollouts = {key: self.rollouts_hist[key][-k:] for key in self.rollouts_hist}
			features = self.construct_features(short_stat_rollouts, self.rollouts_hist)
			self.data['x'].append(self.test_sample[0])
			self.data['y'] = self.rollouts_hist['opp-actions'][k:]
			self.test_sample = features.reshape(1, -1)
			
		# predict opponents move and choose an action
		next_opp_action_pred = self.predict_opponent_move(self.data, self.test_sample)
		action = int((next_opp_action_pred + 1) % 3)

		if self.noise:
			if random.random() < 0.2:
				action = random.randrange(3)

		self.last_move = {'step': observation.step, 'action': action}
		return action
	
	def set_last_action(self, action):
		self.last_move['action'] = action


# Decision Tree 2: https://www.kaggle.com/jamesmcguigan/rock-paper-scissors-multi-stage-decision-tree
class DecisionTree2(Agent):
	def __init__(self):

		# Initialize starting history
		self.history = {
			"step":        [],
			"prediction1": [],
			"prediction2": [],
			"expected":    [],
			"action":      [],
			"opponent":    [],
		}

	def get_winstats(self, history) -> Dict[str,int]:
		total = len(history['action'])
		wins = 0
		draw = 0
		loss = 0 
		for n in range(total):
			if   history['action'][n] == history['opponent'][n] + 1: wins +=  1
			elif history['action'][n] == history['opponent'][n]:     draw +=  1
			elif history['action'][n] == history['opponent'][n] - 1: loss +=  1
		return { "wins": wins, "draw": draw, "loss": loss }

	def get_winrate(self, history):
		winstats = self.get_winstats(history)
		winrate  = winstats['wins'] / (winstats['wins'] + winstats['loss']) if (winstats['wins'] + winstats['loss']) else 0
		return winrate

	def step(self, observation, configuration, window=9, stages=3, random_freq=0.00, warmup_period=10, max_samples=1000):    

		warmup_period   = warmup_period
		models          = [ None ] + [ DecisionTreeClassifier() ] * stages
		actions         = list(range(configuration.signs))  # [0,1,2]
		step            = observation.step
		last_action     = self.history['action'][-1] if len(self.history['action']) else 2
		opponent_action = observation.lastOpponentAction if observation.step > 0   else 2
			
		if observation.step > 0:
			self.history['opponent'].append(opponent_action)
			
		winrate  = self.get_winrate(self.history)
		winstats = self.get_winstats(self.history)
		
		# Set default values     
		prediction1 = random.randint(0,2)
		prediction2 = random.randint(0,2)
		prediction3 = random.randint(0,2)
		expected    = random.randint(0,2)

		# We need at least some turns of history for DecisionTreeClassifier to work
		if observation.step >= window:
			# First we try to predict the opponents next move based on move history
			try:
				n_start = max(1, len(self.history['opponent']) - window - max_samples) 
				if stages >= 1:
					X = np.stack([
						np.array([
							self.history['action'][n:n+window], 
							self.history['opponent'][n:n+window]
						]).flatten()
						for n in range(n_start,len(self.history['opponent'])-window-warmup_period) 
					])
					Y = np.array([
						self.history['opponent'][n+window]
						for n in range(n_start,len(self.history['opponent'])-window-warmup_period) 
					])  
					Z = np.array([
						self.history['action'][-window+1:] + [ last_action ], 
						self.history['opponent'][-window:] 
					]).flatten().reshape(1, -1)

					models[1].fit(X, Y)
					expected = prediction1 = models[1].predict(Z)[0]

				if stages >= 2:
					# Now retrain including prediction history
					X = np.stack([
						np.array([
							self.history['action'][n:n+window], 
							self.history['prediction1'][n:n+window],
							self.history['opponent'][n:n+window],
						]).flatten()
						for n in range(n_start,len(self.history['opponent'])-window-warmup_period) 
					])
					Y = np.array([
						self.history['opponent'][n+window]
						for n in range(n_start,len(self.history['opponent'])-window-warmup_period) 
					])  
					Z = np.array([
						self.history['action'][-window+1:]      + [ last_action ], 
						self.history['prediction1'][-window+1:] + [ prediction1 ],
						self.history['opponent'][-window:] 
					]).flatten().reshape(1, -1)

					models[2].fit(X, Y)
					expected = prediction2 = models[2].predict(Z)[0]

				if stages >= 3:
					# Now retrain including prediction history
					X = np.stack([
						np.array([
							self.history['action'][n:n+window], 
							self.history['prediction1'][n:n+window],
							self.history['prediction2'][n:n+window],
							self.history['opponent'][n:n+window],
						]).flatten()
						for n in range(n_start,len(history['opponent'])-window-warmup_period) 
					])
					Y = np.array([
						self.history['opponent'][n+window]
						for n in range(n_start,len(history['opponent'])-window-warmup_period) 
					])  
					Z = np.array([
						self.history['action'][-window+1:]      + [ last_action ], 
						self.history['prediction1'][-window+1:] + [ prediction1 ],
						self.history['prediction2'][-window+1:] + [ prediction2 ],
						self.history['opponent'][-window:] 
					]).flatten().reshape(1, -1)

					models[3].fit(X, Y)
					expected = prediction3 = models[3].predict(Z)[0]
			
			except Exception as exception:
				# print(exception)
				pass
						
		if (observation.step <= max(warmup_period,window)):
			action = random.randrange(3)
		elif (random.random() <= random_freq):
			action = random.randrange(3)
		else:
			action = (expected + 1) % configuration.signs
		
		# Persist state
		self.history['step'].append(step)
		self.history['prediction1'].append(prediction1)
		self.history['prediction2'].append(prediction2)
		self.history['expected'].append(expected)
		self.history['action'].append(action)
		if observation.step == 0:  # keep arrays equal length
			self.history['opponent'].append(random.randint(0, 2))

		return int(action)
	
	def set_last_action(self, action):
		self.history['action'][-1] = action


# Greenberg Agent: https://www.kaggle.com/group16/rps-roshambo-competition-greenberg
class Greenberg(Agent):

	def __init__(self):
		self.my_hist = []
		self.opponent_hist = []
		self.act = None
	
	def player(self, my_moves, opp_moves):

		rps_to_text = ('rock','paper','scissors')
		rps_to_num  = {'rock':0, 'paper':1, 'scissors':2}
		wins_with = (1,2,0)      #superior
		best_without = (2,0,1)   #inferior

		lengths = (10, 20, 30, 40, 49, 0)
		p_random = random.choice([0,1,2])  #called 'guess' in iocaine

		TRIALS = 1000
		score_table =((0,-1,1),(1,0,-1),(-1,1,0))
		T = len(opp_moves)  #so T is number of trials completed

		def min_index(values):
			return min(enumerate(values), key=operator.itemgetter(1))[0]

		def max_index(values):
			return max(enumerate(values), key=operator.itemgetter(1))[0]

		def find_best_prediction(l):  # l = len
			bs = -TRIALS
			bp = 0
			if self.p_random_score > bs:
				bs = self.p_random_score
				bp = p_random
			for i in range(3):
				for j in range(24):
					for k in range(4):
						new_bs = self.p_full_score[T%50][j][k][i] - (self.p_full_score[(50+T-l)%50][j][k][i] if l else 0)
						if new_bs > bs:
							bs = new_bs
							bp = (self.p_full[j][k] + i) % 3
					for k in range(2):
						new_bs = self.r_full_score[T%50][j][k][i] - (self.r_full_score[(50+T-l)%50][j][k][i] if l else 0)
						if new_bs > bs:
							bs = new_bs
							bp = (self.r_full[j][k] + i) % 3
				for j in range(2):
					for k in range(2):
						new_bs = self.p_freq_score[T%50][j][k][i] - (self.p_freq_score[(50+T-l)%50][j][k][i] if l else 0)
						if new_bs > bs:
							bs = new_bs
							bp = (self.p_freq[j][k] + i) % 3
						new_bs = self.r_freq_score[T%50][j][k][i] - (self.r_freq_score[(50+T-l)%50][j][k][i] if l else 0)
						if new_bs > bs:
							bs = new_bs
							bp = (self.r_freq[j][k] + i) % 3
			return bp

		if not my_moves:
			self.opp_history = [0]  #pad to match up with 1-based move indexing in original
			self.my_history = [0]
			self.gear = [[0] for _ in range(24)]
			# init()
			self.p_random_score = 0
			self.p_full_score = [[[[0 for i in range(3)] for k in range(4)] for j in range(24)] for l in range(50)]
			self.r_full_score = [[[[0 for i in range(3)] for k in range(2)] for j in range(24)] for l in range(50)]
			self.p_freq_score = [[[[0 for i in range(3)] for k in range(2)] for j in range(2)] for l in range(50)]
			self.r_freq_score = [[[[0 for i in range(3)] for k in range(2)] for j in range(2)] for l in range(50)]
			self.s_len = [0] * 6

			self.p_full = [[0,0,0,0] for _ in range(24)]
			self.r_full = [[0,0] for _ in range(24)]
		else:
			self.my_history.append(rps_to_num[my_moves[-1]])
			self.opp_history.append(rps_to_num[opp_moves[-1]])
			# update_scores()
			self.p_random_score += score_table[p_random][self.opp_history[-1]]
			self.p_full_score[T%50] = [[[self.p_full_score[(T+49)%50][j][k][i] + score_table[(self.p_full[j][k] + i) % 3][self.opp_history[-1]] for i in range(3)] for k in range(4)] for j in range(24)]
			self.r_full_score[T%50] = [[[self.r_full_score[(T+49)%50][j][k][i] + score_table[(self.r_full[j][k] + i) % 3][self.opp_history[-1]] for i in range(3)] for k in range(2)] for j in range(24)]
			self.p_freq_score[T%50] = [[[self.p_freq_score[(T+49)%50][j][k][i] + score_table[(self.p_freq[j][k] + i) % 3][self.opp_history[-1]] for i in range(3)] for k in range(2)] for j in range(2)]
			self.r_freq_score[T%50] = [[[self.r_freq_score[(T+49)%50][j][k][i] + score_table[(self.r_freq[j][k] + i) % 3][self.opp_history[-1]] for i in range(3)] for k in range(2)] for j in range(2)]
			self.s_len = [s + score_table[p][self.opp_history[-1]] for s,p in zip(self.s_len,self.p_len)]

		if not my_moves:
			self.my_history_hash = [[0],[0],[0],[0]]
			self.opp_history_hash = [[0],[0],[0],[0]]
		else:
			self.my_history_hash[0].append(self.my_history[-1])
			self.opp_history_hash[0].append(self.opp_history[-1])
			for i in range(1,4):
				self.my_history_hash[i].append(self.my_history_hash[i-1][-1] * 3 + self.my_history[-1])
				self.opp_history_hash[i].append(self.opp_history_hash[i-1][-1] * 3 + self.opp_history[-1])

		for i in range(24):
			self.gear[i].append((3 + self.opp_history[-1] - self.p_full[i][2]) % 3)
			if T > 1:
				self.gear[i][T] += 3 * self.gear[i][T-1]
			self.gear[i][T] %= 9 # clearly there are 9 different gears, but original code only allocated 3 gear_freq's
								# code apparently worked, but got lucky with undefined behavior
								# I fixed by allocating gear_freq with length = 9
		if not my_moves:
			self.freq = [[0,0,0],[0,0,0]]
			value = [[0,0,0],[0,0,0]]
		else:
			self.freq[0][self.my_history[-1]] += 1
			self.freq[1][self.opp_history[-1]] += 1
			value = [[(1000 * (self.freq[i][2] - self.freq[i][1])) / float(T),
					(1000 * (self.freq[i][0] - self.freq[i][2])) / float(T),
					(1000 * (self.freq[i][1] - self.freq[i][0])) / float(T)] for i in range(2)]
		self.p_freq = [[wins_with[max_index(self.freq[i])], wins_with[max_index(value[i])]] for i in range(2)]
		self.r_freq = [[best_without[min_index(self.freq[i])], best_without[min_index(value[i])]] for i in range(2)]

		f = [[[[0,0,0] for k in range(4)] for j in range(2)] for i in range(3)]
		t = [[[0,0,0,0] for j in range(2)] for i in range(3)]

		m_len = [[0 for _ in range(T)] for i in range(3)]

		for i in range(T-1,0,-1):
			m_len[0][i] = 4
			for j in range(4):
				if self.my_history_hash[j][i] != self.my_history_hash[j][T]:
					m_len[0][i] = j
					break
			for j in range(4):
				if self.opp_history_hash[j][i] != self.opp_history_hash[j][T]:
					m_len[1][i] = j
					break
			for j in range(4):
				if self.my_history_hash[j][i] != self.my_history_hash[j][T] or self.opp_history_hash[j][i] != self.opp_history_hash[j][T]:
					m_len[2][i] = j
					break

		for i in range(T-1,0,-1):
			for j in range(3):
				for k in range(m_len[j][i]):
					f[j][0][k][self.my_history[i+1]] += 1
					f[j][1][k][self.opp_history[i+1]] += 1
					t[j][0][k] += 1
					t[j][1][k] += 1

					if t[j][0][k] == 1:
						self.p_full[j*8 + 0*4 + k][0] = wins_with[self.my_history[i+1]]
					if t[j][1][k] == 1:
						self.p_full[j*8 + 1*4 + k][0] = wins_with[self.opp_history[i+1]]
					if t[j][0][k] == 3:
						self.p_full[j*8 + 0*4 + k][1] = wins_with[max_index(f[j][0][k])]
						self.r_full[j*8 + 0*4 + k][0] = best_without[min_index(f[j][0][k])]
					if t[j][1][k] == 3:
						self.p_full[j*8 + 1*4 + k][1] = wins_with[max_index(f[j][1][k])]
						self.r_full[j*8 + 1*4 + k][0] = best_without[min_index(f[j][1][k])]

		for j in range(3):
			for k in range(4):
				self.p_full[j*8 + 0*4 + k][2] = wins_with[max_index(f[j][0][k])]
				self.r_full[j*8 + 0*4 + k][1] = best_without[min_index(f[j][0][k])]

				self.p_full[j*8 + 1*4 + k][2] = wins_with[max_index(f[j][1][k])]
				self.r_full[j*8 + 1*4 + k][1] = best_without[min_index(f[j][1][k])]

		for j in range(24):
			gear_freq = [0] * 9 # was [0,0,0] because original code incorrectly only allocated array length 3

			for i in range(T-1,0,-1):
				if self.gear[j][i] == self.gear[j][T]:
					gear_freq[self.gear[j][i+1]] += 1

			#original source allocated to 9 positions of gear_freq array, but only allocated first three
			#also, only looked at first 3 to find the max_index
			#unclear whether to seek max index over all 9 gear_freq's or just first 3 (as original code)
			self.p_full[j][3] = (self.p_full[j][1] + max_index(gear_freq)) % 3

		self.p_len = [find_best_prediction(l) for l in lengths]
		return rps_to_num[rps_to_text[self.p_len[max_index(self.s_len)]]]

	def initial_step(self, obs, config):
		self.act = self.player(False, self.opponent_hist)
		return self.act

	def step(self, obs, config):
		rps_to_text = ('rock','paper','scissors')
		self.opponent_hist.append(rps_to_text[obs.lastOpponentAction])
		self.act = self.player(self.my_hist, self.opponent_hist)
		return self.act
	
	def set_last_action(self, action):
		rps_to_text = ('rock','paper','scissors')
		self.my_hist.append(rps_to_text[action])


# Iocane Powder Agent: https://www.kaggle.com/group16/rps-roshambo-comp-iocaine-powder
class Iocaine(Agent):

	def __init__(self):
		self.iocaine = None
		
	@classmethod
	def recall(cls, age, hist):
		"""Looking at the last 'age' points in 'hist', finds the
		last point with the longest similarity to the current point,
		returning 0 if none found."""
		end, length = 0, 0
		for past in range(1, min(age + 1, len(hist) - 1)):
			if length >= len(hist) - past: break
			for i in range(-1 - length, 0):
				if hist[i - past] != hist[i]: break
			else:
				for length in range(length + 1, len(hist) - past):
					if hist[-past - length - 1] != hist[-length - 1]: break
				else: length += 1
				end = len(hist) - past
		return end

	class Stats:
		"""Maintains three running counts and returns the highest count based
			on any given time horizon and threshold."""
		def __init__(self):
			self.sum = [[0, 0, 0]]
		def add(self, move, score):
			self.sum[-1][move] += score
		def advance(self):
			self.sum.append(self.sum[-1])
		def max(self, age, default, score):
			if age >= len(self.sum): diff = self.sum[-1]
			else: diff = [self.sum[-1][i] - self.sum[-1 - age][i] for i in range(3)]
			m = max(diff)
			if m > score: return diff.index(m), m
			return default, score

	class Predictor:
		"""The basic iocaine second- and triple-guesser.    Maintains stats on the
			past benefits of trusting or second- or triple-guessing a given strategy,
			and returns the prediction of that strategy (or the second- or triple-
			guess) if past stats are deviating from zero farther than the supplied
			"best" guess so far."""
		def __init__(self):
			self.stats = Iocaine.Stats()
			self.lastguess = -1

		def beat(self, i):
			return (i + 1) % 3
		def loseto(self, i):
			return (i - 1) % 3

		def addguess(self, lastmove, guess):
			if lastmove != -1:
				diff = (lastmove - self.prediction) % 3
				self.stats.add(self.beat(diff), 1)
				self.stats.add(self.loseto(diff), -1)
				self.stats.advance()
			self.prediction = guess
		def bestguess(self, age, best):
			bestdiff = self.stats.max(age, (best[0] - self.prediction) % 3, best[1])
			return (bestdiff[0] + self.prediction) % 3, bestdiff[1]

	class Iocaine:

		def __init__(self):
			"""Build second-guessers for 50 strategies: 36 history-based strategies,
				12 simple frequency-based strategies, the constant-move strategy, and
				the basic random-number-generator strategy.    Also build 6 meta second
				guessers to evaluate 6 different time horizons on which to score
				the 50 strategies' second-guesses."""
			self.ages = [1000, 100, 10, 5, 2, 1]
			self.predictors = []
			self.predict_history = self.predictor((len(self.ages), 2, 3))
			self.predict_frequency = self.predictor((len(self.ages), 2))
			self.predict_fixed = self.predictor()
			self.predict_random = self.predictor()
			self.predict_meta = [Iocaine.Predictor() for a in range(len(self.ages))]
			self.stats = [Iocaine.Stats() for i in range(2)]
			self.histories = [[], [], []]

		def predictor(self, dims=None):
			"""Returns a nested array of predictor objects, of the given dimensions."""
			if dims: return [self.predictor(dims[1:]) for i in range(dims[0])]
			self.predictors.append(Iocaine.Predictor())
			return self.predictors[-1]

		def move(self, them):
			"""The main iocaine "move" function."""

			# histories[0] stores our moves (last one already previously decided);
			# histories[1] stores their moves (last one just now being supplied to us);
			# histories[2] stores pairs of our and their last moves.
			# stats[0] and stats[1] are running counters our recent moves and theirs.
			if them != -1:
				self.histories[1].append(them)
				self.histories[2].append((self.histories[0][-1], them))
				for watch in range(2):
					self.stats[watch].add(self.histories[watch][-1], 1)

			# Execute the basic RNG strategy and the fixed-move strategy.
			rand = random.randrange(3)
			self.predict_random.addguess(them, rand)
			self.predict_fixed.addguess(them, 0)

			# Execute the history and frequency stratgies.
			for a, age in enumerate(self.ages):
				# For each time window, there are three ways to recall a similar time:
				# (0) by history of my moves; (1) their moves; or (2) pairs of moves.
				# Set "best" to these three timeframes (zero if no matching time).
				best = [Iocaine.recall(age, hist) for hist in self.histories]
				for mimic in range(2):
					# For each similar historical moment, there are two ways to anticipate
					# the future: by mimicing what their move was; or mimicing what my
					# move was.    If there were no similar moments, just move randomly.
					for watch, when in enumerate(best):
						if not when: move = rand
						else: move = self.histories[mimic][when]
						self.predict_history[a][mimic][watch].addguess(them, move)
					# Also we can anticipate the future by expecting it to be the same
					# as the most frequent past (either counting their moves or my moves).
					mostfreq, score = self.stats[mimic].max(age, rand, -1)
					self.predict_frequency[a][mimic].addguess(them, mostfreq)

			# All the predictors have been updated, but we have not yet scored them
			# and chosen a winner for this round.    There are several timeframes
			# on which we can score second-guessing, and we don't know timeframe will
			# do best.    So score all 50 predictors on all 6 timeframes, and record
			# the best 6 predictions in meta predictors, one for each timeframe.
			for meta, age in enumerate(self.ages):
				best = (-1, -1)
				for predictor in self.predictors:
					best = predictor.bestguess(age, best)
				self.predict_meta[meta].addguess(them, best[0])

			# Finally choose the best meta prediction from the final six, scoring
			# these against each other on the whole-game timeframe. 
			best = (-1, -1)
			for meta in range(len(self.ages)):
				best = self.predict_meta[meta].bestguess(len(self.histories[0]) , best) 

			# We've picked a next move.    Record our move in histories[0] for next time.
			self.histories[0].append(best[0])

			# And return it.
			return best[0]

	def initial_step(self, observation, configuration):
		self.iocaine = self.Iocaine()
		return self.iocaine.move(-1)

	def step(self, observation, configuration):
		return self.iocaine.move(observation.lastOpponentAction)
	
	def set_last_action(self, action):
		if self.iocaine != None:
			self.iocaine.histories[0][-1] = action


# IO2_fightinguuu Agent From RPSContest: https://web.archive.org/web/20200812062252/http://www.rpscontest.com/entry/885001
class IO2(Agent):
	def __init__(self):
		self.num_predictor = 27
		self.len_rfind = [20]
		self.limit = [10,20,60]
		self.beat = { "R":"P" , "P":"S", "S":"R"}
		self.not_lose = { "R":"PPR" , "P":"SSP" , "S":"RRS" } #50-50 chance
		self.my_his   =""
		self.your_his =""
		self.both_his =""
		self.list_predictor = [""]*self.num_predictor
		self.length = 0
		self.temp1 = { "PP":"1" , "PR":"2" , "PS":"3",
				"RP":"4" , "RR":"5", "RS":"6",
				"SP":"7" , "SR":"8", "SS":"9"}
		self.temp2 = { "1":"PP","2":"PR","3":"PS",
					"4":"RP","5":"RR","6":"RS",
					"7":"SP","8":"SR","9":"SS"} 
		self.who_win = { "PP": 0, "PR":1 , "PS":-1,
					"RP": -1,"RR":0, "RS":1,
					"SP": 1, "SR":-1, "SS":0}
		self.score_predictor = [0]*self.num_predictor
		self.output = random.choice("RPS")
		self.predictors = [self.output]*self.num_predictor
	
	def initial_step(self, obs, config):
		return 'RPS'.index(self.output)
	
	def step(self, obs, config):
		input = 'RPS'[obs.lastOpponentAction]
		#update predictors
		if len(self.list_predictor[0])<5:
			front =0
		else:
			front =1
		for i in range (self.num_predictor):
			if self.predictors[i]==input:
				result ="1"
			else:
				result ="0"
			self.list_predictor[i] = self.list_predictor[i][front:5]+result #only 5 rounds before
		#history matching 1-6
		self.my_his += self.output
		self.your_his += input
		self.both_his += self.temp1[input+self.output]
		self.length +=1
		for i in range(1):
			len_size = min(self.length,self.len_rfind[i])
			j=len_size
			#both_his
			while j>=1 and not self.both_his[self.length-j:self.length] in self.both_his[0:self.length-1]:
				j-=1
			if j>=1:
				k = self.both_his.rfind(self.both_his[self.length-j:self.length],0,self.length-1)
				self.predictors[0+6*i] = self.your_his[j+k]
				self.predictors[1+6*i] = self.beat[self.my_his[j+k]]
			else:
				self.predictors[0+6*i] = random.choice("RPS")
				self.predictors[1+6*i] = random.choice("RPS")
			j=len_size
			#your_his
			while j>=1 and not self.your_his[self.length-j:self.length] in self.your_his[0:self.length-1]:
				j-=1
			if j>=1:
				k = self.your_his.rfind(self.your_his[self.length-j:self.length],0,self.length-1)
				self.predictors[2+6*i] = self.your_his[j+k]
				self.predictors[3+6*i] = self.beat[self.my_his[j+k]]
			else:
				self.predictors[2+6*i] = random.choice("RPS")
				self.predictors[3+6*i] = random.choice("RPS")
			j=len_size
			# my history
			while j>=1 and not self.my_his[self.length-j:self.length] in self.my_his[0:self.length-1]:
				j-=1
			if j>=1:
				k = self.my_his.rfind(self.my_his[self.length-j:self.length],0,self.length-1)
				self.predictors[4+6*i] = self.your_his[j+k]
				self.predictors[5+6*i] = self.beat[self.my_his[j+k]]
			else:
				self.predictors[4+6*i] = random.choice("RPS")
				self.predictors[5+6*i] = random.choice("RPS")

		for i in range(3):
			temp =""
			search = self.temp1[(self.output+input)] #last round
			for start in range(2, min(self.limit[i],self.length) ):
				if search == self.both_his[self.length-start]:
					temp+=self.both_his[self.length-start+1]
			if(temp==""):
				self.predictors[6+i] = random.choice("RPS")
			else:
				collectR = {"P":0,"R":0,"S":0} #take win/lose from opponent into account
				for sdf in temp:
					next_move = self.temp2[sdf]
					if(self.who_win[next_move]==-1):
						collectR[self.temp2[sdf][1]]+=3
					elif(self.who_win[next_move]==0):
						collectR[self.temp2[sdf][1]]+=1
					elif(self.who_win[next_move]==1):
						collectR[self.beat[self.temp2[sdf][0]]]+=1
				max1 = -1
				p1 =""
				for key in collectR:
					if(collectR[key]>max1):
						max1 = collectR[key]
						p1 += key
				self.predictors[6+i] = random.choice(p1)
		
		#rotate 9-27:
		for i in range(9,27):
			self.predictors[i] = self.beat[self.beat[self.predictors[i-9]]]
			
		#choose a predictor
		len_his = len(self.list_predictor[0])
		for i in range(self.num_predictor):
			sum = 0
			for j in range(len_his):
				if self.list_predictor[i][j]=="1":
					sum+=(j+1)*(j+1)
				else:
					sum-=(j+1)*(j+1)
			self.score_predictor[i] = sum
		max_score = max(self.score_predictor)
		if max_score>0:
			predict = self.predictors[self.score_predictor.index(max_score)]
		else:
			predict = random.choice(self.your_his)
		self.output = random.choice(self.not_lose[predict])
		# self.output = self.beat[predict]
		return 'RPS'.index(self.output)

	def set_last_action(self, action):
		self.output = 'RPS'[action]


# Testing Please Ignore Agent from RPSContest: https://web.archive.org/web/20201021153705/http://rpscontest.com/entry/342001
class TestingPleaseIgnore(Agent):

	def __init__(self):

		self.score  = {'RR': 0, 'PP': 0, 'SS': 0, \
				'PR': 1, 'RS': 1, 'SP': 1, \
				'RP': -1, 'SR': -1, 'PS': -1,}
		self.cscore = {'RR': 'r', 'PP': 'r', 'SS': 'r', \
				'PR': 'b', 'RS': 'b', 'SP': 'b', \
				'RP': 'c', 'SR': 'c', 'PS': 'c',}
		self.beat = {'P': 'S', 'S': 'R', 'R': 'P'}
		self.cede = {'P': 'R', 'S': 'P', 'R': 'S'}
		self.rps = ['R', 'P', 'S']
		self.wlt = {1: 0, -1: 1, 0: 2}

		self.played_probs = defaultdict(lambda: 1)
		self.dna_probs = [
			defaultdict(lambda: defaultdict(lambda: 1)) for i in range(18)
		]

		self.wlt_probs = [defaultdict(lambda: 1) for i in range(9)]
		self.answers = [{'c': 1, 'b': 1, 'r': 1} for i in range(12)]
		self.patterndict = [defaultdict(str) for i in range(6)]

		self.consec_strat_usage = [[0] * 6, [0] * 6, [0] * 6]  #consecutive strategy usage
		self.consec_strat_candy = [[], [], []]  #consecutive strategy candidates

		self.output = random.choice(self.rps)
		self.histories = ["", "", ""]
		self.dna = ["" for i in range(12)]

		self.sc = 0
		self.strats = [[] for i in range(3)]
	
	def counter_prob(self, probs):
		weighted_list = []
		for h in self.rps:
			weighted = 0
			for p in probs.keys():
				points = self.score[h + p]
				prob = probs[p]
				weighted += points * prob
			weighted_list.append((h, weighted))

		return max(weighted_list, key=operator.itemgetter(1))[0]

	def initial_step(self, obs, config):
		return 'RPS'.index(self.output)

	def step(self, obs, config):

		input = 'RPS'[obs.lastOpponentAction]

		self.prev_sc = self.sc
		self.sc = self.score[self.output + input]
		for j in range(3):
			prev_strats = self.strats[j][:]
			for i, c in enumerate(self.consec_strat_candy[j]):
				if c == input:
					self.consec_strat_usage[j][i] += 1
				else:
					self.consec_strat_usage[j][i] = 0
			m = max(self.consec_strat_usage[j])
			self.strats[j] = [
				i for i, c in enumerate(self.consec_strat_candy[j])
				if self.consec_strat_usage[j][i] == m
			]

			for s1 in prev_strats:
				for s2 in self.strats[j]:
					self.wlt_probs[j * 3 + self.wlt[self.prev_sc]][chr(s1) + chr(s2)] += 1

			if self.dna[2 * j + 0] and self.dna[2 * j + 1]:
				self.answers[2 * j + 0][self.cscore[input + self.dna[2 * j + 0]]] += 1
				self.answers[2 * j + 1][self.cscore[input + self.dna[2 * j + 1]]] += 1
			if self.dna[2 * j + 6] and self.dna[2 * j + 7]:
				self.answers[2 * j + 6][self.cscore[input + self.dna[2 * j + 6]]] += 1
				self.answers[2 * j + 7][self.cscore[input + self.dna[2 * j + 7]]] += 1

			for length in range(min(10, len(self.histories[j])), 0, -2):
				pattern = self.patterndict[2 * j][self.histories[j][-length:]]
				if pattern:
					for length2 in range(min(10, len(pattern)), 0, -2):
						self.patterndict[2 * j +
									1][pattern[-length2:]] += self.output + input
				self.patterndict[2 * j][self.histories[j][-length:]] += self.output + input
		self.played_probs[input] += 1
		self.dna_probs[0][self.dna[0]][input] += 1
		self.dna_probs[1][self.dna[1]][input] += 1
		self.dna_probs[2][self.dna[1] + self.dna[0]][input] += 1
		self.dna_probs[9][self.dna[6]][input] += 1
		self.dna_probs[10][self.dna[6]][input] += 1
		self.dna_probs[11][self.dna[7] + self.dna[6]][input] += 1

		self.histories[0] += self.output + input
		self.histories[1] += input
		self.histories[2] += self.output

		self.dna = ["" for i in range(12)]
		for j in range(3):
			for length in range(min(10, len(self.histories[j])), 0, -2):
				pattern = self.patterndict[2 * j][self.histories[j][-length:]]
				if pattern != "":
					self.dna[2 * j + 1] = pattern[-2]
					self.dna[2 * j + 0] = pattern[-1]
					for length2 in range(min(10, len(pattern)), 0, -2):
						pattern2 = self.patterndict[2 * j + 1][pattern[-length2:]]
						if pattern2 != "":
							self.dna[2 * j + 7] = pattern2[-2]
							self.dna[2 * j + 6] = pattern2[-1]
							break
					break

		probs = {}
		for hand in self.rps:
			probs[hand] = self.played_probs[hand]

		for j in range(3):
			if self.dna[j * 2] and self.dna[j * 2 + 1]:
				for hand in self.rps:
					probs[hand] *= self.dna_probs[j*3+0][self.dna[j*2+0]][hand] * \
								self.dna_probs[j*3+1][self.dna[j*2+1]][hand] * \
						self.dna_probs[j*3+2][self.dna[j*2+1]+self.dna[j*2+0]][hand]
					probs[hand] *= self.answers[j*2+0][self.cscore[hand+self.dna[j*2+0]]] * \
								self.answers[j*2+1][self.cscore[hand+self.dna[j*2+1]]]
				self.consec_strat_candy[j] = [self.dna[j*2+0], self.beat[self.dna[j*2+0]], self.cede[self.dna[j*2+0]],\
										self.dna[j*2+1], self.beat[self.dna[j*2+1]], self.cede[self.dna[j*2+1]]]
				strats_for_hand = {'R': [], 'P': [], 'S': []}
				for i, c in enumerate(self.consec_strat_candy[j]):
					strats_for_hand[c].append(i)
				pr = self.wlt_probs[self.wlt[self.sc] + 3 * j]
				for hand in self.rps:
					for s1 in self.strats[j]:
						for s2 in strats_for_hand[hand]:
							probs[hand] *= pr[chr(s1) + chr(s2)]
			else:
				self.consec_strat_candy[j] = []
		for j in range(3):
			if self.dna[j * 2 + 6] and self.dna[j * 2 + 7]:
				for hand in self.rps:
					probs[hand] *= self.dna_probs[j*3+9][self.dna[j*2+6]][hand] * \
								self.dna_probs[j*3+10][self.dna[j*2+7]][hand] * \
						self.dna_probs[j*3+11][self.dna[j*2+7]+self.dna[j*2+6]][hand]
					probs[hand] *= self.answers[j*2+6][self.cscore[hand+self.dna[j*2+6]]] * \
								self.answers[j*2+7][self.cscore[hand+self.dna[j*2+7]]]

		self.output = self.counter_prob(probs)
		return 'RPS'.index(self.output)
		
	def set_last_action(self, action):
		self.output = 'RPS'[action]


# Dllu1 Agent from RPSContest: https://web.archive.org/web/20200812060710/http://www.rpscontest.com/entry/498002
class Dllu1(Agent):

	def __init__(self):
		self.numPre = 30
		self.numMeta = 6
		self.limit = 8
		self.beat={'R':'P','P':'S','S':'R'}
		self.moves=['','','','']
		self.pScore=[[5]*self.numPre] * 6
		self.centrifuge={'RP':0,'PS':1,'SR':2,'PR':3,'SP':4,'RS':5,'RR':6,'PP':7,'SS':8}
		self.centripete={'R':0,'P':1,'S':2}
		self.soma = [0,0,0,0,0,0,0,0,0]
		self.rps = [1,1,1]
		self.a="RPS"
		self.best = [0,0,0]
		self.length=0
		self.p=[random.choice("RPS")]*self.numPre
		self.m=[random.choice("RPS")]*self.numMeta
		self.mScore=[5,2,5,2,4,2]

	def initial_step(self, obs, config):
		return self.move()

	def step(self, obs, config):
		input = 'RPS'[obs.lastOpponentAction]
		for i in range(self.numPre):
			pp = self.p[i]
			bpp = self.beat[pp]
			bbpp = self.beat[bpp]
			self.pScore[0][i]=0.9*self.pScore[0][i]+((input==pp)-(input==bbpp))*3
			self.pScore[1][i]=0.9*self.pScore[1][i]+((self.output==pp)-(self.output==bbpp))*3
			self.pScore[2][i]=0.87*self.pScore[2][i]+(input==pp)*3.3-(input==bpp)*1.2-(input==bbpp)*2.3
			self.pScore[3][i]=0.87*self.pScore[3][i]+(self.output==pp)*3.3-(self.output==bpp)*1.2-(self.output==bbpp)*2.3
			self.pScore[4][i]=(self.pScore[4][i]+(input==pp)*3)*(1-(input==bbpp))
			self.pScore[5][i]=(self.pScore[5][i]+(self.output==pp)*3)*(1-(self.output==bbpp))
		for i in range(self.numMeta):
			self.mScore[i]=0.96*(self.mScore[i]+(input==self.m[i])-(input==self.beat[self.beat[self.m[i]]]))
		self.soma[self.centrifuge[input+self.output]] +=1
		self.rps[self.centripete[input]] +=1
		self.moves[0]+=str(self.centrifuge[input+self.output])
		self.moves[1]+=input
		self.moves[2]+=self.output
		self.length+=1
		for y in range(3):
			j=min([self.length,self.limit])
			while j>=1 and not self.moves[y][self.length-j:self.length] in self.moves[y][0:self.length-1]:
				j-=1
			i = self.moves[y].rfind(self.moves[y][self.length-j:self.length],0,self.length-1)
			self.p[0+2*y] = self.moves[1][j+i] 
			self.p[1+2*y] = self.beat[self.moves[2][j+i]]
		j=min([self.length,self.limit])
		while j>=2 and not self.moves[0][self.length-j:self.length-1] in self.moves[0][0:self.length-2]:
			j-=1
		i = self.moves[0].rfind(self.moves[0][self.length-j:self.length-1],0,self.length-2)
		if j+i>=self.length:
			self.p[6] = self.p[7] = random.choice("RPS")
		else:
			self.p[6] = self.moves[1][j+i] 
			self.p[7] = self.beat[self.moves[2][j+i]]
			
		self.best[0] = self.soma[self.centrifuge[self.output+'R']]*self.rps[0]/self.rps[self.centripete[self.output]]
		self.best[1] = self.soma[self.centrifuge[self.output+'P']]*self.rps[1]/self.rps[self.centripete[self.output]]
		self.best[2] = self.soma[self.centrifuge[self.output+'S']]*self.rps[2]/self.rps[self.centripete[self.output]]
		self.p[8] = self.p[9] = self.a[self.best.index(max(self.best))]
		
		for i in range(10,self.numPre):
			self.p[i]=self.beat[self.beat[self.p[i-10]]]
			
		for i in range(0,self.numMeta,2):
			self.m[i]= self.p[self.pScore[i].index(max(self.pScore[i]))]
			self.m[i+1]=self.beat[self.p[self.pScore[i+1].index(max(self.pScore[i+1]))]]
	
		return self.move()
		
	def move(self):
		self.output = self.beat[self.m[self.mScore.index(max(self.mScore))]]
		if max(self.mScore)<3+random.random() or random.randint(3,40)>self.length:# or random.random() < 0.5:
			self.output=self.beat[random.choice("RPS")]
		return 'RPS'.index(self.output)
	
	def set_last_action(self, action):
		self.output = 'RPS'[action]


# Centrifugal Bumblepuppy: https://web.archive.org/web/20201021155550/http://rpscontest.com/entry/315005
class Bumble(Agent):

	def __init__(self):
		self.numPre = 54
		self.numMeta = 24
		self.limits = [50,20,10]
		self.beat={'R':'P','P':'S','S':'R'}
		self.moves=['','','']
		self.pScore=[[3]*self.numPre] * 8
		self.centrifuge={'RP':'a','PS':'b','SR':'c','PR':'d','SP':'e','RS':'f','RR':'g','PP':'h','SS':'i'}
		self.length=0
		self.p=[random.choice("RPS")]*self.numPre
		self.m=[random.choice("RPS")]*self.numMeta
		self.mScore=[3]*self.numMeta
		self.threat = [0,0,0]
		self.outcome = 0
		
	def initial_step(self, obs, config):
		return self.move()

	def step(self, obs, config):
		input = 'RPS'[obs.lastOpponentAction]
		self.oldoutcome = self.outcome
		self.outcome = (self.beat[input]==self.output2) - (input==self.beat[self.output2])
		self.threat[self.oldoutcome + 1] *= 0.957
		self.threat[self.oldoutcome + 1] -= 0.042*self.outcome
		for i in range(self.numPre):
			self.pScore[0][i]=0.8*self.pScore[0][i]+((input==self.p[i])-(input==self.beat[self.beat[self.p[i]]]))*3
			self.pScore[1][i]=0.8*self.pScore[1][i]+((self.output==self.p[i])-(self.output==self.beat[self.beat[self.p[i]]]))*3
			self.pScore[2][i]=0.87*self.pScore[2][i]+(input==self.p[i])*3.3-(input==self.beat[self.p[i]])*0.9-(input==self.beat[self.beat[self.p[i]]])*3
			self.pScore[3][i]=0.87*self.pScore[3][i]+(self.output==self.p[i])*3.3-(self.output==self.beat[self.p[i]])*0.9-(self.output==self.beat[self.beat[self.p[i]]])*3
			self.pScore[4][i]=(self.pScore[4][i]+(input==self.p[i])*3)*(1-(input==self.beat[self.beat[self.p[i]]]))
			self.pScore[5][i]=(self.pScore[5][i]+(self.output==self.p[i])*3)*(1-(self.output==self.beat[self.beat[self.p[i]]]))
			self.pScore[6][i]=(self.pScore[6][i]+(input==self.p[i])*3)*(1-((input==self.beat[self.beat[self.p[i]]]) or (input==self.beat[self.p[i]])))
			self.pScore[7][i]=(self.pScore[7][i]+(self.output==self.p[i])*3)*(1-((self.output==self.beat[self.beat[self.p[i]]]) or (self.output==self.beat[self.p[i]])))
		for i in range(self.numMeta):
			self.mScore[i]=0.94*self.mScore[i]+(input==self.m[i])-(input==self.beat[self.beat[self.m[i]]])
			if input==self.beat[self.beat[self.m[i]]] and random.random()<0.87 or self.mScore[i]<0:
				self.mScore[i]=0
		self.moves[0]+=self.centrifuge[input+self.output]
		self.moves[1]+=input		
		self.moves[2]+=self.output
		self.length+=1
		for z in range(3):
			limit = min([self.length,self.limits[z]])
			for y in range(3):
				j=limit
				while j>=1 and not self.moves[y][self.length-j:self.length] in self.moves[y][0:self.length-1]:
					j-=1
				if j>=1:
					if random.random()<0.6:
						i = self.moves[y].rfind(self.moves[y][self.length-j:self.length],0,self.length-1)
					elif random.random()<0.5:
						i = self.moves[y].rfind(self.moves[y][self.length-j:self.length],0,self.length-1)
						i2 = self.moves[y].rfind(self.moves[y][self.length-j:self.length],0,i)
						if i2!=-1:
							i=i2
					else:
						i = self.moves[y].find(self.moves[y][self.length-j:self.length],0,self.length-1)
					self.p[0+2*y+6*z] = self.moves[1][j+i] 
					self.p[1+2*y+6*z] = self.beat[self.moves[2][j+i]] 
		
		for i in range(18,18*3):
			self.p[i]=self.beat[self.beat[self.p[i-18]]]
			
		for i in range(0,8,2):
			self.m[i]=       self.p[self.pScore[i  ].index(max(self.pScore[i  ]))]
			self.m[i+1]=self.beat[self.p[self.pScore[i+1].index(max(self.pScore[i+1]))]]
		for i in range(8,24):
			self.m[i]=self.beat[self.beat[self.m[i-8]]]
		
		return self.move()
	
	def move(self):
		self.output2 = self.output = self.beat[self.m[self.mScore.index(max(self.mScore))]]
		if random.random()<0.1 or random.randint(3,40)>self.length:
			self.output=self.beat[random.choice("RPS")]
		return 'RPS'.index(self.output)
	
	def set_last_action(self, action):
		self.output = 'RPS'[action]


# Memory Patterns V7 Agent: https://www.kaggle.com/yegorbiryukov/rock-paper-scissors-with-memory-patterns?scriptVersionId=46447097
class MemoryPatterns(Agent):

	def __init__(self, noise = False):

		# random noise
		self.noise = noise

		# maximum steps in the pattern
		self.steps_max = 6
		# minimum steps in the pattern
		self.steps_min = 3
		# maximum amount of steps until reassessment of effectiveness of current memory patterns
		self.max_steps_until_memory_reassessment = random.randint(80, 120)

		# current memory of the agent
		self.current_memory = []
		# list of 1, 0 and -1 representing win, tie and lost results of the game respectively
		# length is max_steps_until_memory_r_t
		self.results = []
		# current best sum of results
		self.best_sum_of_results = 0
		# memory length of patterns in first group
		# steps_max is multiplied by 2 to consider both my_agent's and opponent's actions
		self.group_memory_length = self.steps_max * 2
		# list of groups of memory patterns
		self.groups_of_memory_patterns = []
		for i in range(self.steps_max, self.steps_min - 1, -1):
			self.groups_of_memory_patterns.append({
				# how many steps in a row are in the pattern
				"memory_length": self.group_memory_length,
				# list of memory patterns
				"memory_patterns": []
			})
			self.group_memory_length -= 2

	def find_pattern(self, memory_patterns, memory, memory_length):
		""" find appropriate pattern in memory """
		for pattern in memory_patterns:
			actions_matched = 0
			for i in range(memory_length):
				if pattern["actions"][i] == memory[i]:
					actions_matched += 1
				else:
					break
			# if memory fits this pattern
			if actions_matched == memory_length:
				return pattern
		# appropriate pattern not found
		return None

	def get_step_result_for_my_agent(self, my_agent_action, opp_action):
		""" 
			get result of the step for my_agent
			1, 0 and -1 representing win, tie and lost results of the game respectively
		"""
		if my_agent_action == opp_action: return 0
		elif (my_agent_action == (opp_action + 1) % 3): return 1
		else: return -1

	def step(self, obs, config):
		# action of my_agent
		my_action = None
		# if it's not first step, add opponent's last action to agent's current memory
		# and reassess effectiveness of current memory patterns
		if obs.step > 0:
			self.current_memory.append(obs.lastOpponentAction)
			# previous step won or lost
			self.results.append(self.get_step_result_for_my_agent(self.current_memory[-2], self.current_memory[-1]))
			# if there is enough steps added to results for memery reassessment
			if len(self.results) == self.max_steps_until_memory_reassessment:
				results_sum = sum(self.results)
				# if effectiveness of current memory patterns has decreased significantly
				if results_sum < (self.best_sum_of_results * 0.5):
					# flush all current memory patterns
					self.best_sum_of_results = 0
					self.results = []
					for group in self.groups_of_memory_patterns:
						group["memory_patterns"] = []
				else:
					# if effectiveness of current memory patterns has increased
					if results_sum > self.best_sum_of_results:
						self.best_sum_of_results = results_sum
					del self.results[:1]

		for group in self.groups_of_memory_patterns:
			# if length of current memory is bigger than necessary for a new memory pattern
			if len(self.current_memory) > group["memory_length"]:
				# get momory of the previous step
				previous_step_memory = self.current_memory[:group["memory_length"]]
				previous_pattern = self.find_pattern(group["memory_patterns"], previous_step_memory, group["memory_length"])
				if previous_pattern == None:
					previous_pattern = {
						"actions": previous_step_memory.copy(),
						"opp_next_actions": [
							{"action": 0, "amount": 0, "response": 1},
							{"action": 1, "amount": 0, "response": 2},
							{"action": 2, "amount": 0, "response": 0}
						]
					}
					group["memory_patterns"].append(previous_pattern)
				# if such pattern already exists
				for action in previous_pattern["opp_next_actions"]:
					if action["action"] == obs.lastOpponentAction:
						action["amount"] += 1
				# delete first two elements in current memory (actions of the oldest step in current memory)
				del self.current_memory[:2]
				# if action was not yet found
				if my_action == None:
					pattern = self.find_pattern(group["memory_patterns"], self.current_memory, group["memory_length"])
					# if appropriate pattern is found
					if pattern != None:
						my_action_amount = 0
						for action in pattern["opp_next_actions"]:
							# if this opponent's action occurred more times than currently chosen action
							# or, if it occured the same amount of times and this one is choosen randomly among them
							if (action["amount"] > my_action_amount or
									(action["amount"] == my_action_amount and random.random() > 0.5)):
								my_action_amount = action["amount"]
								my_action = action["response"]
		# if no action was found
		if my_action == None or (self.noise and random.random() < 0.2):
			my_action = random.randint(0, 2)
		self.current_memory.append(my_action)
		return my_action
	
	def set_last_action(self, action):
		self.current_memory[-1] = action


# RPS_Meta_Fix Agent: https://web.archive.org/web/20200220023240/http://www.rpscontest.com/entry/5649874456412160
class MetaFix(Agent):
	def __init__(self):
		self.RNA={'RR':'1','RP':'2','RS':'3','PR':'4','PP':'5','PS':'6','SR':'7','SP':'8','SS':'9'}
		self.mix={'RR':'R','RP':'R','RS':'S','PR':'R','PP':'P','PS':'P','SR':'S','SP':'P','SS':'S'}
		self.rot={'R':'P','P':'S','S':'R'}

		self.DNA=[""]*3
		self.prin=[random.choice("RPS")]*18
		self.meta=[random.choice("RPS")]*6
		self.skor1=[[0]*18,[0]*18,[0]*18,[0]*18,[0]*18,[0]*18]
		self.skor2=[0]*6

		self.output = random.choice('RPS')
	
	def initial_step(self, obs, config):
		self.output=self.rot[self.meta[self.skor2.index(max(self.skor2))]]
		return 'RPS'.index(self.output)

	def step(self, obs, config):
		input = 'RPS'[obs.lastOpponentAction]
		for j in range(18):
			for i in range(4):
				self.skor1[i][j]*=0.8
			for i in range(4,6):
				self.skor1[i][j]*=0.5
			for i in range(0,6,2):
				self.skor1[i][j]-=(input==self.rot[self.rot[self.prin[j]]])
				self.skor1[i+1][j]-=(self.output==self.rot[self.rot[self.prin[j]]])
			for i in range(2,6,2):
				self.skor1[i][j]+=(input==self.prin[j])
				self.skor1[i+1][j]+=(self.output==self.prin[j])
			self.skor1[0][j]+=1.3*(input==self.prin[j])-0.3*(input==self.rot[self.prin[j]])
			self.skor1[1][j]+=1.3*(self.output==self.prin[j])-0.3*(self.output==self.rot[self.prin[j]])
		for i in range(6):
			self.skor2[i]=0.9*self.skor2[i]+(input==self.meta[i])-(input==self.rot[self.rot[self.meta[i]]])
		self.DNA[0]+=input
		self.DNA[1]+=self.output
		self.DNA[2]+=self.RNA[input+self.output]
		for i in range(3):
			j=min(21,len(self.DNA[2]))
			k=-1
			while j>1 and k<0:
				j-=1
				k=self.DNA[i].rfind(self.DNA[i][-j:],0,-1)
			self.prin[2*i]=self.DNA[0][j+k]
			self.prin[2*i+1]=self.rot[self.DNA[1][j+k]]
			k=self.DNA[i].rfind(self.DNA[i][-j:],0,j+k-1)
			self.prin[2*i]=self.mix[self.prin[2*i]+self.DNA[0][j+k]]
			self.prin[2*i+1]=self.mix[self.prin[2*i+1]+self.rot[self.DNA[1][j+k]]]
		for i in range(6,18):
			self.prin[i]=self.rot[self.prin[i-6]]
		for i in range(0,6,2):
			self.meta[i]=self.prin[self.skor1[i].index(max(self.skor1[i]))]
			self.meta[i+1]=self.rot[self.prin[self.skor1[i+1].index(max(self.skor1[i+1]))]]
		self.output=self.rot[self.meta[self.skor2.index(max(self.skor2))]]
		return 'RPS'.index(self.output)
	
	def set_last_action(self, action):
		self.output = 'RPS'[action]


# RFind Agent: https://www.kaggle.com/riccardosanson/rps-simple-rfind-agent
class RFind(Agent):
	
	def __init__(self):
		self.max_limit = 23  # can be modified
		self.add_rotations = True

		# number of predictors
		self.numPre = 6
		if self.add_rotations:
			self.numPre *= 3

		# number of meta-predictors
		self.numMeta = 4
		if self.add_rotations:
			self.numMeta *= 3

		# saves history
		self.moves = ['', '', '']

		self.beat = {'R':'P', 'P':'S', 'S':'R'}
		self.dna =  {'RP':0, 'PS':1, 'SR':2,
				'PR':3, 'SP':4, 'RS':5,
				'RR':6, 'PP':7, 'SS':8}

		self.p = ["P"]*self.numPre
		self.m = ["P"]*self.numMeta
		self.pScore = [[0]*self.numPre for i in range(8)]
		self.mScore = [0]*self.numMeta

		self.length = 0
		self.threat = 0
		self.output = "P"
	
	def step(self, observation, configuration):    

		if observation.step < 2:
			self.output = self.beat[self.output]
			return 'RPS'.index(self.output)

		input = "RPS"[observation.lastOpponentAction]

		# threat of opponent
		outcome = (self.beat[input]==self.output) - (input==self.beat[self.output])
		self.threat = 0.9*self.threat - 0.1*outcome
		
		# refresh pScore
		for i in range(self.numPre):
			pp = self.p[i]
			bpp = self.beat[pp]
			bbpp = self.beat[bpp]
			self.pScore[0][i] = 0.9*self.pScore[0][i] + 0.1*((input==pp)-(input==bbpp))
			self.pScore[1][i] = 0.9*self.pScore[1][i] + 0.1*((self.output==pp)-(self.output==bbpp))
			self.pScore[2][i] = 0.8*self.pScore[2][i] + 0.3*((input==pp)-(input==bbpp)) + \
							0.1*(self.length % 3 - 1)
			self.pScore[3][i] = 0.8*self.pScore[3][i] + 0.3*((self.output==pp)-(self.output==bbpp)) + \
							0.1*(self.length % 3 - 1)

		# refresh mScore
		for i in range(self.numMeta):
			self.mScore[i] = 0.9*self.mScore[i] + 0.1*((input==self.m[i])-(input==self.beat[self.beat[self.m[i]]])) + \
						0.05*(self.length % 5 - 2)

		# refresh moves
		self.moves[0] += str(self.dna[input+self.output])
		self.moves[1] += input
		self.moves[2] += self.output

		# refresh length
		self.length += 1

		# new predictors
		limit = min([self.length,self.max_limit])
		for y in range(3):	# my moves, his, and both
			j = limit
			while j>=1 and not self.moves[y][self.length-j:self.length] in self.moves[y][0:self.length-1]:
				j-=1
			if j>=1:
				i = self.moves[y].rfind(self.moves[y][self.length-j:self.length],0,self.length-1)
				self.p[0+2*y] = self.moves[1][j+i] 
				self.p[1+2*y] = self.beat[self.moves[2][j+i]]

		# rotations of predictors
		if self.add_rotations:
			for i in range(int(self.numPre/3),self.numPre):
				self.p[i]=self.beat[self.beat[self.p[i-int(self.numPre/3)]]]

		# new meta
		for i in range(0,4,2):
			self.m[i] = self.p[self.pScore[i].index(max(self.pScore[i]))]
			self.m[i+1] = self.beat[self.p[self.pScore[i+1].index(max(self.pScore[i+1]))]]

		# rotations of meta
		if self.add_rotations:
			for i in range(4,12):
				self.m[i]=self.beat[self.beat[self.m[i-4]]]
		
		self.output = self.beat[self.m[self.mScore.index(max(self.mScore))]]
		if self.threat > 0.4:
			self.output = self.beat[self.beat[self.output]]

		return 'RPS'.index(self.output)
	
	def set_last_action(self, action):
		self.output = 'RPS'[action]


# Lucker Agent from RPSContest: https://web.archive.org/web/20191201105926/http://www.rpscontest.com/entry/892001
class Lucker(Agent):
	def __init__(self):
		self.num_predictors =27
		self.num_meta= 18
		self.len_rfind = [20]
		self.limit = [10,20,60]
		self.beat = { "P":"S" , "R":"P" , "S":"R" }
		self.not_lose = { "R":"PR", "P":"SP", "S":"RS" } 
		self.your_his =""
		self.my_his = ""
		self.both_his=""
		self.both_his2=""
		self.length =0
		self.score1=[3]*self.num_predictors
		self.score2=[3]*self.num_predictors
		self.score3=[3]*self.num_predictors
		self.score4=[3]*self.num_predictors
		self.score5=[3]*self.num_predictors
		self.score6=[3]*self.num_predictors
		self.metascore=[3]*self.num_meta
		self.temp1 = { "PP":"1","PR":"2","PS":"3",
				"RP":"4","RR":"5","RS":"6",
				"SP":"7","SR":"8","SS":"9"}
		self.temp2 = { "1":"PP","2":"PR","3":"PS",
					"4":"RP","5":"RR","6":"RS",
					"7":"SP","8":"SR","9":"SS"} 
		self.who_win = { "PP": 0, "PR":1 , "PS":-1,
					"RP": -1,"RR":0, "RS":1,
					"SP": 1, "SR":-1, "SS":0}
		self.index = { "P":0, "R":1, "S":2 }
		self.chance =[0]*self.num_predictors
		self.chance2 =[0]*self.num_predictors
		self.output = random.choice("RPS")
		self.predictors = [self.output]*self.num_predictors
		self.metapredictors = [self.output]*self.num_meta

	def initial_step(self, obs, config):
		return 'RPS'.index(self.output)

	def step(self, obs, config):
		input = 'RPS'[obs.lastOpponentAction]
		#calculate score
		for i in range(self.num_predictors):
			#meta 1
			self.score1[i]*=0.8
			if input==self.predictors[i]:
				self.score1[i]+=3
			else:
				self.score1[i]-=3
			#meta 2
			if input==self.predictors[i]:
				self.score2[i]+=3
			else:
				self.score2[i]=0
			#meta 3
			self.score3[i]*=0.8
			if self.output==self.predictors[i]:
				self.score3[i]+=3
			else:
				self.score3[i]-=3
			#meta 4
			if self.output==self.predictors[i]:
				self.score4[i]+=3
			else:
				self.score4[i]=0
			#meta 5
			self.score5[i]*=0.8
			if input==self.predictors[i]:
				self.score5[i]+=3
			else:
				if self.chance[i]==1:
					self.chance[i]=0
					self.score5[i]-=3
				else:
					self.chance[i]=1
					self.score5[i]=0
			#meta 6
			self.score6[i]*=0.8
			if self.output==self.predictors[i]:
				self.score6[i]+=3
			else:
				if self.chance2[i]==1:
					self.chance2[i]=0
					self.score6[i]-=3
				else:
					self.chance2[i]=1
					self.score6[i]=0
		#calculate metascore
		for i in range(self.num_meta):
			self.metascore[i]*=0.9
			if input==self.metapredictors[i]:
				self.metascore[i]+=3
			else:
				self.metascore[i]=0
		#Predictors
		#if length>1:
		#    output=beat[predict]
		self.your_his+=input
		self.my_his+=self.output
		self.both_his+=self.temp1[(input+self.output)]
		self.both_his2+=self.temp1[(self.output+input)]
		self.length+=1

		#history matching
		for i in range(1):
			len_size = min(self.length,self.len_rfind[i])
			j=len_size
			#both_his
			while j>=1 and not self.both_his[self.length-j:self.length] in self.both_his[0:self.length-1]:
				j-=1
			if j>=1:
				k = self.both_his.rfind(self.both_his[self.length-j:self.length],0,self.length-1)
				self.predictors[0+6*i] = self.your_his[j+k]
				self.predictors[1+6*i] = self.beat[self.my_his[j+k]]
			else:
				self.predictors[0+6*i] = random.choice("RPS")
				self.predictors[1+6*i] = random.choice("RPS")
			j=len_size
			#your_his
			while j>=1 and not self.your_his[self.length-j:self.length] in self.your_his[0:self.length-1]:
				j-=1
			if j>=1:
				k = self.your_his.rfind(self.your_his[self.length-j:self.length],0,self.length-1)
				self.predictors[2+6*i] = self.your_his[j+k]
				self.predictors[3+6*i] = self.beat[self.my_his[j+k]]
			else:
				self.predictors[2+6*i] = random.choice("RPS")
				self.predictors[3+6*i] = random.choice("RPS")
			j=len_size
			#my_his
			while j>=1 and not self.my_his[self.length-j:self.length] in self.my_his[0:self.length-1]:
				j-=1
			if j>=1:
				k = self.my_his.rfind(self.my_his[self.length-j:self.length],0,self.length-1)
				self.predictors[4+6*i] = self.your_his[j+k]
				self.predictors[5+6*i] = self.beat[self.my_his[j+k]]
			else:
				self.predictors[4+6*i] = random.choice("RPS")
				self.predictors[5+6*i] = random.choice("RPS")
		
		#Reverse
		for i in range(3):
			temp =""
			search = self.temp1[(self.output+input)] #last round
			for start in range(2, min(self.limit[i],self.length)):
				if search == self.both_his2[self.length-start]:
					temp+=self.both_his2[self.length-start+1]
			if(temp==""):
				self.predictors[6+i] = random.choice("RPS")
			else:
				collectR = {"P":0,"R":0,"S":0} #take win/lose from opponent into account
				for sdf in temp:
					next_move = self.temp2[sdf]
					if(self.who_win[next_move]==-1):
						collectR[self.temp2[sdf][1]]+=3
					elif(self.who_win[next_move]==0):
						collectR[self.temp2[sdf][1]]+=1
					elif(self.who_win[next_move]==1):
						collectR[self.beat[self.temp2[sdf][0]]]+=1
				max1 = -1
				p1 =""
				for key in collectR:
					if(collectR[key]>max1):
						max1 = collectR[key]
						p1 += key
				self.predictors[6+i] = random.choice(p1)
		
		for i in range(9,27):
			self.predictors[i]=self.beat[self.beat[self.predictors[i-9]]]
		#find prediction for each meta
		self.metapredictors[0]=self.predictors[self.score1.index(max(self.score1))]
		self.metapredictors[1]=self.predictors[self.score2.index(max(self.score2))]
		self.metapredictors[2]=self.beat[self.predictors[self.score3.index(max(self.score3))]]
		self.metapredictors[3]=self.beat[self.predictors[self.score4.index(max(self.score4))]]
		self.metapredictors[4]=self.predictors[self.score5.index(max(self.score5))]
		self.metapredictors[5]=self.beat[self.predictors[self.score6.index(max(self.score6))]]
		for i in range(6,18):
			self.metapredictors[i] = self.beat[self.metapredictors[i-6]]
		
		predict = self.metapredictors[self.metascore.index(max(self.metascore))]
		# self.output = self.beat[predict]
		self.output = random.choice(self.not_lose[predict])
		return 'RPS'.index(self.output)
	
	def set_last_action(self, action):
		self.output = 'RPS'[action]


AGENTS = {

	'random': Agent(),

	# ------------------------------------

	'dllu1': Dllu1(),
	'IO2': IO2(),

	'meta-fix': MetaFix(),
	'rfind': RFind(),
	'lucker': Lucker(),

	'testing-please-ignore': TestingPleaseIgnore(),
	'centrifugal-bumblepuppy': Bumble(),

	'iocaine': Iocaine(),
	'greenberg': Greenberg(),

	'decision-tree': DecisionTree(),
	'decision-tree-2': DecisionTree2(),
	'memory-patterns': MemoryPatterns(),

	# ------------------------------------

	'inverse-dllu1': Dllu1(),
	'inverse-IO2': IO2(),

	'inverse-meta-fix': MetaFix(),
	'inverse-rfind': RFind(),
	'inverse-lucker': Lucker(),

	'inverse-testing-please-ignore': TestingPleaseIgnore(),
	'inverse-centrifugal-bumblepuppy': Bumble(),

	'inverse-iocaine': Iocaine(),
	'inverse-greenberg': Greenberg(),

	'inverse-decision-tree': DecisionTree(),
	'inverse-decision-tree-2': DecisionTree2(),
	'inverse-memory-patterns': MemoryPatterns()

}


class GreedySelector:

	def __init__(self, nActions):
		self.nActions = nActions
		self.n = np.zeros(nActions, dtype = int) # action counts n(a)
		self.Q = np.zeros(nActions, dtype = float) # value Q(a)
		self.lastAction = None

	def update(self, action, reward):
		''' Update Q action-value given (action, reward) '''
		self.n[action] += 1
		self.Q[action] += (1.0 / self.n[action]) * (reward - self.Q[action])

	def get_action(self):
		# Greedy policy
		self.lastAction = int(np.random.choice(np.flatnonzero(self.Q == self.Q.max())))
		return self.lastAction


class Hydra:

	def __init__(self, config):
		
		self.config = config

		self.agents = list(AGENTS.keys())

		self.previous = []
		self.epsilon = 0.1

		self.best_agent = None
		self.action = 0

		self.mab_selector = GreedySelector(len(self.agents))
	
	def step(self, obs):

		start_time = time.perf_counter()

		# Locally speeding up solved matches
		if LOCAL_MODE:
			if abs(obs.reward) - self.config.tieRewardThreshold > self.config.episodeSteps - obs.step:
				return random.randrange(3)

		self.previous.append({})

		Struct = namedtuple('obs', ['lastOpponentAction', 'reward', 'step'])
		inverse_obs = Struct(
			lastOpponentAction = self.action, 
			reward = -obs.reward, 
			step = obs.step
		)

		for name, agent in AGENTS.items():

			if obs.step > 0:

				prev_step = self.previous[obs.step - 1][name]
				index = self.agents.index(name)

				if prev_step == (obs.lastOpponentAction + 1) % 3:
					self.mab_selector.update(index, 1)
				elif prev_step == (obs.lastOpponentAction - 1) % 3:
					self.mab_selector.update(index, 0)
				else:
					self.mab_selector.update(index, 0.5)
				
				if name[:8] == 'inverse-':
					agent.set_last_action(obs.lastOpponentAction)
				else: agent.set_last_action(self.action)

			if name[:8] == 'inverse-':
				agent_step = (agent.get_action(inverse_obs, self.config) + 1) % 3
			else: agent_step = agent.get_action(obs, self.config)
			
			self.previous[obs.step][name] = agent_step

		self.best_agent = self.agents[self.mab_selector.get_action()]
		self.action = self.previous[obs.step][self.best_agent]
		
		# Override action here --------------
		if random.random() < self.epsilon:
			self.action = random.randrange(3)
			self.best_agent = 'random'

		if PRINT_OUTPUT:
			score = round(self.mab_selector.Q[self.mab_selector.lastAction], 3)
			reward_pad = ' ' * (3 - len(str(obs.reward))); score_pad = ' ' * (5 - len(str(score)))
			elapsed_time = time.perf_counter() - start_time
			print(f'{obs.step} | reward {obs.reward}{reward_pad} | score {score}{score_pad} | {self.best_agent} ')

		return self.action

def AGENT(obs, config):
	global agent
	if obs.step == 0:
		agent = Hydra(config)
	return agent.step(obs)
