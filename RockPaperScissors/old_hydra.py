
from sklearn.tree import DecisionTreeClassifier
from collections import defaultdict
import pandas as pd
import numpy as np
import operator
import getpass
import random
import json
import time

LOCAL_MODE = getpass.getuser() == 'taaha'
SAVE_DATA = LOCAL_MODE and False
OUTPUT = not LOCAL_MODE

class agent():
	''' Base class for all agents '''

	def initial_step(self, obs = None, config = None):
		''' Move to play on initial step '''
		return random.randrange(3)
	
	def step(self, history, obs = None, config = None):
		''' Next moves with historic move data '''
		return self.initial_step()
	
	def set_last_action(self, actions):
		''' Overwriting Personal Agent History '''
		return None
	

# similar to the transition matrix but rely on both previous steps
class transition_tensor(agent):
	
	def __init__(self, deterministic = False, counter_strategy = False, init_value = 0.1, decay = 1):
		self.deterministic = deterministic
		self.counter_strategy = counter_strategy
		if counter_strategy:
			self.step_type1 = 'step' 
			self.step_type2 = 'competitorStep'
		else:
			self.step_type2 = 'step' 
			self.step_type1 = 'competitorStep'
		self.init_value = init_value
		self.decay = decay
		
	def step(self, history, obs, config):
		matrix = np.zeros((3, 3, 3)) + 0.1
		for i in range(len(history) - 1):
			matrix = (matrix - self.init_value) / self.decay + self.init_value
			matrix[int(history[i][self.step_type1]), int(history[i][self.step_type2]), int(history[i+1][self.step_type1])] += 1

		if self.deterministic:
			step = np.argmax(matrix[int(history[-1][self.step_type1]), int(history[-1][self.step_type2])])
		else:
			step = np.random.choice([0,1,2], p = matrix[int(history[-1][self.step_type1]), int(history[-1][self.step_type2])]/matrix[int(history[-1][self.step_type1]), int(history[-1][self.step_type2])].sum())
		
		if self.counter_strategy:
			# we predict our step using transition matrix (as competitor can do) and beat probable competitor step
			return (step + 2) % 3 
		else:
			# we just predict competitors step and beat it
			return (step + 1) % 3


# looks for the same pattern in history and returns the best answer to the most possible counter strategy
class pattern_matching(agent):
	def __init__(self, steps = 3, deterministic = False, counter_strategy = False, init_value = 0.1, decay = 1):
		self.deterministic = deterministic
		self.counter_strategy = counter_strategy
		if counter_strategy:
			self.step_type = 'step' 
		else:
			self.step_type = 'competitorStep'
		self.init_value = init_value
		self.decay = decay
		self.steps = steps
		
	def step(self, history, obs, config):
		if len(history) < self.steps + 1:
			return self.initial_step()
		
		next_step_count = np.zeros(3) + self.init_value
		pattern = [history[i][self.step_type] for i in range(- self.steps, 0)]
		
		for i in range(len(history) - self.steps):
			next_step_count = (next_step_count - self.init_value)/self.decay + self.init_value
			current_pattern = [history[j][self.step_type] for j in range(i, i + self.steps)]
			if np.sum([pattern[j] == current_pattern[j] for j in range(self.steps)]) == self.steps:
				next_step_count[history[i + self.steps][self.step_type]] += 1
		
		if next_step_count.max() == self.init_value:
			return self.initial_step()
		
		if self.deterministic:
			step = np.argmax(next_step_count)
		else:
			step = random.choice([0,1,2], p = next_step_count/next_step_count.sum())
		
		if self.counter_strategy:
			# we predict our step using transition matrix (as competitor can do) and beat probable competitor step
			return (step + 2) % 3 
		else:
			# we just predict competitors step and beat it
			return (step + 1) % 3


# Decision Tree Classifier: https://www.kaggle.com/alexandersamarin/decision-tree-classifier
class decision_tree(agent):

	def __init__(self, noise = False):

		# Given Data
		self.config = None
		self.obs = None

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

	def step(self, history, observation, configuration):

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
	
	def set_last_action(self, actions):
		self.last_move['action'] = actions[-1]


# Greenberg Agent: https://www.kaggle.com/group16/rps-roshambo-competition-greenberg
class greenberg(agent):
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

	def step(self, history, obs, config):
		rps_to_text = ('rock','paper','scissors')
		self.my_hist.append(rps_to_text[history[-1]['step']])
		self.opponent_hist.append(rps_to_text[history[-1]['competitorStep']])
		self.act = self.player(self.my_hist, self.opponent_hist)
		return self.act


# Iocane Powder Agent: https://www.kaggle.com/group16/rps-roshambo-comp-iocaine-powder
class iocaine(agent):

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
			self.stats = iocaine.Stats()
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
			self.predict_meta = [iocaine.Predictor() for a in range(len(self.ages))]
			self.stats = [iocaine.Stats() for i in range(2)]
			self.histories = [[], [], []]

		def predictor(self, dims=None):
			"""Returns a nested array of predictor objects, of the given dimensions."""
			if dims: return [self.predictor(dims[1:]) for i in range(dims[0])]
			self.predictors.append(iocaine.Predictor())
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
				best = [iocaine.recall(age, hist) for hist in self.histories]
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

	def step(self, history, observation, configuration):
		return self.iocaine.move(observation.lastOpponentAction)
	
	def set_last_action(self, actions):
		if self.iocaine != None:
			self.iocaine.histories[0] = actions


# Rank 1 Agent From RPSContest
class rank1(agent):
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
		self.to_number = {'R': 0, 'P': 1, 'S': 2}
	
	def initial_step(self, obs, config):
		return self.to_number[self.output]
	
	def step(self, history, obs, config):
		input = 'RPS'[history[-1]['competitorStep']]
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
		return self.to_number[self.output]

	def set_last_action(self, actions):
		self.output = 'RPS'[actions[-1]]


# Testing Please Ignore Agent from RPSContest
class testing_please_ignore(agent):

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
		self.to_number = {'R': 0, 'P': 1, 'S': 2}

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
		return self.to_number[self.output]

	def step(self, history, obs, config):

		input = 'RPS'[history[-1]['competitorStep']]

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
		return self.to_number[self.output]
		
	def set_last_action(self, actions):
		self.output = 'RPS'[actions[-1]]


# Memory Patterns V7 Agent
class memory_patterns(agent):

	def __init__(self):

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

	def initial_step(self, obs, config):
		return self.step([], obs, config)

	def step(self, history, obs, config):
		# action of my_agent
		my_action = None
		# if it's not first step, add opponent's last action to agent's current memory
		# and reassess effectiveness of current memory patterns
		if obs.step > 0:
			self.current_memory.append(obs["lastOpponentAction"])
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
					if action["action"] == obs["lastOpponentAction"]:
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
		if my_action == None:
			my_action = random.randint(0, 2)
		self.current_memory.append(my_action)
		return my_action
	
	def set_last_action(self, actions):
		self.current_memory[-1] = actions[-1]


# Hydra Net of Agents
agents = {

	'rank1': rank1(),
	'testing_please_ignore': testing_please_ignore(),
	'memory_patterns': memory_patterns(),

	'iocaine': iocaine(),
	'greenberg': greenberg(),

	'deterministic_decision_tree': decision_tree(),
	'random_decision_tree': decision_tree(True),

	'random_transitison_tensor': transition_tensor(False, False),
	'determenistic_transitison_tensor': transition_tensor(True, False),
	'random_self_trans_tensor': transition_tensor(False, True),
	'determenistic_self_trans_tensor': transition_tensor(True, True),
	
	'random_transitison_tensor_decay': transition_tensor(False, False, decay = 1.05),
	'random_self_trans_tensor_decay': transition_tensor(False, True, decay = 1.05),
	'determenistic_transitison_tensor_decay': transition_tensor(True, False, decay = 1.05),
	'determenistic_self_trans_tensor_decay': transition_tensor(True, True, decay = 1.05),
	
	'determenistic_pattern_matching_decay_3': pattern_matching(3, True, False, decay = 1.001),
	'determenistic_self_pattern_matching_decay_3': pattern_matching(3, True, True, decay = 1.001),

}

history = []
bandit_state = {k: [1, 1] for k in agents.keys()}
data = {}

def hydra_agent(obs, config):

	global history, bandit_state, data

	start_time = time.time()
	
	# bandits' params
	step_size = 3 # how much we increase a and b 
	decay_rate = 1.05 # how much do we decay old historical data
	
	def log_step(step = None, history = None, agent = None, competitorStep = None, file = 'hydra_history.csv'):
		history.append({'step': step, 'competitorStep': competitorStep, 'agent': agent})
		if SAVE_DATA:
			if file is not None:
				pd.DataFrame(history).to_csv(file, index = False)
		return step
	
	def update_competitor_step(history, competitorStep):
		history[-1]['competitorStep'] = int(competitorStep)
		return history

	if obs.step > 0: 
		history = update_competitor_step(history, obs.lastOpponentAction)
	data[obs.step] = {}

	for name, agent in agents.items():

		# First move: initialize
		if obs.step == 0:
			agent_step = int(agent.initial_step(obs, config))
		
		# Run agent's move
		else: 

			prev_step = data[obs.step - 1][name]

			bandit_state[name][1] = (bandit_state[name][1] - 1) / decay_rate + 1
			bandit_state[name][0] = (bandit_state[name][0] - 1) / decay_rate + 1
			
			if (history[-1]['competitorStep'] - prev_step) % 3 == 1:
				bandit_state[name][1] += step_size
			elif (history[-1]['competitorStep'] - prev_step) % 3 == 2:
				bandit_state[name][0] += step_size
			else:
				bandit_state[name][0] += step_size / 2
				bandit_state[name][1] += step_size / 2

			agent_step = int(agent.step(history, obs, config))

		data[obs.step][name] = agent_step

	# we can use it for analysis later
	if SAVE_DATA:
		with open('hydra_data.json', 'w') as outfile:
			json.dump(bandit_state, outfile)
	
	# generate random number from Beta distribution for each agent and select the most lucky one
	best_proba = float('-inf')
	best_agent = None
	for k in agents.keys():
		proba = np.random.beta(bandit_state[k][0], bandit_state[k][1])
		if proba > best_proba:
			best_proba = proba
			best_agent = k

	action = data[obs.step][best_agent]

	last_actions = [packet['step'] for packet in history] + [action]
	for name in agents.keys():
		agents[name].set_last_action(last_actions)

	if OUTPUT:
		score = round(best_proba, 3)
		pad = ' ' * (5 - len(str(score)))
		print(f'Score: {score}{pad}  Agent: {best_agent}')

	return log_step(action, history, best_agent)