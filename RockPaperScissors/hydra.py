
from sklearn.tree import DecisionTreeClassifier
from operator import itemgetter
import pandas as pd
import numpy as np
import getpass
import random
import json
import time

LOCAL_MODE = getpass.getuser() == 'taaha'
SAVE_DATA = LOCAL_MODE and False
OUTPUT = not LOCAL_MODE

class agent():
	''' Base class for all agents '''

	# Move to play on first step
	def initial_step(self, obs = None, config = None):
		return random.randrange(3)
	
	# Moves with historic move data
	def step(self, history, obs = None, config = None):
		return self.initial_step()
	
	# Resetting Personal Agent History
	def set_last_action(self, actions):
		return None
	
# simple transition matrix: previous step -> next step
class transition_matrix(agent):
	def __init__(self, deterministic = False, counter_strategy = False, init_value = 0.1, decay = 1):
		self.deterministic = deterministic
		self.counter_strategy = counter_strategy
		if counter_strategy:
			self.step_type = 'step' 
		else:
			self.step_type = 'competitorStep'
		self.init_value = init_value
		self.decay = decay
		
	def step(self, history, obs, config):
		matrix = np.zeros((3,3)) + self.init_value
		for i in range(len(history) - 1):
			matrix = (matrix - self.init_value) / self.decay + self.init_value
			matrix[int(history[i][self.step_type]), int(history[i+1][self.step_type])] += 1

		if self.deterministic:
			step = np.argmax(matrix[int(history[-1][self.step_type])])
		else:
			step = np.random.choice([0,1,2], p = matrix[int(history[-1][self.step_type])]/matrix[int(history[-1][self.step_type])].sum())
		
		if self.counter_strategy:
			# we predict our step using transition matrix (as competitor can do) and beat probable competitor step
			return (step + 2) % 3 
		else:
			# we just predict competitors step and beat it
			return (step + 1) % 3
	

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
			return min(enumerate(values), key=itemgetter(1))[0]

		def max_index(values):
			return max(enumerate(values), key=itemgetter(1))[0]

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

# Hydra Net of Agents
agents = {

	'iocaine': iocaine(),

	'deterministic_decision_tree': decision_tree(),
	'random_decision_tree': decision_tree(True),

	'random_transitison_matrix': transition_matrix(False, False),
	'determenistic_transitison_matrix': transition_matrix(True, False),
	'random_self_trans_matrix': transition_matrix(False, True),
	'determenistic_self_trans_matrix': transition_matrix(True, True),

	'random_transitison_tensor': transition_tensor(False, False),
	'determenistic_transitison_tensor': transition_tensor(True, False),
	'random_self_trans_tensor': transition_tensor(False, True),
	'determenistic_self_trans_tensor': transition_tensor(True, True),
	
	'random_transitison_matrix_decay': transition_matrix(False, False, decay = 1.05),
	'random_self_trans_matrix_decay': transition_matrix(False, True, decay = 1.05),
	'random_transitison_tensor_decay': transition_tensor(False, False, decay = 1.05),
	'random_self_trans_tensor_decay': transition_tensor(False, True, decay = 1.05),
	
	'determenistic_transitison_matrix_decay': transition_matrix(True, False, decay = 1.05),
	'determenistic_self_trans_matrix_decay': transition_matrix(True, True, decay = 1.05),
	'determenistic_transitison_tensor_decay': transition_tensor(True, False, decay = 1.05),
	'determenistic_self_trans_tensor_decay': transition_tensor(True, True, decay = 1.05),
	
	'determenistic_pattern_matching_decay_3': pattern_matching(3, True, False, decay = 1.001),
	'determenistic_self_pattern_matching_decay_3': pattern_matching(3, True, True, decay = 1.001),
}

history = []
bandit_state = {k:[1,1] for k in agents.keys()}
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

	if obs.step:
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