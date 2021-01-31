import numpy as np
import cmath
from collections import namedtuple

class Geometry:
		
	basis = np.array([1, cmath.exp(2j * cmath.pi * 1 / 3), cmath.exp(2j * cmath.pi * 2 / 3)])
	HistMatchResult = namedtuple("HistMatchResult", "idx length")

	@classmethod
	def find_all_longest(cls, seq, max_len=None):
		"""Find all indices where end of `seq` matches some past."""
		result = []
		i_search_start = len(seq) - 2

		while i_search_start > 0:
			i_sub = -1
			i_search = i_search_start
			length = 0

			while i_search >= 0 and seq[i_sub] == seq[i_search]:
				length += 1
				i_sub -= 1
				i_search -= 1
				if max_len is not None and length > max_len:
					break

			if length > 0:
				result.append(Geometry.HistMatchResult(i_search_start + 1, length))
			i_search_start -= 1
		result = sorted(result, key=lambda a: a.length, reverse=True)
		return result

	@classmethod
	def complex_to_probs(cls, z):
		probs = (2 * (z * Geometry.basis.conjugate()).real + 1) / 3
		if min(probs) < 0:
			probs -= min(probs)
		probs /= sum(probs)
		return probs

	@classmethod
	def z_from_action(cls, action):
		return Geometry.basis[action]

	@classmethod
	def sample_from_z(cls, z):
		probs = Geometry.complex_to_probs(z)
		return np.random.choice(3, p=probs)

	@classmethod
	def norm(cls, z):
		return (Geometry.complex_to_probs(z / abs(z))) @ Geometry.basis

	class Pred:
		def __init__(self, *, alpha):
			self.offset = 0
			self.alpha = alpha
			self.last_feat = None

		def train(self, target):
			if self.last_feat is not None:
				offset = target * self.last_feat.conjugate()   # fixed
				self.offset = (1 - self.alpha) * self.offset + self.alpha * offset

		def predict(self, feat):
			"""
			feat is an arbitrary feature with a probability on 0,1,2
			anything which could be useful anchor to start with some kind of sensible direction
			"""
			feat = Geometry.norm(feat)
			result = feat * self.offset
			self.last_feat = feat
			return result
	
	def __init__(self, alpha = 0.01):
		self.my_hist = []
		self.opp_hist = []
		self.my_opp_hist = []
		self.outcome_hist = []
		self.predictor = self.Pred(alpha=alpha)

	def __call__(self, obs, conf):
		if obs.step == 0:
			action = np.random.choice(3)
			self.my_hist.append(action)
			return action

		opp = int(obs.lastOpponentAction)
		my = self.my_hist[-1]

		self.my_opp_hist.append((my, opp))
		self.opp_hist.append(opp)

		outcome = {0: 0, 1: 1, 2: -1}[(my - opp) % 3]
		self.outcome_hist.append(outcome)

		action = self.action()
		self.my_hist.append(action)

		return action
	
	def action(self):
		self.train()
		pred = self.preds()
		return_action = Geometry.sample_from_z(pred)
		return return_action

	def train(self):
		last_beat_opp = Geometry.z_from_action((self.opp_hist[-1] + 1) % 3)
		self.predictor.train(last_beat_opp)

	def preds(self):
		hist_match = Geometry.find_all_longest(self.my_opp_hist, max_len=20)
		if not hist_match:
			return 0
		feat = Geometry.z_from_action(self.opp_hist[hist_match[0].idx])
		pred = self.predictor.predict(feat)
		return pred
	
	def set_last_action(self, action):
		self.my_hist[-1] = action

agent = Geometry()

def call_agent(obs, conf):
	return agent(obs, conf)