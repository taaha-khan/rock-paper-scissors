# This will run the cell and re-write the `submission.py` file.

from abc import ABC, abstractmethod
import random
import functools
import numpy as np

import collections


# Set this to True only if you're debugging your agent
DEBUG_MODE = True
# Append the content of this cell to `submission.py`

#----------------------------------------------------------
#  CONSTANTS
#----------------------------------------------------------

NUM_TO_MOVE = ['R', 'P', 'S']
MOVE_TO_NUM = {'R': 0, 'P': 1, 'S': 2}

BEAT = {'R': 'P', 'P': 'S', 'S': 'R', None: None}
CEDE = {'R': 'S', 'P': 'R', 'S': 'P', None: None}
DNA_ENCODE = {
    'RP': 'a', 'PS': 'b', 'SR': 'c',
    'PR': 'd', 'SP': 'e', 'RS': 'f',
    'RR': 'g', 'PP': 'h', 'SS': 'i'}
# Run the cell and append its contentto `submission.py`

#----------------------------------------------------------
#  SYMMETRIC HISTORY STORAGE
#----------------------------------------------------------

class HistoryHolder:
  """Holds the sequence of moves since the start of the game"""
  def __init__(self):
    self.our_moves = ''
    self.his_moves = ''
    self.dna_moves = ''

  def add_moves(self, our_move, his_move):
    self.our_moves += our_move
    self.his_moves += his_move
    self.dna_moves += DNA_ENCODE[our_move + his_move]

  def __len__(self):
    if DEBUG_MODE:
      assert len(self.our_moves) == len(self.his_moves)
      assert len(self.our_moves) == len(self.dna_moves)
    return len(self.our_moves)


class HolisticHistoryHolder:
  """Holds actual history and the history in opponent's shoes"""
  def __init__(self):
    self.actual_history = HistoryHolder()
    self.mirror_history = HistoryHolder()

  def add_moves(self, our_move, his_move):
    self.actual_history.add_moves(our_move, his_move)
    self.mirror_history.add_moves(his_move, our_move)

  def __len__(self):
    if DEBUG_MODE:
      assert len(self.actual_history) == len(self.mirror_history)
    return len(self.actual_history)
# Run the cell and append its contentto `submission.py`

class BaseAtomicStrategy(ABC):
  """Interface for all atomic strategies"""

  @abstractmethod
  def __call__(self, history):
    """Returns an action to take, given the game history"""
    pass
# Run the cell and append its contentto `submission.py`


def shift_action(action, shift):
  shift = shift % 3
  if shift == 0: return action
  elif shift == 1: return BEAT[action]
  elif shift == 2: return CEDE[action]


def generate_meta_strategy_pair(atomic_strategy_cls,
                                *args, **kwargs):
  """Generate pair of strategy and anti-strategies"""
  actual_atomic = atomic_strategy_cls(*args, **kwargs)
  def _actual_strategy(holistic_history):
    return actual_atomic(holistic_history.actual_history)

  mirror_atomic = atomic_strategy_cls(*args, **kwargs)
  def _mirror_strategy(holistic_history):
    move = mirror_atomic(holistic_history.mirror_history)
    return BEAT[move]
  return _actual_strategy, _mirror_strategy
# Run the cell and append its contentto `submission.py`

#----------------------------------------------------------
#  SCORING FUNCTION FACTORIES
#----------------------------------------------------------

def get_dllu_scoring(decay=1.,
                     win_value=1.,
                     draw_value=0.,
                     lose_value=-1.,
                     drop_prob=0.,
                     drop_draw=False,
                     clip_zero=False):
  """Returns a DLLU score (daniel.lawrence.lu/programming/rps/)

  Adds 1 to previous score if we won, subtract if we lose the
  round. Previous score is multiplied by a decay parameter >0.
  Thus, if the opponent occasionally switches strategies, this
  should be able to cope.

  If a predictor loses even once, its score is reset to zero
  with some probability. This allows for much faster response
  to opponents with switching strategies.
  """
  def _scoring_func(score, our_move, his_move):
    if our_move == his_move:
      retval = decay * score + draw_value
    elif our_move == BEAT[his_move]:
      retval = decay * score + win_value
    elif our_move == CEDE[his_move]:
      retval = decay * score + lose_value

    if drop_prob > 0. and random.random() < drop_prob:
      if our_move == CEDE[his_move]:
        score = 0.
      elif drop_draw and our_move == his_move:
        score = 0.

    if clip_zero: retval = max(0., retval)
    return retval

  return _scoring_func
# Run the cell and append its contentto `submission.py`

#----------------------------------------------------------
#  STRATEGY 1: RFIND
#----------------------------------------------------------

class RFindStrategy(BaseAtomicStrategy):
  def __init__(self, limit=None, src='his'):
    self.limit = limit
    self.src = src

  def __call__(self, history):
    if len(history) == 0:
      return NUM_TO_MOVE[random.randint(0, 2)]

    # Type of lookback sequence
    if self.src == 'his':
      sequence = history.his_moves
    elif self.src == 'our':
      sequence = history.our_moves
    elif self.src == 'dna':
      sequence = history.dna_moves
    else:
      raise ValueError(f'Invalid `src` value (got {self.src}')

    # Define lookback window
    length = len(history)
    if self.limit == None:
      lb = length
    else:
      lb = min(length, self.limit)

    # RFind choose action
    while lb >= 1 and \
        not sequence[length - lb:length] in sequence[0:length - 1]:
      lb -= 1
    if lb >= 1:
      if random.random() < 0.6:
        idx = sequence.rfind(
            sequence[length - lb:length], 0, length - 1)
      elif random.random() < 0.5:
        idx = sequence.rfind(
            sequence[length - lb:length], 0, length - 1)
        idx2 = sequence.rfind(
            sequence[length - lb:length], 0, idx)
        if idx2 != -1:
          idx = idx2
      else:
        idx = sequence.find(
            sequence[length - lb:length], 0, length - 1)

      return BEAT[history.his_moves[idx + lb]]
    else:
      return random.choice('RPS')
# Run the cell and append its contentto `submission.py`

#----------------------------------------------------------
#  KUMOKO AGENT
#----------------------------------------------------------

class KumokoV1:
  def __init__(self):
    """Define scoring functions and strategies"""
    self.strategies = []
    self.proposed_actions = []
    self.proposed_meta_actions = []
    self.our_last_move = None
    self.holistic_history = HolisticHistoryHolder()

    # Add DLLU's scoring methods from his blog
    # https://daniel.lawrence.lu/programming/rps/
    dllu_scoring_configs = [
        # decay, win_val, draw_val, lose_val, drop_prob, drop_draw, clip_zero
        [ 0.80,  3.00,    0.00,     -3.00,    0.00,      False,     False    ],
        [ 0.87,  3.30,    -0.90,    -3.00,    0.00,      False,     False    ],
        [ 1.00,  3.00,    0.00,     -3.00,    1.00,      False,     False    ],
        [ 1.00,  3.00,    0.00,     -3.00,    1.00,      True,      False    ],
    ]
    self.scoring_funcs = [
        get_dllu_scoring(*cfg)
        for cfg in dllu_scoring_configs]

    # Add RFind strategies (2 meta-strategies P0 and P'0 for each)
    limits = [50, 20, 10]
    sources = ['his', 'our', 'dna']
    for limit in limits:
      for source in sources:
        self.strategies.extend(
            generate_meta_strategy_pair(RFindStrategy,
                                        *(limit, source)))

    # Add initial scores for each strategy in the list
    self.scores = 3. * np.ones(
        shape=(len(self.scoring_funcs),
               3 * len(self.strategies)))
    self.proposed_actions = [
      random.choice('RPS')
      for _ in range(self.scores.shape[1])]

    # Add meta-scores for each of the scoring function
    self.meta_scoring_func = get_dllu_scoring(
        decay=0.94,
        win_value=1.0,
        draw_value=0.0,
        lose_value=-1.0,
        drop_prob=0.87,
        drop_draw=False,
        clip_zero=True)

    self.meta_scores = 3. * np.ones(
        shape=(len(self.scoring_funcs)))
    self.proposed_meta_actions = [
        random.choice('RPS')
        for _ in range(self.meta_scores.shape[0])]

  def next_action(self, our_last_move, his_last_move):
    """Generate next move based on opponent's last move"""

    # Force last move, so that we can use Kumoko as part of
    # a larger meta-agent
    self.our_last_move = our_last_move

    # Update game history with the moves from previous
    # game step
    if his_last_move is not None:
      if DEBUG_MODE:
        assert self.our_last_move is not None
      self.holistic_history.add_moves(
          self.our_last_move, his_last_move)

    # Update score for the previous game step
    if his_last_move is not None and \
        len(self.proposed_actions) > 0:

      if DEBUG_MODE:
        assert len(self.proposed_actions) == \
          3 * len(self.strategies)
        assert len(self.proposed_meta_actions) == \
          len(self.meta_scores)
        assert self.scores.shape[0] == \
          len(self.scoring_funcs)

      # Meta-strategy selection score
      for sf in range(len(self.scoring_funcs)):
        for pa in range(len(self.proposed_actions)):
          self.scores[sf, pa] = self.scoring_funcs[sf](
              self.scores[sf, pa],
              self.proposed_actions[pa],
              his_last_move)

      # Selector selection score
      for sf in range(len(self.scoring_funcs)):
        self.meta_scores[sf] = self.meta_scoring_func(
            self.meta_scores[sf],
            self.proposed_meta_actions[sf],
            his_last_move)

    # Generate next move for each strategy
    if len(self.proposed_actions) == 0:
      self.proposed_actions = [
          random.choice('RPS')
          for _ in range(len(self.strategies) * 3)]
    else:
      for st in range(len(self.strategies)):
        proposed_action = \
          self.strategies[st](self.holistic_history)
        if proposed_action is not None:
          self.proposed_actions[st] = proposed_action
          self.proposed_actions[st + len(self.strategies)] = \
            BEAT[self.proposed_actions[st]]
          self.proposed_actions[st + 2 * len(self.strategies)] = \
            CEDE[self.proposed_actions[st]]

    # For each scoring function (selector), choose the
    # action with highest score
    best_actions_idx = np.argmax(self.scores, axis=1)
    if DEBUG_MODE:
      assert best_actions_idx.shape == \
        (len(self.scoring_funcs), )
    self.proposed_meta_actions = [
        self.proposed_actions[idx]
        for idx in best_actions_idx]

    # Meta-Selector: selecting the scoring function
    if DEBUG_MODE:
      assert len(self.meta_scores) == \
        len(self.proposed_meta_actions)
    best_meta_action_idx = np.argmax(self.meta_scores)
    self.our_last_move = \
      self.proposed_meta_actions[best_meta_action_idx]

    return self.our_last_move
# Run the cell and append its contentto `submission.py`

#----------------------------------------------------------
#  GOING META WITH KUMOKO
#----------------------------------------------------------

class MetaKumoko:
  def __init__(self,
               kumoko_cls,
               kumoko_args=[],
               kumoko_kwargs={}):
    self.kumoko_1 = kumoko_cls(
        *kumoko_args, **kumoko_kwargs)
    self.kumoko_2 = kumoko_cls(
        *kumoko_args, **kumoko_kwargs)
    self.proposed_actions = []
    self.scores = 3. * np.ones(shape=(6,))
    self.scoring_func = get_dllu_scoring(
        decay=0.94,
        win_value=1.0,
        draw_value=0.0,
        lose_value=-1.0,
        drop_prob=0.87,
        drop_draw=False,
        clip_zero=True)
    self.our_last_move = None

  def next_action(self, our_last_move, his_last_move):
    """Generate next move based on opponent's last move"""

    # Force last move, so that we can use Kumoko as part of
    # a larger meta-agent
    self.our_last_move = our_last_move

    # Score the last actions
    if his_last_move is not None and \
        len(self.proposed_actions) > 0:
      for i in range(6):
        self.scores[i] = self.scoring_func(
            self.scores[i],
            self.proposed_actions[i],
            his_last_move)

    # Generate actions for Kumoko in our shoes and in the
    # shoes of opponents (i.e. 6 meta-strategies)
    a1 = self.kumoko_1.next_action(our_last_move, his_last_move)
    a2 = self.kumoko_2.next_action(his_last_move, our_last_move)
    a2 = BEAT[a2]
    self.proposed_actions = [
        a1, a2, BEAT[a1], BEAT[a2], CEDE[a1], CEDE[a2]]

    # Selecting the best action
    best_idx = np.argmax(self.scores)
    self.our_last_move = self.proposed_actions[best_idx]
    return self.our_last_move
# Run the cell and append its contentto `submission.py`


#----------------------------------------------------------
#  FINAL AGENT IN THE COMPETITION'S FORMAT
#----------------------------------------------------------

global kumoko_agent
global latest_action
kumoko_agent = MetaKumoko(KumokoV1)
latest_action = None


def agent(obs, conf):
  global kumoko_agent
  global latest_action

  if obs.step == 0:
    s_move = kumoko_agent.next_action(None, None)
  else:
    s_his_last_move = NUM_TO_MOVE[obs.lastOpponentAction]
    s_our_last_move = NUM_TO_MOVE[latest_action]
    s_move = kumoko_agent.next_action(
        s_our_last_move, s_his_last_move)

  latest_action = MOVE_TO_NUM[s_move]

  # Surprise motherfucker
  if random.random() < 0.1 or random.randint(3, 40) > obs.step:
    latest_action = random.randint(0, 2)
  return latest_action
