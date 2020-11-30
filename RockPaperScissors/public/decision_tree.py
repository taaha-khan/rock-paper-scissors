from sklearn.tree import DecisionTreeClassifier
import numpy as np
import random

def construct_local_features(rollouts):
	features = np.array([[step % k for step in rollouts['steps']] for k in (2, 3, 5)])
	features = np.append(features, rollouts['steps'])
	features = np.append(features, rollouts['actions'])
	features = np.append(features, rollouts['opp-actions'])
	return features

def construct_global_features(rollouts):
	features = []
	for key in ['actions', 'opp-actions']:
		for i in range(3):
			actions_count = np.mean([r == i for r in rollouts[key]])
			features.append(actions_count)
	return np.array(features)

def construct_features(short_stat_rollouts, long_stat_rollouts):
	lf = construct_local_features(short_stat_rollouts)
	gf = construct_global_features(long_stat_rollouts)
	features = np.concatenate([lf, gf])
	return features

def predict_opponent_move(train_data, test_sample):
	classifier = DecisionTreeClassifier(random_state = 42)
	classifier.fit(train_data['x'], train_data['y'])
	return classifier.predict(test_sample)

def update_rollouts_hist(rollouts_hist, last_move, opp_last_action):
	rollouts_hist['steps'].append(last_move['step'])
	rollouts_hist['actions'].append(last_move['action'])
	rollouts_hist['opp-actions'].append(opp_last_action)
	return rollouts_hist

def warmup_strategy(observation, configuration):
	global rollouts_hist, last_move

	action = random.randint(0, 2)

	if observation.step == 0:
		last_move = {'step': 0, 'action': action}
		rollouts_hist = {'steps': [], 'actions': [], 'opp-actions': []}
	else:
		rollouts_hist = update_rollouts_hist(rollouts_hist, last_move, observation.lastOpponentAction)
		last_move = {'step': observation.step, 'action': action}
	return action

def init_training_data(rollouts_hist, k):
	for i in range(len(rollouts_hist['steps']) - k + 1):
		short_stat_rollouts = {key: rollouts_hist[key][i : i + k] for key in rollouts_hist}
		long_stat_rollouts = {key: rollouts_hist[key][:i + k] for key in rollouts_hist}
		features = construct_features(short_stat_rollouts, long_stat_rollouts)        
		data['x'].append(features)
	test_sample = data['x'][-1].reshape(1, -1)
	data['x'] = data['x'][:-1]
	data['y'] = rollouts_hist['opp-actions'][k:]
	return (data, test_sample)

def decision_tree_agent(observation, configuration):

	global rollouts_hist, last_move, data, test_sample
	
	# hyperparameters
	min_samples = 25
	k = 5

	if observation.step == 0:
		data = {'x': [], 'y': []}

	# if not enough data -> randomize
	if observation.step <= min_samples + k:
		return warmup_strategy(observation, configuration)
	
	# update statistics
	rollouts_hist = update_rollouts_hist(rollouts_hist, last_move, observation.lastOpponentAction)
	
	# update training data
	if len(data['x']) == 0:
		data, test_sample = init_training_data(rollouts_hist, k)
	else:
		short_stat_rollouts = {key: rollouts_hist[key][-k:] for key in rollouts_hist}
		features = construct_features(short_stat_rollouts, rollouts_hist)
		data['x'].append(test_sample[0])
		data['y'] = rollouts_hist['opp-actions'][k:]
		test_sample = features.reshape(1, -1)
		
	# predict opponents move and choose an action
	next_opp_action_pred = predict_opponent_move(data, test_sample)
	action = int((next_opp_action_pred + 1) % 3)

	# if random.random() < 0.2:
	# 	action = random.randint(0, 2)

	last_move = {'step': observation.step, 'action': action}
	
	return action
