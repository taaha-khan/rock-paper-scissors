
# Random Forest Random Agent: https://www.kaggle.com/jumaru/random-forest-random-rock-paper-scissors
# WARNING: SLOW AF

import random
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

actions =  np.empty((0,0), dtype = int)
observations =  np.empty((0,0), dtype = int)
total_reward = 0

def random_forest_random(observation, configuration):
	global actions, observations, total_reward
	
	if observation.step == 0:
		action = random.randint(0,2)
		actions = np.append(actions , [action])
		return action
	
	if observation.step == 1:
		action = random.randint(0,2)
		actions = np.append(actions , [action])
		observations = np.append(observations , [observation.lastOpponentAction])
		# Keep track of score
		winner = int((3 + actions[-1] - observation.lastOpponentAction) % 3);
		if winner == 1:
			total_reward = total_reward + 1
		elif winner == 2:
			total_reward = total_reward - 1        
		return action

	# Get Observation to make the tables (actions & obervations) even.
	observations = np.append(observations , [observation.lastOpponentAction])
	
	# Prepare Data for training
	# :-1 as we dont have feedback yet.
	X_train = np.vstack((actions[:-1], observations[:-1])).T
	
	# Create Y by rolling observations to bring future a step earlier 
	shifted_observations = np.roll(observations, -1)
	
	# trim rolled & last element from rolled observations
	y_train = shifted_observations[:-1].T
	
	# Set the history period. Long chains here will need a lot of time
	if len(X_train) > 25:
		random_window_size = 10 + random.randint(0,10)
		X_train = X_train[-random_window_size:]
		y_train = y_train[-random_window_size:]
   
	# Train a classifier model
	model = RandomForestClassifier(n_estimators=25)
	model.fit(X_train, y_train)

	# Predict
	X_test = np.empty((0,0), dtype = int)
	X_test = np.append(X_test, [int(actions[-1]), observation.lastOpponentAction])
	prediction = model.predict(X_test.reshape(1, -1))

	# Keep track of score
	winner = int((3 + actions[-1] - observation.lastOpponentAction) % 3);
	if winner == 1:
		total_reward = total_reward + 1
	elif winner == 2:
		total_reward = total_reward - 1
   
	# Prepare action
	action = int((prediction + 1) % 3)
	
	# If losing a bit then change strategy and break the patterns by playing a bit random
	if total_reward < -2:
		win_tie = random.randint(0,1)
		action = int((prediction + win_tie) % 3)

	# Update actions
	actions = np.append(actions , [action])

	# Action 
	return action 