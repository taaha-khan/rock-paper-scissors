
# XGBoost Agent: https://www.kaggle.com/ollyattwood/xgboost-for-predicting-opponent-s-action
# WARNING: SLOW AF

import random
import pandas as pd
from xgboost import XGBClassifier

numTurnsPredictors = 5 #number of previous turns to use as predictors
minTrainSetRows = numTurnsPredictors * 2 #only start predicting moves after we have enough data
myLastMove = None
mySecondLastMove = None
opponentLastMove = None
numDummies = 2 #how many dummy vars we need to represent a move
predictors = pd.DataFrame(columns=[str(x) for x in range(numTurnsPredictors * 2 * numDummies)])
predictors = predictors.astype("int")
opponentsMoves = []
roundHistory = [] #moves made by both players in each round
clf = XGBClassifier(n_estimators=10, verbosity=0, use_label_encoder=False)

def randomMove():
	return random.randint(0,2)

#converts my and opponents moves into dummy variables i.e. [1,2] into [0,1,1,0]
def convertToDummies(moves):
	newMoves = []
	dummies = [[0,0], [0,1], [1,0]]
	for move in moves:
		newMoves.extend(dummies[move])
	return newMoves

def updateRoundHistory(myMove, opponentMove):
	global roundHistory
	roundHistory.append(convertToDummies([myMove, opponentMove]))

def flattenData(data):
	return sum(data, [])

def updateFeatures(rounds):
	global predictors
	flattenedRounds = flattenData(rounds)
	predictors.loc[len(predictors)] = flattenedRounds

def fitAndPredict(clf, x, y, newX):
	df = pd.DataFrame.from_records([newX], columns=[str(i) for i in range(numTurnsPredictors * 2 * numDummies)])
	clf.fit(x, y)
	return int(clf.predict(df)[0])

def makeMove(observation, configuration):
	global myLastMove
	global mySecondLastMove
	global opponentLastMove
	global predictors
	global opponentsMoves
	global roundHistory

	if observation.step == 0:
		myLastMove = randomMove()
		return myLastMove

	if observation.step == 1:
		updateRoundHistory(myLastMove, observation.lastOpponentAction)
		myLastMove = randomMove()
		return myLastMove

	else:
		updateRoundHistory(myLastMove, observation.lastOpponentAction)
		opponentsMoves.append(observation.lastOpponentAction)

		if observation.step > numTurnsPredictors:
			updateFeatures(roundHistory[-numTurnsPredictors - 1: -1])

		if len(predictors) > minTrainSetRows:
			predictX = flattenData(roundHistory[-numTurnsPredictors:]) #data to predict next move
			predictedMove = fitAndPredict(clf, predictors, opponentsMoves[(numTurnsPredictors-1):], predictX)
			myLastMove = (predictedMove + 1) % 3
			return myLastMove
		else:
			myLastMove = randomMove()
			return myLastMove