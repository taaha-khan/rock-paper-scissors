import numpy as np
import torch

nn = torch.nn

class RPS(nn.Module):
	"""
	Class that predict logits of action probabilities given game history.
		Inputs: game history [bs, 2, 10].
		Outputs: logits of action probabilities [bs, 3].
	"""
	def __init__(self):
		super().__init__()
		self.conv_layers = nn.Sequential(
			nn.Conv1d(2, 4, 3, 1, 1, bias=False),
			nn.ReLU(),
			nn.AvgPool1d(2),
			nn.Conv1d(4, 8, 3, 1, 1, bias=False),
			nn.ReLU(),
			nn.AvgPool1d(2),
			nn.Conv1d(8, 16, 2, 1, 1, bias=False),
			nn.ReLU(),
			nn.AvgPool1d(2)
		)
		self.head = nn.Sequential(
			nn.Linear(16, 6),
			nn.ReLU(),
			nn.Linear(6, 3)
		)

	def forward(self, x):
		x = self.conv_layers(x)
		x = torch.flatten(x, 1)
		x = self.head(x)
		return x

def soft_cross_entropy(target, prediciton):
	log_probs = nn.functional.log_softmax(prediciton, dim=1)
	sce = -(target * log_probs).sum() / target.shape[0]
	return sce

def train_step(model, data, optimizer):
	model.train()
	torch.set_grad_enabled(True)

	X = data['X'].view(-1, 2, 10)
	y = data['y'].view(-1, 3)
	prd = model(X)
	loss = soft_cross_entropy(y, prd)

	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

bs = 6 # batch size  

opponent_actions = []
agent_actions = []
actions = []
batch_x = []
batch_y = []

model = RPS()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def agent(observation, configuration):

	# print(f'{observation.step} {observation.reward}')

	global actions, agent_actions, opponent_actions
	global model, optimizer
	global batch_x, batch_y
	global bs
	
	# first step
	if observation.step == 0:
		hand = np.random.randint(2)
		actions.append(hand)
		return hand
	
	# first warm up rounds
	if 0 < observation.step < 12:
		opponent_actions.append(observation.lastOpponentAction)
		agent_actions.append(actions[-1])
		hand = np.random.randint(2)
		actions.append(hand)
		return hand
	
	# start to train CNN
	elif observation.step >= 12:
		opponent_actions.append(observation.lastOpponentAction)
		agent_actions.append(actions[-1])
		
		wining_action = (opponent_actions[-1] + 1) % 3 
		fair_action = opponent_actions[-1]
		lose_action = (opponent_actions[-1] - 1) % 3 

		# soft labels for target    
		y = [0, 0, 0]
		y[wining_action] = 0.7
		y[fair_action] = 0.2
		y[lose_action] = 0.1 
		
		# add data for history
		batch_x.append([opponent_actions[-2:-12:-1],
						agent_actions[-2:-12:-1]])
		batch_y.append(y)
		
		# data for single CNN update 
		data = {'X': torch.Tensor([opponent_actions[-2:-12:-1],
								   agent_actions[-2:-12:-1]]),
				'y': torch.Tensor(y)} 
		
		# evaluate single training step
		train_step(model, data, optimizer)
		
		# evaluate mini-batch training steps
		if observation.step % 10 == 0:
			k = 1 if observation.step < 100 else 3
			for _ in range(k):
				idxs = np.random.choice(list(range(len(batch_y))), bs)
				data = {'X': torch.Tensor(np.array(batch_x)[idxs]),
						'y': torch.Tensor(np.array(batch_y)[idxs])}
				train_step(model, data, optimizer)
		
		# data for current action prediction
		X_prd = torch.Tensor([opponent_actions[-1:-11:-1],
							  agent_actions[-1:-11:-1]]).view(1, 2, -1)
		
		# predict logits
		probs = model(X_prd).view(3)
		# calculate probabilities
		probs = nn.functional.softmax(probs, dim=0).detach().cpu().numpy()
		
		# choose action
		hand = np.random.choice([0, 1, 2], p=probs)
		actions.append(hand)
		
		return int(hand)