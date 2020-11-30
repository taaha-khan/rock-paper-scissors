import numpy as np
import collections

def markov_agent(observation, configuration):
    k = 2
    global table, action_seq
    if observation.step == 0:
        action_seq, table = [], collections.defaultdict(lambda: [0, 0, 0])    
    if observation.step <= k:
        action = int(np.random.randint(3))
        if observation.step > 0:
            action_seq.extend([observation.lastOpponentAction, action])
        else:
            action_seq.append(action)
        return action
    # update table
    key = ''.join([str(a) for a in action_seq[:-1]])
    table[key][observation.lastOpponentAction] += 1
    # update action seq
    action_seq[:-k] = action_seq[k:]
    action_seq[-k] = observation.lastOpponentAction
    # predict opponent next move
    key = ''.join([str(a) for a in action_seq[:-1]])
    next_opponent_action_pred = np.argmax(table[key])
    action = (next_opponent_action_pred + 1) % 3
    action_seq[-1] = action

    return int(action)