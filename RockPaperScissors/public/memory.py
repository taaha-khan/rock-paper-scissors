# start executing cells from here to rewrite submission.py

import random

def evaluate_pattern_efficiency(previous_step_result):
    """ 
        evaluate efficiency of the pattern and, if pattern is inefficient,
        remove it from agent's memory
    """
    pattern_group_index = previous_action["pattern_group_index"]
    pattern_index = previous_action["pattern_index"]
    pattern = groups_of_memory_patterns[pattern_group_index]["memory_patterns"][pattern_index]
    pattern["reward"] += previous_step_result
    # if pattern is inefficient
    if pattern["reward"] <= EFFICIENCY_THRESHOLD:
        # remove pattern from agent's memory
        del groups_of_memory_patterns[pattern_group_index]["memory_patterns"][pattern_index]
    
def find_action(group, group_index):
    """ if possible, find my_action in this group of memory patterns """
    if len(current_memory) > group["memory_length"]:
        this_step_memory = current_memory[-group["memory_length"]:]
        memory_pattern, pattern_index = find_pattern(group["memory_patterns"], this_step_memory, group["memory_length"])
        if memory_pattern != None:
            my_action_amount = 0
            for action in memory_pattern["opp_next_actions"]:
                # if this opponent's action occurred more times than currently chosen action
                # or, if it occured the same amount of times and this one is choosen randomly among them
                if (action["amount"] > my_action_amount or
                        (action["amount"] == my_action_amount and random.random() > 0.5)):
                    my_action_amount = action["amount"]
                    my_action = action["response"]
            return my_action, pattern_index
    return None, None

def find_pattern(memory_patterns, memory, memory_length):
    """ find appropriate pattern and its index in memory """
    for i in range(len(memory_patterns)):
        actions_matched = 0
        for j in range(memory_length):
            if memory_patterns[i]["actions"][j] == memory[j]:
                actions_matched += 1
            else:
                break
        # if memory fits this pattern
        if actions_matched == memory_length:
            return memory_patterns[i], i
    # appropriate pattern not found
    return None, None

def get_step_result_for_my_agent(my_agent_action, opp_action):
    """ 
        get result of the step for my_agent
        1, 0 and -1 representing win, tie and lost results of the game respectively
        reward will be taken from observation in the next release of kaggle environments
    """
    if my_agent_action == opp_action:
        return 0
    elif (my_agent_action == (opp_action + 1)) or (my_agent_action == 0 and opp_action == 2):
        return 1
    else:
        return -1
    
def update_current_memory(obs, my_action):
    """ add my_agent's current step to current_memory """
    # if there's too many actions in the current_memory
    if len(current_memory) > current_memory_max_length:
        # delete first two elements in current memory
        # (actions of the oldest step in current memory)
        del current_memory[:2]
    # add agent's last action to agent's current memory
    current_memory.append(my_action)
    
def update_memory_pattern(obs, group):
    """ if possible, update or add some memory pattern in this group """
    # if length of current memory is suitable for this group of memory patterns
    if len(current_memory) > group["memory_length"]:
        # get memory of the previous step
        # considering that last step actions of both agents are already present in current_memory
        previous_step_memory = current_memory[-group["memory_length"] - 2 : -2]
        previous_pattern, pattern_index = find_pattern(group["memory_patterns"], previous_step_memory, group["memory_length"])
        if previous_pattern == None:
            previous_pattern = {
                # list of actions of both players
                "actions": previous_step_memory.copy(),
                # total reward earned by using this pattern
                "reward": 0,
                # list of observed opponent's actions after each occurrence of this pattern
                "opp_next_actions": [
                    # action that was made by opponent,
                    # amount of times that action occurred,
                    # what should be the response of my_agent
                    {"action": 0, "amount": 0, "response": 1},
                    {"action": 1, "amount": 0, "response": 2},
                    {"action": 2, "amount": 0, "response": 0}
                ]
            }
            group["memory_patterns"].append(previous_pattern)
        # update previous_pattern
        for action in previous_pattern["opp_next_actions"]:
            if action["action"] == obs["lastOpponentAction"]:
                action["amount"] += 1
    
# "%%writefile -a submission.py" will append the code below to submission.py,
# it WILL NOT rewrite submission.py

# maximum steps in a memory pattern
STEPS_MAX = 5
# minimum steps in a memory pattern
STEPS_MIN = 3
# lowest efficiency threshold of a memory pattern before being removed from agent's memory
EFFICIENCY_THRESHOLD = -3
# amount of steps between forced random actions
FORCED_RANDOM_ACTION_INTERVAL = random.randint(STEPS_MIN, STEPS_MAX)

# current memory of the agent
current_memory = []
# previous action of my_agent
previous_action = {
    "action": None,
    # action was taken from pattern
    "action_from_pattern": False,
    "pattern_group_index": None,
    "pattern_index": None
}
# amount of steps remained until next forced random action
steps_to_random = FORCED_RANDOM_ACTION_INTERVAL
# maximum length of current_memory
current_memory_max_length = STEPS_MAX * 2
# current reward of my_agent
# will be taken from observation in the next release of kaggle environments
reward = 0
# memory length of patterns in first group
# STEPS_MAX is multiplied by 2 to consider both my_agent's and opponent's actions
group_memory_length = current_memory_max_length
# list of groups of memory patterns
groups_of_memory_patterns = []
for i in range(STEPS_MAX, STEPS_MIN - 1, -1):
    groups_of_memory_patterns.append({
        # how many steps in a row are in the pattern
        "memory_length": group_memory_length,
        # list of memory patterns
        "memory_patterns": []
    })
    group_memory_length -= 2
    
# "%%writefile -a submission.py" will append the code below to submission.py,
# it WILL NOT rewrite submission.py

def my_agent(obs, conf):
    """ your ad here """
    # action of my_agent
    my_action = None
    
    # forced random action
    global steps_to_random
    steps_to_random -= 1
    if steps_to_random <= 0:
        steps_to_random = FORCED_RANDOM_ACTION_INTERVAL
        # choose action randomly
        my_action = random.randint(0, 2)
        # save action's data
        previous_action["action"] = my_action
        previous_action["action_from_pattern"] = False
        previous_action["pattern_group_index"] = None
        previous_action["pattern_index"] = None
    
    # if it's not first step
    if obs["step"] > 0:
        # add opponent's last step to current_memory
        current_memory.append(obs["lastOpponentAction"])
        # previous step won or lost
        previous_step_result = get_step_result_for_my_agent(current_memory[-2], current_memory[-1])
        global reward
        reward += previous_step_result
        # if previous action of my_agent was taken from pattern
        if previous_action["action_from_pattern"]:
            evaluate_pattern_efficiency(previous_step_result)
    
    for i in range(len(groups_of_memory_patterns)):
        # if possible, update or add some memory pattern in this group
        update_memory_pattern(obs, groups_of_memory_patterns[i])
        # if action was not yet found
        if my_action == None:
            my_action, pattern_index = find_action(groups_of_memory_patterns[i], i)
            if my_action != None:
                # save action's data
                previous_action["action"] = my_action
                previous_action["action_from_pattern"] = True
                previous_action["pattern_group_index"] = i
                previous_action["pattern_index"] = pattern_index
    
    # if no action was found
    if my_action == None:
        # choose action randomly
        my_action = random.randint(0, 2)
        # save action's data
        previous_action["action"] = my_action
        previous_action["action_from_pattern"] = False
        previous_action["pattern_group_index"] = None
        previous_action["pattern_index"] = None
    
    # add my_agent's current step to current_memory
    update_current_memory(obs, my_action)
    return my_action
