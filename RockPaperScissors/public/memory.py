'''
Rock, Paper, Scissors with Memory Patterns V7: https://www.kaggle.com/yegorbiryukov/rock-paper-scissors-with-memory-patterns?scriptVersionId=46447097
'''

import random

# maximum steps in the pattern
steps_max = 6
# minimum steps in the pattern
steps_min = 3
# maximum amount of steps until reassessment of effectiveness of current memory patterns
max_steps_until_memory_reassessment = random.randint(80, 120)

# current memory of the agent
current_memory = []
# list of 1, 0 and -1 representing win, tie and lost results of the game respectively
# length is max_steps_until_memory_r_t
results = []
# current best sum of results
best_sum_of_results = 0
# memory length of patterns in first group
# steps_max is multiplied by 2 to consider both my_agent's and opponent's actions
group_memory_length = steps_max * 2
# list of groups of memory patterns
groups_of_memory_patterns = []
for i in range(steps_max, steps_min - 1, -1):
    groups_of_memory_patterns.append({
        # how many steps in a row are in the pattern
        "memory_length": group_memory_length,
        # list of memory patterns
        "memory_patterns": []
    })
    group_memory_length -= 2

    
def find_pattern(memory_patterns, memory, memory_length):
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

def get_step_result_for_my_agent(my_agent_action, opp_action):
    """ 
        get result of the step for my_agent
        1, 0 and -1 representing win, tie and lost results of the game respectively
    """
    if my_agent_action == opp_action:
        return 0
    elif (my_agent_action == (opp_action + 1)) or (my_agent_action == 0 and opp_action == 2):
        return 1
    else:
        return -1

def agent(obs, conf):
    """ your ad here """
    global results
    global best_sum_of_results
    # action of my_agent
    my_action = None
    # if it's not first step, add opponent's last action to agent's current memory
    # and reassess effectiveness of current memory patterns
    if obs["step"] > 0:
        current_memory.append(obs["lastOpponentAction"])
        # previous step won or lost
        results.append(get_step_result_for_my_agent(current_memory[-2], current_memory[-1]))
        # if there is enough steps added to results for memery reassessment
        if len(results) == max_steps_until_memory_reassessment:
            results_sum = sum(results)
            # if effectiveness of current memory patterns has decreased significantly
            if results_sum < (best_sum_of_results * 0.5):
                # flush all current memory patterns
                best_sum_of_results = 0
                results = []
                for group in groups_of_memory_patterns:
                    group["memory_patterns"] = []
            else:
                # if effectiveness of current memory patterns has increased
                if results_sum > best_sum_of_results:
                    best_sum_of_results = results_sum
                del results[:1]
    for group in groups_of_memory_patterns:
        # if length of current memory is bigger than necessary for a new memory pattern
        if len(current_memory) > group["memory_length"]:
            # get momory of the previous step
            previous_step_memory = current_memory[:group["memory_length"]]
            previous_pattern = find_pattern(group["memory_patterns"], previous_step_memory, group["memory_length"])
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
            del current_memory[:2]
            # if action was not yet found
            if my_action == None:
                pattern = find_pattern(group["memory_patterns"], current_memory, group["memory_length"])
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
    current_memory.append(my_action)
    return my_action
