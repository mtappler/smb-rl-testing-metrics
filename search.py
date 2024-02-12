from util import make_search_state
import copy

# This file implements the DFS for a reference trace to the goal
def find_actions_idx(env, meaning0='right B', meaning1='right A B'):
    """
    Given an environment, this function returns the index of the actions with the meanings specified by the parameters,
    where the meaning corresponds to button presses
    Args:
        env: an SMB environment
        meaning0: a meaning (button presses), for which we want to look up the index
        meaning1: a meaning (button presses), for which we want to look up the index

    Returns: a pair of action index for the specified meanings

    """
    action0 = 0
    action1 = 1
    # check all meanings of the actions registered in the environment to find the specified meanings
    action_meanings = env.get_action_meanings()
    for i in range(len(action_meanings)):
        meaning = action_meanings[i]
        if meaning == meaning0:
            action0 = i
        if meaning == meaning1:
            action1 = i
    return action0, action1


def default_action(invert_action_selection, action0=0, action1=1):
    """
    The function returns the default action that shall be tried first at the current level of the DFS.
    It may make sense to sometimes invert the selection after executing an action in the SMB environments, e.g., to hold
    jump for a longer time, when we switch from running to jumping.
    Args:
        invert_action_selection: True if the default action shall be switched from running to jumping
        action0: an action index (at start this is the default)
        action1: an action index

    Returns: the index of the default action in the current stage of the search

    """
    if invert_action_selection:
        return action1
    else:
        return action0


def other_actions(invert_action_selection, action0=0, action1=1):
    """
    This functions returns a list containing the actions other than the default action. Currently, the search is limited
    to two actions so the function returns a list of one action.
    Args:
        invert_action_selection: True if the default action selection is currently inverted
        action0: an action index (default action at the start)
        action1: an action index

    Returns: a list containing a single action index

    """
    if invert_action_selection:
        return [action0]
    else:
        return [action1]


def search_act(action_stack, search_steps, invert_action_selection, action0, action1):
    # new step
    if search_steps == len(action_stack):
        action_stack.append(other_actions(invert_action_selection, action0, action1))
        return default_action(invert_action_selection, action0, action1)
    else:
        # backtracking -> trying other actions
        available_other_actions = action_stack.pop()
        if available_other_actions:
            action = available_other_actions.pop()
            action_stack.append(available_other_actions)
            return action
        else:
            return None


def stuck(local_state_stack, frames):
    # determine whether mario does not move for 8 frames
    # mario does not move initially, so we ignore the first 8 steps
    if len(local_state_stack) < frames:
        return False
    current_xpos = local_state_stack[-1][1]['x_pos']
    for i in range(1, frames):
        if current_xpos != local_state_stack[-1 - i][1]['x_pos']:
            return False
    return True


def search(env, unwrapped_env):
    action0, action1 = find_actions_idx(env)
    invert_action_selection = False
    # search_steps = depth in search tree
    search_steps = 0
    # tuples of RL state (i.e., image data) and info returned from the
    # environment
    local_state_stack = []
    # action_stack contains backtracking points into search tree
    action_stack = []
    # bad_states: states from which we explored all paths and all lead to a stop
    # (death or getting stuck)
    bad_states = set()
    # performed_action_stack: at each point in time the trace from the initial
    # state to current state
    performed_action_stack = []
    # all traces that lead to a stop
    performed_traces = []

    finished_stage = False

    # initialization of state
    rl_state = env.reset()
    info = unwrapped_env._get_info()
    state = (rl_state, info)
    unwrapped_env._backup_stack()
    local_state_stack.append(state)

    while not finished_stage:
        # Show environment
        #env.render()
        rl_state, old_info = state
        search_state = make_search_state(unwrapped_env, old_info, search_steps)
        # when we know that we are in a bad state, we just replace the actions
        # that we would still need to execute it (i.e., we remove the
        # backtracking point)
        if search_state in bad_states:
            action_stack.pop()
            action_stack.append([])
            action = None
        else:
            # choose an action
            action = search_act(action_stack, search_steps, invert_action_selection, action0, action1)
        if action is not None:
            # back up emulator state on stack (unintentionally created the method "privately")
            unwrapped_env._backup_stack()
            next_rl_state, reward, done, info = env.step(action)
            performed_action_stack.append(action)
            next_state = (next_rl_state, info)
            local_state_stack.append(next_state)
            # here we stop and backtrack to try the other action (s) available at
            # the current depth in the search tree
            if done and not info['flag_get'] or stuck(local_state_stack, frames=4):
                # reset done because environment does not let us step in a "done" environment
                unwrapped_env.done = False
                # restore last emulator state
                unwrapped_env._restore_stack()
                state = local_state_stack.pop()
                performed_traces.append(copy.deepcopy(performed_action_stack))
                performed_action_stack.pop()
                # whenever we backtrack we invert the default action
                invert_action_selection = not invert_action_selection
            # finished when we reach the flag
            elif info['flag_get']:
                finished_stage = True
                return performed_action_stack
            else:  # otherwise, just carry on with the search
                state = next_state
                search_steps += 1
        else:
            # here we reached a point we have no other actions to try, so we
            # backtrack and go one level up in the search tree, the current
            # state is bad because of that
            rl_state, info = state
            search_state = make_search_state(unwrapped_env, info, search_steps)
            bad_states.add(search_state)
            unwrapped_env.done = False
            unwrapped_env._restore_stack()
            state = local_state_stack.pop()
            performed_action_stack.pop()
            # whenever we backtrack we invert the default action
            invert_action_selection = not invert_action_selection
            search_steps -= 1