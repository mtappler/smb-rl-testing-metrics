import torch

from aalpy.automata import Mdp

"""
This file contains some utility functions for fuzzing and search.
"""

# global variable that controls the state abstraction, if True, we consider only x and y coordinate in the abstraction
# otherwise, we include speed and momentum
use_small_state_space = False
run_trace_steps = 0


def save_explicit_mdp(mdp : Mdp, input_dict, file_name):
    lab_string, tra_string = mdp_to_tra_lab(mdp,input_dict)
    with open(f"{file_name}.lab", "w") as fp:
        fp.write(lab_string)
    with open(f"{file_name}.tra", "w") as fp:
        fp.write(tra_string)
        
        
def mdp_to_tra_lab(mdp : Mdp, input_dict):
    # input_dict : input -> int
    # assume input enabledness, does not work well otherwise
    output_dict = dict()
    state_reordering = dict()
    q_i = 0
    for s in mdp.states:
        if s not in state_reordering.keys():
            state_reordering[s.state_id] = q_i
            q_i += 1
        
    i = 0
    for s in mdp.states:
       labels = s.output.split("__")
       for l in labels:
           if l not in output_dict.keys():
               if l == "Init":
                   l = "init"
               output_dict[l] = i
               i += 1
    label_strings = ["#DECLARATION"]
    label_strings.append(" ".join(list(output_dict.keys())))
    label_strings.append("#END")
    for s in mdp.states:
        labels = "init" if s.output == "Init" else " ".join(s.output.split("__")) 
        label_strings.append(f"{state_reordering[s.state_id]} {labels}")
    complete_lab_string = "\n".join(label_strings)
    #mdp
    # 0 0 1 0.3
    # 0 0 4 0.7
    # 0 1 0 0.5
    # 0 1 1 0.5
    # 1 0 1 1.0
    tra_strings = ["mdp"]
    for s in mdp.states:
        source_id = state_reordering[s.state_id]
        for inp in input_dict.keys():
            inp_id = input_dict[inp]
            for t in s.transitions[inp]:
                target_id = state_reordering[t[0].state_id]
                prob = t[1]
                tra_strings.append(f"{source_id} {inp_id} {target_id} {prob}")
    complete_tra_string = "\n".join(tra_strings)
    return complete_lab_string, complete_tra_string
                
                
    
    

# capture the game state
def make_search_state(unwrapped_env, info, search_steps):
    """
    This function creates an abstract state that is used by the search to detect loops in the search and to identify bad
    states from which we do not perform a search. We use the ram addresses that can be found at:
     https://gist.github.com/1wErt3r/4048722

    Args:
        unwrapped_env: unwrapped environment, so we can access the emulator RAM
        info: the info dictionary returned by the step function of the gym environment
        search_steps: current level in the DFS (unused in experiments)

    Returns: abstract state

    """
    momentum_x = unwrapped_env.ram[0x0705]  # Player_X_MoveForce
    momentum_y = unwrapped_env.ram[0x0433]  # Player_Y_MoveForce
    speed_x = unwrapped_env.ram[0x86]  # Player_X_Speed
    speed_y = unwrapped_env.ram[0x9f]  # Player_Y_Speed
    # ignore that, unused actually
    enemy_pos = (unwrapped_env.ram[0x87], unwrapped_env.ram[0x88], unwrapped_env.ram[0x89], unwrapped_env.ram[0x8a],
                 unwrapped_env.ram[0x8b], unwrapped_env.ram[0x8c])  # Enemy_X_Position
    if use_small_state_space:
        return info['x_pos'], info['y_pos']
    elif search_steps:
        return search_steps, info['x_pos'], info['y_pos'], momentum_x, momentum_y, speed_x, speed_y
    else:
        return info['x_pos'], info['y_pos'], momentum_x, momentum_y, speed_x, speed_y



# replay an existing trace for debugging, e.g., finding a trace leading to the
# flag (such a trace might not exist if we skip too many frames)
def run_trace(unwrapped_env, env, replay_trace, do_render, visited_state_list=None, do_print=False):
    """
    This function runs am action trace in the environment to determine if it is successful and the abstract states
    visited states along the trace. The execution can also be rendered it for debugging purpose.
    Args:
        unwrapped_env: unwrapped env. without transformations
        env: wrapped env. with transformations applied
        replay_trace: the action trace to be executed
        do_render: True if execution shall be rendered, False otherwise
        visited_state_list: ignored if None, otherwise the function expects a list, and it adds the visited states to
        the list
        do_print: if True, the executed actions and info dictionaries return from steps are printed to the console
        (for debugging)

    Returns: pair (success, done) of Booleans where success indicates if the goal is reached by the execution, done
    indicates if a terminal reached is reached

    """
    global run_trace_steps
    env.reset()
    success = False
    for i, a in enumerate(replay_trace):
        # perform action in environment
        next_rl_state, reward, done, info = env.step(a)
        run_trace_steps += 1
        if do_render:
            env.render()
        # keep track of visited abstract states
        if visited_state_list is not None:
            visited_state_list.append(make_search_state(unwrapped_env, info, None))
        # print info to console
        if do_print:
            print(f"Action {i}: {a}")
            print(info)
            print(done)
        if done:
            if info['flag_get']:
                success = True
            break
    return success, done


def compute_additional_state(unwrapped_env):
    """
    A function for getting additional from the RAM that captures most of the information required to complete a stage.
    This includes Mario's position, speed, and momentum, and the positions of enemies.
    It is currently not used. We used it for debugging of the training setup, where we passed this information directly
    into the linear layers of the neural network. However, including the information does make learning a lot easier,
    so we do not include it in general.
    Args:
        unwrapped_env: environment without wrappers so that RAM can be accessed

    Returns: tensor containing additional of a shape so that the tensor can be easily concatenated to the tensors
    representing observations (game images)

    """
    additional_state = torch.zeros([4,1,84])
    additional_state[0,0,0] = unwrapped_env.ram[0x03ad]
    additional_state[0,0,1] = unwrapped_env.ram[0x03b8]
    additional_state[0,0,2] = unwrapped_env.ram[0x0705]
    additional_state[0,0,3] = unwrapped_env.ram[0x0433]
    additional_state[0,0,4] = unwrapped_env.ram[0x86]
    additional_state[0,0,5] = unwrapped_env.ram[0x9f]
    additional_state[0,0,6] = unwrapped_env.ram[0x87]
    additional_state[0,0,7] = unwrapped_env.ram[0x88]
    additional_state[0,0,8] = unwrapped_env.ram[0x89]
    additional_state[0,0,9] = unwrapped_env.ram[0x8a]
    additional_state[0,0,10] = unwrapped_env.ram[0x8b]
    additional_state[0,0,11] = unwrapped_env.ram[0x8c]
    return additional_state
