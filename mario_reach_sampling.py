from schedulers import PrismInterface, StormInterface
from MarioTD import MarioTD
from aalpy.utils import load_automaton_from_file
import gym_super_mario_bros
import os
import pickle
import resource
import configparser
import datetime
import torch
import numpy as np
from math import sqrt, log
import time
from pathlib import Path
import json
import sys
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation
from wrappers import ResizeObservation, SkipFrame
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros import actions
import random
from agent import Mario
from collections import defaultdict 

from schedulers import extract_coords
from scipy.stats import fisher_exact

max_rec = 0x20000
action_dim = 2

from neuron_coverage import CoverageInfo

def print_current_coverage_infos(all_coverage_infos):
    combined_infos = all_coverage_infos[0]
    for current_info in all_coverage_infos[1:]:
        combined_infos = combined_infos.combine(current_info)
    combined_infos.print_coverage()

# May segfault without this line. 0x100 is a guess at the size of each stack frame.
resource.setrlimit(resource.RLIMIT_STACK, [0x100 * max_rec, resource.RLIM_INFINITY])
sys.setrecursionlimit(max_rec)
from enum import Enum
class Verdict(Enum):
    PASS = 1
    FAIL = 2
    INCONC = 3

def save(x, path):
    with open(path, 'wb') as handle:
        pickle.dump(x, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load(file_name):
    if os.path.exists(file_name):
        with open(file_name, 'rb') as handle:
            return pickle.load(handle)
    else:
        return None

def setup_scheduler(model_name,rev_input_dict,target):
    #prism_int_name = model_file_name.replace(".dot", f"p_int_{target}.pickle")
    #interface = load(prism_int_name)
    #model_file_name = "storm/" + model_name + ".dot"
    # if mdp is None:
    #     model = load_automaton_from_file(model_file_name, "mdp")
    # else:
    #     model = mdp
    # print("Loaded MDP")
    interface = StormInterface(target, model_name,rev_input_dict)
    print("Scheduler initialized")
    scheduler = interface.scheduler
    return scheduler

def setup_reachability(model_name,rev_input_dict,target,stage,params):
    scheduler = setup_scheduler(model_name,rev_input_dict,target)
    mario = setup_mario(params)
    return mario, scheduler 

def setup_mario(params):
    stage = params["SETUP"]["STAGE"]
    style = params["SETUP"]["STYLE"]
    env = gym_super_mario_bros.make(f"SuperMarioBros-{stage}-{style}")
    # due to an episode limit, make in the above line returns TimeLimit environment,
    # so to get the mario environment directly, we need to unwrap
    unwrapped_env = env.env

    # Limit the action-space
    action_space = {
        'SIMPLE_MOVEMENT': JoypadSpace(env, actions.SIMPLE_MOVEMENT),
        'COMPLEX_MOVEMENT': JoypadSpace(env, actions.COMPLEX_MOVEMENT),
        'RIGHT_ONLY': JoypadSpace(env, actions.RIGHT_ONLY),
        'FAST_RIGHT': JoypadSpace(env, [['right','B'], ['right', 'A','B']])
    }
    env = action_space.get("FAST_RIGHT")
    
    # Apply Wrappers to environment
    env = SkipFrame(env, skip_min=3, skip_max=5)
    env = GrayScaleObservation(env, keep_dim=False)
    env = ResizeObservation(env, shape=84)
    env = TransformObservation(env, f=lambda x: x / 255.)
    env = FrameStack(env, num_stack=4)
    rev_act_map = {"right":0, "jump":1}
    mario = MarioTD(env,rev_act_map)
    return mario

def setup_test_schedulers(params, rev_input_dict,x_positions):
    def k_closest_label_coord(all_label_coords, x_pos, k=3):
        min_diff = 10**10
        min_oc = None
        labels_with_dists = [(o,(x,y),(x-x_pos)) for (o,(x,y)) in all_label_coords if x-x_pos > 0] # only want to go beyond
        labels_with_dists.sort(key = lambda x : x[2])

        return [(o,c) for (o,c,d) in labels_with_dists[:k]]
    def closest_label_coord(all_label_coords, x_pos):
        min_diff = 10**10
        min_oc = None
        for (o,c) in all_label_coords:
            x,y = c
            if abs(x-x_pos) < min_diff:
                min_diff = abs(x-x_pos)
                min_oc = (o,c)
        return min_oc
    model_name = params["TESTING"]["MODEL"]
    model_path = params["TESTING"]["MODEL_PATH"]
    lab_file_name = f"{model_path}{model_name}.lab"
    
    all_label_coords = get_all_positions(lab_file_name)
    schedulers = dict()     
    for x in x_positions:
        #closest = closest_label_coord(all_label_coords,x)
        closest = k_closest_label_coord(all_label_coords, x, k=1)[0]
        target, coord = closest
        scheduler = setup_scheduler(model_name,rev_input_dict,target)
        schedulers[x] = scheduler

    #model = load_automaton_from_file(model_file_name, "mdp")
    #all_labels = [s.output.replace("__game_over","").replace("__win","") for s in model.states]
    #all_labels.remove("Init")
    #all_label_coords =  [(o,extract_coords(o)) for o in all_labels]
    return schedulers

def setup_suts(params):
    suts_list = []
    suts_names = tuple(json.loads(params["TESTING"]["SUTs"]))
    print(suts_names)
    for sut_name in suts_names:
        save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
        import time
        time.sleep(2)
        save_dir.mkdir(parents=True,exist_ok=True)
        
        checkpoint_path = sut_name
        checkpoint = Path(checkpoint_path) 
        sut = Mario(state_dim=(4, 84, 84), action_dim=action_dim, save_dir=save_dir, params=params,
                  checkpoint=checkpoint,load_only_conv=False,disable_cuda=True)
        suts_list.append(sut)
    return suts_names,tuple(suts_list)

def run_test_prefix(scheduler,mario_td,x_goal):
    while True: 
        action = scheduler.get_input()
        if action is None:
            #print(f"Sampling a random action in {obs}")
            action = random.choice(["right","jump"])
        rl_state,obs = mario_td.step(action,render=False,return_state=True)
        if "game_over" in obs or "win" in obs:
            return False,rl_state
        x,y = extract_coords(obs)
        if x >= x_goal:
            break
        reached_state = scheduler.step_to(action, obs)
        if reached_state is None:
            scheduler.step_to_closest(action,obs)
    return True,rl_state

def run_single_test(scheduler,x_goal, sut, mario_td,test_len):
    scheduler.reset()
    mario_td.reset()
    succ, rl_state = run_test_prefix(scheduler,mario_td,x_goal)

    # while True: 
    #     action = scheduler.get_input()
    #     if action is None:
    #         #print(f"Sampling a random action in {obs}")
    #         action = random.choice(["right","jump"])
    #     rl_state,obs = mario_td.step(action,render=False,return_state=True)
    #     if "game_over" in obs:
    #         return Verdict.INCONC
    #     x,y = extract_coords(obs)
    #     if x >= x_goal:
    #         break
    #     reached_state = scheduler.step_to(action, obs)
    #     if reached_state is None:
    #         scheduler.step_to_closest(action,obs)
    if not succ:
        return Verdict.INCONC
    
    rl_state = torch.from_numpy(np.array(rl_state)).float()
    for i in range(test_len):
        action = sut.act(rl_state,eval_mode=True)
        rl_state,obs = mario_td.step(action,reverse=False,render=False,return_state=True)
       
        rl_state = torch.from_numpy(np.array(rl_state)).float()
        
        if "game_over" in obs:
            return Verdict.FAIL
        if "win" in obs:
            break
    return Verdict.PASS

def diff_test(f1, n1, f2, n2,eps):
    contingency_table = np.array([[f1,n1-f1],[f2,n2-f2]])
    res = fisher_exact(contingency_table, alternative='two-sided')
    if res[1] < eps:
        return True,res
    else:
        return False
    #if abs(f1 / n1 - f2 / n2) > ((sqrt(1 / n1) + sqrt(1 / n2)) * sqrt(0.5 * log(2 / eps))):
    #    return True
    #return False


def repeated_test_from_state(params,x_goal,scheduler,sut_names,suts,mario_td):
    max_tries = params.getint("TESTING","MAX_TRIES")
    eps = params.getfloat("TESTING","ALPHA")
    test_len = params.getint("TESTING","LENGTH")
    
    
    fails_sut_1 = 0
    succ_sut_1 = 0
    succ_sut_2 = 0
    fails_sut_2 = 0
    current_sut_index = 0
    
    for i in range(max_tries):
        if i % 20 == 0:
            print(f"Try: {i}")
        sut = suts[current_sut_index]
        verdict = run_single_test(scheduler,x_goal, sut, mario_td,test_len)
        if verdict == Verdict.INCONC:
            continue
        elif verdict == Verdict.PASS:
            if current_sut_index == 0:
                succ_sut_1 += 1
            else:
                succ_sut_2 += 1
            
            current_sut_index = (current_sut_index + 1) % 2
        else:
            if current_sut_index == 0:
                fails_sut_1 += 1
            else:
                fails_sut_2 += 1
            
            current_sut_index = (current_sut_index + 1) % 2
        # perform hoeffding test here
        n1 = fails_sut_1 + succ_sut_1
        n2 = fails_sut_2 + succ_sut_2
        diff_res = diff_test(fails_sut_1, n1,fails_sut_2,n2,eps)
        if n1 > 0 and n2 > 0 and diff_res:
            print("Stopping early")
            print(f"{fails_sut_1/n1} vs {fails_sut_2/n2}")
            return(i,fails_sut_1,n1,fails_sut_2,n2, diff_res[1])
    print("Similarly safe")
    #return(i,fails_sut_1,n1,fails_sut_2,n2)
    return(max_tries + 1,fails_sut_1,n1,fails_sut_2,n2)
    
    

def differential_testing(params,rev_input_dict):
    stage = params["SETUP"]["STAGE"]
    model_file_name = params["TESTING"]["MODEL"]
    x_positions = json.loads(params["TESTING"]["X_POSITIONS"])
    print(x_positions)
    for x in x_positions:
        print(x, type(x))
    
    schedulers = setup_test_schedulers(params,rev_input_dict, x_positions)
    mario_td = setup_mario(params)
    sut_names,suts = setup_suts(params)
    result_dict = dict()
    import random
    id = random.randint(0,10000)
    with open(f"test_result_{stage}_{id}.txt", "w") as fp:
        fp.write("Test mode: {params['TESTING']['TESTS']} \n")
        fp.write(str(sut_names))
        fp.write("\n")
        for x in schedulers.keys():
            print(f"Going to test for {x}")
            start = time.time()
            result = repeated_test_from_state(params,x,schedulers[x],sut_names,suts,mario_td)
            end = time.time()
            fp.write(str(result))
            fp.write("\n")
            fp.write(f"Time: {end-start}")
            fp.write("\n")
            fp.flush()
            result_dict[x] = result
    print(result_dict)

def get_all_positions(lab_file_name):
    with open(lab_file_name,"r") as lab_file:
        lab_lines = lab_file.readlines()
        header = lab_lines[1]
        position_labels = filter(lambda x : "pos" in x, header.split(" "))
        all_pos = []
        for l in position_labels:
            x,y = extract_coords(l)
            all_pos.append((l,(x,y)))
        return all_pos
    
def determine_targets(params):
    if params["TESTING"]["TESTS"].startswith("RANDOM") or params["TESTING"]["TESTS"].startswith("EQUI_DIST"):
        nr_tests = int(params["TESTING"]["TESTS"].replace("RANDOM(","").replace("EQUI_DIST(","").replace(")",""))
        model_name = params["TESTING"]["MODEL"]
        model_path = params["TESTING"]["MODEL_PATH"]
        lab_file_name = f"{model_path}{model_name}.lab"
        all_positions = get_all_positions(lab_file_name)
        all_x_positions = [x for (l,(x,y)) in all_positions]
        if params["TESTING"]["TESTS"].startswith("RANDOM"):
            return sorted(random.sample(all_x_positions,nr_tests))
        else:
            max_x_pos = max(all_x_positions)
            choices = list(range(1,nr_tests+1))
            points = [int(round((c/(nr_tests+1))*max_x_pos)) for c in choices] 
            return points
    elif params["TESTING"]["TESTS"].startswith("FROM_FILE"):
        test_file_name = params["TESTING"]["TESTS"].replace("FROM_FILE(","").replace(")","")
        with open(test_file_name, "r") as fp:
            test_positions_string = fp.readlines()[0]
            test_positions = json.loads(test_positions_string)
            x_positions = [x for [x,y] in test_positions]
            return x_positions        
    elif params["TESTING"]["TESTS"].startswith("FIXED"):
        x_list = params["TESTING"]["TESTS"].replace("FIXED(","").replace(")","")
        return json.loads(x_list)
   
def coverage_test_from_state(params,x_goal,scheduler,sut_names,suts,mario_td,coverage_infos):
    max_tries = params.getint("TESTING","MAX_TRIES")
    test_len = params.getint("TESTING","LENGTH")
    current_sut_index = 0
    successful_run = False
    
    action_choices = defaultdict(int)
    
    for i in range(max_tries):
        if i % 20 == 0:
            print(f"Try: {i}")
        scheduler.reset()
        mario_td.reset()
        succ, rl_state = run_test_prefix(scheduler,mario_td,x_goal)
        rl_state = torch.from_numpy(np.array(rl_state)).float()
        if succ:
            successful_run = True
            print(f"Successful cov check at try {i}")
            for j in range(len(sut_names)):
                sut = suts[j]
                name = sut_names[j]
                
                action = sut.act(rl_state,eval_mode=True)
                action_choices[action] += 1
                imp_value = sut.net.importance_value(rl_state)
                cov_info = sut.net.check_coverage(rl_state)
                coverage_infos[name] = (coverage_infos[name][0].combine(cov_info),cov_info,imp_value)
            break
    difference = max(action_choices.values()) / sum(action_choices.values()) if successful_run else 0
    return successful_run,difference

def safety_ratios_from_state(params,x_goal,scheduler,sut_names,suts,mario_td):
    max_tries = params.getint("TESTING","MAX_TRIES")
    test_len = params.getint("TESTING","LENGTH")
    nr_tests = params.getint("TESTING","N_TESTS")
    safety_results = defaultdict(list)

    for current_sut_index,sut_name in enumerate(sut_names):
        sut = suts[current_sut_index]
        print(f"Testing {sut_name}")
        for i in range(max_tries):
            if i % 20 == 0:
                print(f"Try: {i}")
            scheduler.reset()
            mario_td.reset()
            verdict = run_single_test(scheduler,x_goal, sut, mario_td,test_len)
            if verdict == Verdict.INCONC:
                continue
            else:
                safety_results[sut_name].append(verdict)
            if len(safety_results[sut_name]) == nr_tests:
                break
    results = {sut_name : (count_pass(verdicts) / len(verdicts),len(verdicts)) for sut_name, verdicts in safety_results.items()}
    return results

def count_pass(verdict_list) :
    return len([v for v in verdict_list if v == Verdict.PASS])
                    
def coverage_testing(params,rev_input_dict):
    stage = params["SETUP"]["STAGE"]
    model_file_name = params["TESTING"]["MODEL"]
    x_positions = determine_targets(params)
    mario_td = setup_mario(params)

    print(x_positions)
    for x in x_positions:
        print(x, type(x))
    schedulers = setup_test_schedulers(params,rev_input_dict, x_positions)
    sut_names,suts = setup_suts(params)
    result_dict = dict()
    coverage_infos = dict()
    for sut_name in sut_names:
        coverage_infos[sut_name] = (CoverageInfo([]),None,None)
    import random
    id = random.randint(0,10000)
    file_name_suffix = ""
    if params["TESTING"]["TESTS"].startswith("RANDOM"):
        file_name_suffix = "rand"
    elif params["TESTING"]["TESTS"].startswith("EQUI_DIST"):
        file_name_suffix = "eqd"
    elif "boundary" in params["TESTING"]["TESTS"]:
        file_name_suffix = "bp" 
    with open(f"coverage_result_immediate_safety/test_result_{stage}_{file_name_suffix}_{id}.txt", "w") as fp:
        fp.write(str(sut_names))
        fp.write("\n")
        for x in schedulers.keys():
            print(f"Going to test for {x}")
            start = time.time()
            succ,difference = coverage_test_from_state(params,x,schedulers[x],sut_names,suts,mario_td,coverage_infos)
            end = time.time()
            fp.write(f"Tested x-coordinate:{x}\n")
            for sut_name,cov_info in coverage_infos.items():
                agg_info, ind_info, imp_value = cov_info
                fp.write(sut_name)
                fp.write(":\n")
                fp.write(str(agg_info.compute_coverage()))
                fp.write("\n")
                fp.write(str(ind_info.compute_coverage()))
                fp.write("\n")
                fp.write(str(imp_value))
                fp.write("\n")
            fp.write(f"Conclusive:{succ}\n")
            fp.write(f"Difference:{difference}")
            fp.write("\n")
            fp.flush()
            result_dict[x] = (succ,difference)
    print(result_dict)

def safety_ratio_testing(params,rev_input_dict):
    stage = params["SETUP"]["STAGE"]
    model_file_name = params["TESTING"]["MODEL"]
    x_positions = determine_targets(params)
    mario_td = setup_mario(params)

    print(x_positions)
    for x in x_positions:
        print(x, type(x))
    schedulers = setup_test_schedulers(params,rev_input_dict, x_positions)
    sut_names,suts = setup_suts(params)
    result_dict = dict()

    import random
    id = random.randint(0,10000)
    file_name_suffix = ""
    if params["TESTING"]["TESTS"].startswith("RANDOM"):
        file_name_suffix = "rand"
    elif params["TESTING"]["TESTS"].startswith("EQUI_DIST"):
        file_name_suffix = "eqd"
    elif "boundary" in params["TESTING"]["TESTS"]:
        file_name_suffix = "bp"
    elif "FIXED" in params["TESTING"]["TESTS"]:
        file_name_suffix = params["TESTING"]["TESTS"].replace("[","_").replace("]","_")
        
    with open(f"safety_ratios_500_ep/test_result_{stage}_{file_name_suffix}_{id}.txt", "w") as fp:
        fp.write(str(sut_names))
        fp.write("\n")
        for x in schedulers.keys():
            print(f"Going to test for {x}")
            start = time.time()
            results = safety_ratios_from_state(params,x,schedulers[x],sut_names,suts,mario_td)
            end = time.time()
            fp.write(f"Tested x-coordinate:{x}\n")
            for sut_name,(ratio,n_conc_tests) in results.items():
                fp.write(sut_name)
                fp.write(":\n")
                fp.write(f"Safety ratio:{ratio}\n")
                fp.write(f"Conclusive tests:{n_conc_tests}\n")
            fp.flush()
    print("Done safety ratio testing")

def test_main(params_file):
    params = configparser.ConfigParser()
    params.read(params_file)
    
    rev_input_dict = {0 : "right", 1 : "jump"} # hard-coded for now
    if params["TESTING"]["TYPE"] == "coverage":
        coverage_testing(params,rev_input_dict)
    elif params["TESTING"]["TYPE"] == "safety_ratio":
        safety_ratio_testing(params,rev_input_dict)
    elif params["TESTING"]["TYPE"] == "differential":
        differential_testing(params,rev_input_dict)

def eval_single(mario_td,sut, n_eval, n_eval_succ):
    pairwise_eval_single(mario_td,sut,sut, n_eval, n_eval_succ)
    
    # rewards = []
    # imp_values = []
    # wins = []
    # overall_coverage = CoverageInfo([])
    # agg_coverages = []
    # for i in range(n_eval):
    #     if i % 1 == 0:
    #         print(f"Eval: {i}")
    #     single_reward = 0
    #     rl_state = mario_td.reset(return_state=True)

    #     coverage_trace = CoverageInfo([])
    #     imp_values_trace = []
    #     while True:
    #         rl_state = torch.from_numpy(np.array(rl_state)).float()
            
    #         imp_value = sut.net.importance_value(rl_state)
    #         cov_info = sut.net.check_coverage(rl_state)
    #         overall_coverage = overall_coverage.combine(cov_info)
    #         coverage_trace = coverage_trace.combine(cov_info)
    #         imp_values_trace.append(imp_value)
                
    #         action = sut.act(rl_state,eval_mode=True)
    #         rl_state,obs,rew = mario_td.step(action,reverse=False,render=False,return_state=True,return_reward=True)
    #         single_reward += rew
    #         if "game_over" in obs or "win" in obs:
    #             win = "win" in obs
    #             imp_values.append(imp_values_trace)
    #             agg_coverages.append(coverage_trace)
    #             wins.append(win)
    #             break
    #     rewards.append(single_reward)
    #     if sum(wins) >= n_eval_succ:
    #         break
    # mean_reward = sum(rewards)/len(rewards)
    # std_dev = sqrt(sum(map(lambda r : (r - mean_reward)**2,rewards)) / len(rewards))
    # return mean_reward, std_dev,agg_coverages,overall_coverage,imp_values,wins

def select_tc(mario_td,sut, n_select,n_episodes):
    imp_values_at_x_coords = defaultdict(int)
    x_coords_count = defaultdict(int)
    #wins = []
    #overall_coverage = CoverageInfo([])
    #agg_coverages = []
    for i in range(n_episodes):
        if i % 1 == 0:
            print(f"Select: {i}")
        rl_state = mario_td.reset(return_state=True)
        rl_state = torch.from_numpy(np.array(rl_state)).float()
        while True:
            action = sut.act(rl_state,eval_mode=False)
            rl_state,obs,rew = mario_td.step(action,reverse=False,render=False,return_state=True,return_reward=True)
            rl_state = torch.from_numpy(np.array(rl_state)).float()
            imp_value = sut.net.importance_value(rl_state)                
            x,y = extract_coords(obs)
            imp_values_at_x_coords[x] += imp_value
            x_coords_count[x] += 1
            
            if "game_over" in obs or "win" in obs:
                break

    avg_imps_at_x = []
    for x in imp_values_at_x_coords.keys():
        avg_imps_at_x.append((x,imp_values_at_x_coords[x] / x_coords_count[x]))
    avg_imps_at_x.sort(key = lambda xi: xi[1],reverse = True)
    print(avg_imps_at_x)
    return list(map(lambda xi: xi[0],avg_imps_at_x[:n_select])) 

    
def pairwise_eval_single(mario_td,actor_sut,eval_sut, n_eval, n_eval_succ):
    rewards = []
    imp_values = []
    wins = []
    overall_coverage = CoverageInfo([])
    agg_coverages = []
    for i in range(n_eval):
        if i % 1 == 0:
            print(f"Eval: {i}")
        single_reward = 0
        rl_state = mario_td.reset(return_state=True)

        coverage_trace = CoverageInfo([])
        imp_values_trace = []
        while True:
            rl_state = torch.from_numpy(np.array(rl_state)).float()
            
            imp_value = eval_sut.net.importance_value(rl_state)
            cov_info = eval_sut.net.check_coverage(rl_state)
            overall_coverage = overall_coverage.combine(cov_info)
            coverage_trace = coverage_trace.combine(cov_info)
            imp_values_trace.append(imp_value)
                
            action = actor_sut.act(rl_state,eval_mode=True)
            rl_state,obs,rew = mario_td.step(action,reverse=False,render=False,return_state=True,return_reward=True)
            single_reward += rew
            if "game_over" in obs or "win" in obs:
                win = "win" in obs
                imp_values.append(imp_values_trace)
                agg_coverages.append(coverage_trace)
                wins.append(win)
                break
        rewards.append(single_reward)
        if sum(wins) >= n_eval_succ:
            break
    mean_reward = sum(rewards)/len(rewards)
    std_dev = sqrt(sum(map(lambda r : (r - mean_reward)**2,rewards)) / len(rewards))
    return mean_reward, std_dev,agg_coverages,overall_coverage,imp_values,wins

def avg_imp_value(imp_values,wins):
    imps_all = []
    imps_wins = []
    
    for i,w in enumerate(wins):
        imp_values_trace = imp_values[i]
        imps_all.extend(imp_values_trace)
        if w:
            imps_wins.extend(imp_values_trace)
    avg_imp_all = sum(imps_all) / len(imps_all)
    avg_imp_win = sum(imps_wins) / len(imps_wins) if len(imps_wins) > 0 else 0
    
    std_dev_all = sqrt(sum(map(lambda i : (i - avg_imp_all)**2,imps_all)) / len(imps_all))
    std_dev_win = sqrt(sum(map(lambda i : (i - avg_imp_win)**2,imps_wins)) / len(imps_wins)) if len(imps_wins) > 0 else 0
    return avg_imp_all, std_dev_all,avg_imp_win,std_dev_win

def mean_agg_coverage(agg_coverages):
    agg_covs = []
    for ac in agg_coverages:
        agg_covs.append(ac.compute_coverage()[0])
    avg_cov = sum(agg_covs) / len(agg_covs)
    
    std_dev = sqrt(sum(map(lambda i : (i - avg_cov)**2,agg_covs)) / len(agg_covs))
    
    return avg_cov,std_dev

def select_important(params_file):
    params = configparser.ConfigParser()
    params.read(params_file)
    stage = params["SETUP"]["STAGE"]
    mario_td = setup_mario(params)
    sut_names,suts = setup_suts(params)
    n_select = params.getint("TESTING","N_SELECT")
    n_episodes = params.getint("TESTING","N_EPISODES")
    
    assert len(suts) == 1
    res = []
    sut = suts[0]
    selected_x = select_tc(mario_td,sut, n_select,n_episodes)
    print("Selected x values:")
    print(selected_x)
    




def eval_main(params_file):
    params = configparser.ConfigParser()
    params.read(params_file)
    stage = params["SETUP"]["STAGE"]
    mario_td = setup_mario(params)
    sut_names,suts = setup_suts(params)
    res = []
    for sut in suts:
        mean_reward, std_dev,agg_coverages,overall_coverage,imp_values,wins = eval_single(mario_td,sut,params.getint("TESTING","N_EVAL"),params.getint("TESTING","N_EVAL_SUCC"))
        avg_imp_all, std_dev_all,avg_imp_win,std_dev_win = avg_imp_value(imp_values,wins)
        mean_agg_cov, std_dev_agg_cov = mean_agg_coverage(agg_coverages)
        res.append((mean_reward, std_dev,agg_coverages,overall_coverage,imp_values,wins, avg_imp_all, std_dev_all,avg_imp_win,std_dev_win,mean_agg_cov, std_dev_agg_cov))
        
    with open(f"eval_result_{stage}.txt", "w") as fp:
        fp.write(str(sut_names))
        fp.write("\n")
        for i,r in enumerate(res):
            print(sut_names[i])
            mean_reward, std_dev,agg_coverages,overall_coverage,imp_values,wins, avg_imp_all, std_dev_all,avg_imp_win,std_dev_win,mean_agg_cov, std_dev_agg_cov = r
            fp.write(f"Reward: {mean_reward} +- {std_dev} \n")
            fp.write(f"Overall coverage: {overall_coverage.compute_coverage()} \n")
            fp.write(f"Avg. trace cov.: {mean_agg_cov} +- {std_dev_agg_cov} \n")
            fp.write(f"Avg. Imp. : {avg_imp_all} +- {std_dev_all} \n")
            fp.write(f"Avg. Imp. (win): {avg_imp_win} +-{std_dev_win} \n")
            fp.write(f"# wins: {sum(wins)} \n")
 
def pairwise_eval_main(params_file):
    params = configparser.ConfigParser()
    params.read(params_file)
    stage = params["SETUP"]["STAGE"]
    mario_td = setup_mario(params)
    sut_names,suts = setup_suts(params)
    res = []
    assert len(suts) == 2
    actor_sut = suts[0]
    eval_sut = suts[1]
    for sut in suts:
        mean_reward, std_dev,agg_coverages,overall_coverage,imp_values,wins = pairwise_eval_single(mario_td,actor_sut,eval_sut,
                                                                                                   params.getint("TESTING","N_EVAL"),params.getint("TESTING","N_EVAL_SUCC"))
        avg_imp_all, std_dev_all,avg_imp_win,std_dev_win = avg_imp_value(imp_values,wins)
        mean_agg_cov, std_dev_agg_cov = mean_agg_coverage(agg_coverages)
        res.append((mean_reward, std_dev,agg_coverages,overall_coverage,imp_values,wins, avg_imp_all, std_dev_all,avg_imp_win,std_dev_win,mean_agg_cov, std_dev_agg_cov))
        
    with open(f"pairwise_eval_result_{stage}.txt", "w") as fp:
        fp.write(str(sut_names))
        fp.write("\n")
        for i,r in enumerate(res):
            print(sut_names[i])
            mean_reward, std_dev,agg_coverages,overall_coverage,imp_values,wins, avg_imp_all, std_dev_all,avg_imp_win,std_dev_win,mean_agg_cov, std_dev_agg_cov = r
            fp.write(f"Reward: {mean_reward} +- {std_dev} \n")
            fp.write(f"Overall coverage: {overall_coverage.compute_coverage()} \n")
            fp.write(f"Avg. trace cov.: {mean_agg_cov} +- {std_dev_agg_cov} \n")
            fp.write(f"Avg. Imp. : {avg_imp_all} +- {std_dev_all} \n")
            fp.write(f"Avg. Imp. (win): {avg_imp_win} +-{std_dev_win} \n")
            fp.write(f"# wins: {sum(wins)} \n")    
    
def main(params_file):
    params = configparser.ConfigParser()
    params.read(params_file)
    
    stage = params["SETUP"]["STAGE"]
    model_name = params["TESTING"]["MODEL"]
    rev_input_dict = {0 : "right", 1 : "jump"}

    mario,scheduler = setup_reachability(model_name,rev_input_dict,"win",stage,params)
    
    render = False
    n_ep = 800
    for e in range(n_ep):
        scheduler.reset()
        obs = mario.reset()

        while True:
            action = scheduler.get_input()
            if action is None:
                print(f"Sampling a random action in {obs}")
                action = random.choice(["right","jump"])
            else:
                pass
                #print(action)
            obs = mario.step(action,render=render)
            reached_state = scheduler.step_to(action, obs)
            if reached_state is None:
                scheduler.step_to_closest(action,obs)
                #print(f"Scheduler undefined at {obs}")
                #break
            if "game_over" in obs or "win" in obs:
                print(obs)
                break
            
if __name__ == "__main__":
    
    import sys
    params_file = None
    
    for s in sys.argv:
        if ".ini" in s:
            params_file = s
    if "test" in sys.argv:
        test_main(params_file)
    elif "pairwise" in sys.argv:
        pairwise_eval_main(params_file)
    elif "eval" in sys.argv:
        eval_main(params_file)
    elif "select" in sys.argv:
        select_important(params_file)
    else:
        main(params_file)
