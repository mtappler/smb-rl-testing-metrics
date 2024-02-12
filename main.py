import os
import datetime
import logging
import configparser
import pickle
import time
from pathlib import Path

import gym_super_mario_bros
import torch
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation
from gym_super_mario_bros import actions
from nes_py.wrappers import JoypadSpace
import numpy as np

import util
from fuzzing import fuzz
from search import search

import random
from metrics import MetricLogger, EvaluationLogger
from agent import Mario
from wrappers import ResizeObservation, SkipFrame
from util import run_trace_steps

from gym import Wrapper

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

log = logging.getLogger("FooBar")
log.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s | %(levelname)-10s | %(message)s")
handler = logging.StreamHandler()
handler.setFormatter(formatter)
handler.setLevel(logging.DEBUG)
log.addHandler(handler)
MAX_PRETRAIN_TRACES = 150

params = configparser.ConfigParser()
eval_logger = None

class StepCounterEnv(Wrapper):
    """
    A wrapper for gym environments that tracks how many steps are performed in every episode. This is used to track the
    number of steps performed for fuzzing.
    """
    def __init__(self, wrapped_env):
        super().__init__(wrapped_env)
        self.ep_counter = 0
        self.step_counter = 0
        self.steps_list = []
        self.current_steps_in_ep = 0

    def init_step_counter(self):
        """
        Initializes the counter.
        Returns: None

        """
        self.ep_counter = 0
        self.step_counter = 0
        self.steps_list = []
        self.current_steps_in_ep = 0

    def reset(self):
        """
        Records the reset of an environment, setting the steps in the current episode to zero and incrementing the
        episode counter.

        Returns: an observation from the wrapped environment

        """
        self.ep_counter += 1
        if self.current_steps_in_ep > 0:
             self.steps_list.append(self.current_steps_in_ep)
             self.current_steps_in_ep = 0
        return self.env.reset()
        
    def step(self,action):
        """
        Records a step in the environments and calls the wrapped environments
        Args:
            action: action to be performed

        Returns: observation from the wrapped environment

        """
        self.step_counter += 1
        self.current_steps_in_ep += 1
        return self.env.step(action)

    def write_steps_to_file(self):
        """
        Write the recorded steps to a file that tracks how many steps are performed during fuzzing. The file name is
        derived from the stage that is played.
        Returns:
            None
        """
        global params
        world = params.get('SETUP', 'STAGE')
        self.steps_list.append(self.current_steps_in_ep)
        with open(f"fuzz_data_{world}.txt", "w") as fp:
            fp.write(str(self.step_counter) + "\n")
            fp.write(str(self.ep_counter) + "\n")
            fp.write(str(self.steps_list) + "\n")


def main(eval_mode = False, params_file = None):
    """
    Main method for the whole DQfD with fuzzed demonstrations and for plain DDQ.

    Args:
        eval_mode: Boolean value indicating whether a saved agent shall just be evaluated rather than trained
        params_file: path to an ini-file containing the configuration for learning

    Returns: None

    """
    global params

    if params_file:
        params.read(params_file)
    else:
        params.read('params.ini')

    log_level = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR
    }

    # Set Log Level
    log.setLevel(log_level.get(params.get('LOGGING', 'LOG_LEVEL'), logging.DEBUG))

    # Initialize Super Mario environment
    env, unwrapped_env = setup_env()
    
    # Run the Training
    run(env, unwrapped_env,eval_mode)

def setup_env():
    """
    This function sets up the environment by instantiating the environment with the configured SMB stage and action
    space. It also wraps with several wrappers to skip a configured number of frames, performing a gray scale
    transformation, resizing and normalizing images, stacking 4 images, and to track step counts.

    Returns: a pair of wrapped environment, and unwrapped environment that allows for backtracking

    """
    global params
    
    env = gym_super_mario_bros.make(f"SuperMarioBros-{params.get('SETUP', 'STAGE')}-{params.get('SETUP', 'STYLE')}")
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
    env = action_space.get(params.get('SETUP', 'ACTION_SPACE'))
    
    # Apply Wrappers to environment
    env = SkipFrame(env, skip_min=3, skip_max=5)
    env = GrayScaleObservation(env, keep_dim=False)
    env = ResizeObservation(env, shape=84)
    env = TransformObservation(env, f=lambda x: x / 255.)
    env = FrameStack(env, num_stack=4)
    env = StepCounterEnv(env)

    env.reset()
    return env, unwrapped_env


def run(env, unwrapped_env, eval_mode):
    """
    The function for performing the training, either via DQfD or DDQ, depending on the configuration.
    Args:
        env: the environment wrapped with all transformation
        unwrapped_env: an environment that has the backtracking functionality
        eval_mode: a Boolean value indicating whether an existing agent shall evaluated, rather than training an agent

    Returns: None

    """
    # global variables for the configuration parameters and a logger
    global params
    global eval_logger

    # directory where neural networks and intermediate results and data are stored
    save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    save_dir.mkdir(parents=True)

    # instantiate the logger
    eval_logger = EvaluationLogger(save_dir, eval_mode)

    # potentially loading a checkpoint which is the neural network of a pretrained agent
    checkpoint_path = params.get('TRAINING', 'CHECKPOINT')
    checkpoint = Path(checkpoint_path) if checkpoint_path != 'None' else None
    load_only_conv = params.getboolean('TRAINING', 'LOAD_ONLY_CONV')

    if checkpoint and not checkpoint.exists():
        log.fatal("Checkpoint not found")
        exit(-1)

    # instatiate the mario agent with correct dimensions for the neural network 4 (stacked images) x 84 x 84
    # (downscaled image), all parameters and potentially initial neural netwoks
    mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir, params=params,
                  checkpoint=checkpoint,load_only_conv=load_only_conv)
    stage = params["SETUP"]["STAGE"]
    if eval_mode:
        evaluate_training(env, unwrapped_env, mario, 20)
    else:
        # file prefix for DQfD pretraining time measurements, only relevant for DQfD
        file_prefix = f"DQfD_{stage}"
        file_suffix = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')

        pretrain_steps = params.getint('TRAINING', 'PRETRAIN_STEPS')
        # if pretrain steps shall be performed, then set up the pretraining, i.e., perform search and fuzzing
        if pretrain_steps > 0:
            pretrain_start_overall = time.time()
            # "TRUE_EXPERT" performs DQfD with prerecorded demonstrations instead of fuzzed demonstrations
            if "TRUE_EXPERT" in params:
                fuzz_time = 0
                search_time = 0
                file_path = params['TRUE_EXPERT']['LOC']
                load_true_expert_memory_from_traces(mario, env,file_path)
            else:
                # DFS for the goal
                search_trace = run_search(env, unwrapped_env)            
                search_time = time.time() - pretrain_start_overall
            
                fuzz_start_time = time.time()
                # fuzzing of demonstrations with the initial trace leading to the goal as a seed
                fuzz_traces = run_fuzz(env, unwrapped_env, search_trace)
                fuzz_time = time.time() - fuzz_start_time

                load_expert_memory_from_traces(mario, env, fuzz_traces, dump=True)
            
            pretrain_start_time = time.time()
            pretrain_mario(mario, pretrain_steps)            
            pretrain_time = time.time() - pretrain_start_time

            # we write the time required for pretraining steps to a text file
            pretrain_overall_time = time.time() - pretrain_start_overall
            time_log = save_dir / f"{file_prefix}_pretrain_time_{file_suffix}"
            with open(time_log, "w") as f:
                f.write(f"search_time (sec): {search_time}\n")
                f.write(f"fuzz_time (sec): {fuzz_time}\n")
                f.write(f"pretrain_time (sec): {pretrain_time}\n")
                f.write(f"overall_pretrain_time (sec): {pretrain_overall_time}")
        # after potentially pretraining, we start the main training loop
        train_mario(mario, env, unwrapped_env)
    
def scale_reward(reward):
    """
    Scale the reward by a constant number. We divide it by 100 as we found that to improve performance in all
    configurations.
    Args:
        reward: unscaled reward

    Returns: scaled reward

    """
    return reward / 100

def load_expert_trace_file(file_path):
    """
    Load an expert traces, i.e., prerecorded traces of game play from the text file at file_path. The function assumes
    that the file contains one trace per line, where a trace is encoded as list of integers with each integer
    corresponding to an action of the RL agent.
    Args:
        file_path: path to the trace file

    Returns: a list of traces

    """
    import json
    with open(file_path, "r") as demo_file:
        lines = demo_file.readlines()
        traces = []
        for l in lines:
            trace = json.loads(l)
            traces.append(trace)

    print(f"Loaded {len(traces)} expert traces")
    return traces
        
def load_true_expert_memory_from_traces(mario, env, file_path):
    """
    Initialize the expert buffer of the given RL agent with prerecorded expert traces. We execute each in the
    environment and store the experiences from the execution.
    Args:
        mario: RL agent
        env: SMB gym environment
        file_path: path to expert traces

    Returns: None

    """
    log.debug("Loading true expert traces into expert-memory")
    trace_cnt = 0
    sample_traces = load_expert_trace_file(file_path)
    
    n_step_return = params.getint('TRAINING', 'N_STEP_RET')

    for trace in sample_traces:
        log.debug(f"Running with pre-computed trace {trace_cnt + 1}")

        # Reset environment
        state = env.reset()
        state = torch.from_numpy(np.array(state)).float()

        episode = []
        for action in trace:
            # perform an action in the SMB gym environment
            next_state, reward, done, info = env.step(action)
            reward = scale_reward(reward)
            next_state = torch.from_numpy(np.array(next_state)).float()
            # potentially add reward corresponding to gained coins
            reward = mario.compute_added_reward(info, reward, coin=params.getboolean('TRAINING', 'REWARD_COINS'),
                                                score=params.getboolean('TRAINING', 'REWARD_SCORE'))
            episode.append((state, next_state, action, reward, done))
            state = next_state
            if done:
                print(f"Win: {info['flag_get']}")
                break

        # for every executed episode fill the expert replay buffer (cache_expert) step by step with the collected
        # experiences and the corresponding n-step reward
        for i in range(len(episode)):
            (state, next_state, action, reward, done) = episode[i]
            if i + n_step_return < len(episode):
                last_state = episode[i+n_step_return-1][1] if not episode[i+n_step_return-1][4] else None
                n_rewards = (list(map(lambda step : step[3],episode[i:i+n_step_return])),last_state)
            else:
                last_state = None
                n_rewards = (list(map(lambda step : step[3],episode[i:])),last_state)
            mario.cache_expert(state, next_state, action, reward, n_rewards, done)

        trace_cnt += 1
        
def load_expert_memory_from_traces(mario, env, traces_to_learn,dump=False):
    """

    Initialize the expert buffer of the given RL agent with traces from fuzzing. We execute each in the
    environment and store the experiences from the execution.

    Args:
        mario: RL agent
        env: SMB environment
        traces_to_learn: list of traces to be used
        dump: Boolean value if the cache shall be stored as a pickle file

    Returns:

    """
    log.debug("Loading pre-computed trace into expert-memory")
    
    trace_cnt = 0
    
    n_step_return = params.getint('TRAINING', 'N_STEP_RET')
    # if there are too many fuzz traces, we discard some to avoid having imbalanced sets of experiences between
    # expert and normal experiences
    if len(traces_to_learn) > MAX_PRETRAIN_TRACES:
        sample_traces = random.sample(traces_to_learn,MAX_PRETRAIN_TRACES)
    else:
        sample_traces = traces_to_learn
    for trace in sample_traces:
        log.debug(f"Running with pre-computed trace {trace_cnt + 1}")

        # Reset environment
        state = env.reset()
        state = torch.from_numpy(np.array(state)).float()

        episode = []
        for action in trace:
            # perform an action in the SMB gym environment
            next_state, reward, done, info = env.step(action)
            reward = scale_reward(reward)
            next_state = torch.from_numpy(np.array(next_state)).float()
            # potentially add reward corresponding to gained coins
            reward = mario.compute_added_reward(info, reward, coin=params.getboolean('TRAINING', 'REWARD_COINS'),
                                                score=params.getboolean('TRAINING', 'REWARD_SCORE'))
            episode.append((state, next_state, action, reward, done))
            state = next_state
            if done:
                break

        # for every executed episode fill the expert replay buffer (cache_expert) step by step with the collected
        # experiences and the corresponding n-step reward
        for i in range(len(episode)):
            (state, next_state, action, reward, done) = episode[i]
            if i + n_step_return < len(episode):
                last_state = episode[i+n_step_return-1][1] if not episode[i+n_step_return-1][4] else None
                n_rewards = (list(map(lambda step : step[3],episode[i:i+n_step_return])),last_state)
            else:
                last_state = None
                n_rewards = (list(map(lambda step : step[3],episode[i:])),last_state)
            mario.cache_expert(state, next_state, action, reward, n_rewards, done)

        trace_cnt += 1
    if dump:
        # store as a pickle file
        mario.dump_expert_memory(params)
        
def run_search(env, unwrapped_env):
    """
    This function runs the DFS for the initial reference trace to the goal
    Args:
        env: wrapped environment with a transformation
        unwrapped_env: unwrapped environmen for backtracking

    Returns: list containing the reference trace to the goal

    """
    global params

    if params.getboolean('MODE_SEARCH', 'ENABLE'):
        log.info("Running Mode: SEARCH")

        # Load previously generated trace
        if params.getboolean('MODE_SEARCH', 'LOAD_SAVED_TRACE'):
            # Get path
            search_trace_load_path = params.get('MODE_SEARCH', 'LOAD_PATH')
            prev_search_trace = Path(search_trace_load_path)

            # Check if file exists
            if not prev_search_trace.exists():
                log.fatal("Previous search trace not found")
                exit(-1)

            log.info(f"Loading previously generated trace at {search_trace_load_path}")

            # Load saved path
            with open(prev_search_trace, 'rb') as file:
                search_trace = pickle.load(file)

        # Generate new path
        else:
            # Reset environment to start
            log.debug("Resetting environment")
            env.reset()

            # Run the algorithm
            log.debug("Generating search trace")
            search_trace = search(env, unwrapped_env)
            log.debug("Search trace generated")
            # run_trace(unwrapped_env, env, search_trace, True)

            # Save the generated trace
            if params.getboolean('MODE_SEARCH', 'SAVE_GENERATED_TRACE'):
                search_trace_save_path = params.get('MODE_SEARCH', 'SAVE_PATH')
                save_search_trace = Path(search_trace_save_path)
                save_search_trace.parent.mkdir(parents=True) if not save_search_trace.parent.exists() else None

                # store the trace also as a pickle file
                with open(save_search_trace, 'wb') as file:
                    pickle.dump(search_trace, file)
    else:
        return None

    return [search_trace]


def run_fuzz(env, unwrapped_env, search_trace):
    """
    This function runs the fuzzing of the traces.
    Args:
        env: wrapped environment with all transformations
        unwrapped_env: unwrapped environment for backtracking (not used but for consistency)
        search_trace: initial reference trace used as seed

    Returns: list of successful fuzzed demonstrations

    """
    global params

    # Extract search trace from list, this is necessary for other code parts to work
    search_trace = search_trace[0]
    # initialize the step counting wrapper
    env.init_step_counter()
    if params.getboolean('MODE_FUZZ', 'ENABLE'):
        log.info("Running Mode: FUZZ")
        # Load previously generated trace
        if params.getboolean('MODE_FUZZ', 'LOAD_SAVED_TRACE'):
            # Get path
            fuzz_trace_load_path = params.get('MODE_FUZZ', 'LOAD_PATH')
            prev_fuzz_trace = Path(fuzz_trace_load_path)

            # Check if file exists
            if not prev_fuzz_trace.exists():
                log.fatal("Previous fuzz trace not found")
                exit(-1)

            log.info(f"Loading previously generated trace at {fuzz_trace_load_path}")

            # Load saved path
            with open(prev_fuzz_trace, 'rb') as file:
                success_traces = pickle.load(file)
         
        else:
            # Generate search trace, which is necessary for fuzzing - if not exists
            if search_trace is None:
                log.debug("No previous search trace found, generating a new one")
                search_trace = search(env, unwrapped_env)
                log.debug("Search trace generated")

            # Reset environment to start
            log.debug("Resetting environment")
            env.reset()

            # Run the algorithm
            action_indexes = []
            action_meanings = env.get_action_meanings()
            for i in range(len(action_meanings)):
                meaning = action_meanings[i]
                if ('right' in meaning) or ('down' in meaning) or ('NOOP' in meaning):
                    action_indexes.append(i)
            print(action_indexes)
            fuzzing_generations = params.getint('MODE_FUZZ', 'GENERATIONS')
            # run the fuzzing with default parameters, such as a population size of 100
            success_traces, best_traces = fuzz(unwrapped_env, env, search_trace, 100, fuzzing_generations, 0.2,
                                               action_indexes, truncation_selection_ratio=0.25,
                                               cross_over_prob=0.15, render_good_traces=False)

            # Save the generated traces
            if params.getboolean('MODE_FUZZ', 'SAVE_GENERATED_TRACE'):
                fuzz_trace_save_path = params.get('MODE_FUZZ', 'SAVE_PATH')
                save_fuzz_trace = Path(fuzz_trace_save_path)
                save_fuzz_trace.parent.mkdir(parents=True) if not save_fuzz_trace.parent.exists() else None

                with open(save_fuzz_trace, 'wb') as file:
                    pickle.dump(success_traces, file)
            print("=="*40)
            print("Run trace steps after fuzzing: ", run_trace_steps)
            print("=="*40)
            # write sampling required by fuzzing to a file
            env.write_steps_to_file()
        return success_traces
    else:
        return None

def pretrain_mario(mario, runs):
    """
    Pretraining stage for DQfD, where we perform a configured number of minibatch updates of the neural network.
    Args:
        mario: RL agent
        runs: number of updates

    Returns: None

    """
    n_refresh_expert = params.getint('TRAINING', 'REFRESH_EXPERT') * 50
    for i in range(runs):
        mario.pretrain()
        if n_refresh_expert > 0 and i > 0 and i % n_refresh_expert == 0:
            mario.refresh_expert_cache()
        
def train_mario(mario, env, unwrapped_env):
    """
    Actual training of RL agent, i.e., training stage in DQfD and training in DDQ. I.e., we perform a configured number
    of episodes and update the neural networks in fixed intervals.
    Args:
        mario: RL agent
        env: environment with transformation
        unwrapped_env: unwrapped environment for API consistency

    Returns: None

    """
    global params

    # Read settings for RL mode
    do_render = params.getboolean("ADMIN", "RENDER")
    episodes = params.getint('TRAINING', 'EPISODES')
    n_step_return = params.getint('TRAINING', 'N_STEP_RET')
    # this setting allows for refreshing the expert, so that cached intermediate data is cleared
    # this may simply prevent errors from to accumulate
    # It is currently not used anymore
    refresh_expert_cache = params.getint('TRAINING', 'REFRESH_EXPERT')

    logger = MetricLogger(mario.save_dir)

    """
    MARIO RL MODE
    """

    log.debug("Starting RL Mode")

    # Start with episode 1
    # For better readability
    mario.curr_episode += 1

    # evaluate initial agent after pretraining if DQfD is used
    if params.getint('TRAINING', 'PRETRAIN_STEPS'):
        evaluate_training(env, unwrapped_env, mario, 20)
    
    # Train the model until max. episodes is reached
    while (e := mario.curr_episode) <= episodes:
        log.debug(f"Running episode {e}")

        # reset the environment
        state = env.reset()
        state = torch.from_numpy(np.array(state)).float()

        episode = []
        # Play the game!
        while True:
            if do_render:
                env.render()

            # Pick an action
            action = mario.act(state)

            # Perform action
            next_state, reward, done, info = env.step(action)
            next_state = torch.from_numpy(np.array(next_state)).float()

            # potentially add reward corresponding to gained coins
            reward = mario.compute_added_reward(info, reward, coin=params.getboolean('TRAINING', 'REWARD_COINS'),
                                                score=params.getboolean('TRAINING', 'REWARD_SCORE'))

            reward = scale_reward(reward)
            episode.append((state, next_state, action, reward, done))

            # Update state
            state = next_state

            # Check if end of game
            if done or info['flag_get']:
                break

        # after each episode, we compute n-step returns, because for this computations we need to look into the future
        # at each step
        for i in range(len(episode)):
            (state, next_state, action, reward, done) = episode[i]
            # Remember
            if i + n_step_return < len(episode):
                last_state = episode[i+n_step_return-1][1]
                n_rewards = (list(map(lambda step : step[3],episode[i:i+n_step_return])),last_state)
            else:
                last_state = None
                n_rewards = (list(map(lambda step : step[3],episode[i:])),last_state)
            mario.cache(state, next_state, action, reward, n_rewards, done)
            # Learn, i.e., perform a minibatch update in fixed intervals
            q, loss = mario.learn()

            # Log
            logger.log_step(reward, loss,q)

        # reinitialize expert cache (unused in experiments)
        if refresh_expert_cache > 0 and e % refresh_expert_cache == 0 and e > 0:
            mario.refresh_expert_cache()
        # finish the logging of an episode and start a new episode log
        logger.log_episode()

        if e % 20 == 0:
            logger.record(
                episode=e,
                epsilon=mario.exploration_rate,
                step=mario.curr_step
            )

        # save the neural network and some data
        if e % mario.save_every == 0:
            mario.save(params)

        # evaluate the current policy in the environment
        if e % mario.eval_every == 0:
            evaluate_training(env, unwrapped_env, mario, 20)

        mario.curr_episode += 1

    log.info("Training done, saving...")
    # final save of policy network
    mario.save(params)


def evaluate_training(env, unwrapped_env, mario, episodes=100):
    """
    Evaluation of a policy, i.e., execution of policy in the environment without exploration.
    Args:
        env: wrapped environment with transformation
        unwrapped_env: unwrapped env. for API consistency
        mario: RL agent
        episodes: how many episodes shall be performed for evaluation and averaging

    Returns: None (results are logged to a file)

    """
    global params
    global eval_logger

    assert isinstance(eval_logger, EvaluationLogger)

    # execute "episodes" many episodes in the environment with the given policy
    for ep in range(episodes):
        log.debug(f"Running eval cycle {ep + 1}")
        state = env.reset()
        state = torch.from_numpy(np.array(state)).float()

        # run till reaching a terminal state
        while True:
            action = mario.act(state, True)
            next_state, reward, done, info = env.step(action)
            next_state = torch.from_numpy(np.array(next_state)).float()
            eval_logger.log_step(reward)

            state = next_state
            if done or info['flag_get']:
                break

        eval_logger.log_episode()

    eval_logger.log_evaluation_cycle(mario, mario.exploration_rate)


if __name__ == '__main__':
    """
    Main of the python file, which expects an ini-file for configuration
    """
    import sys
    params_file = None
    
    for s in sys.argv:
        if ".ini" in s:
            params_file = s
        
    if "eval" in sys.argv:
        main(eval_mode = True, params_file = params_file)
    else:
        main(eval_mode = False, params_file = params_file)
