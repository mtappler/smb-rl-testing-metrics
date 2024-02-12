from util import make_search_state, run_trace
import copy
import random
import math

# fixed weights for individual parts of the fitness of individuals
position_weight = 2
new_state_weight = 1.5
speed_weight = 1


class FitnessStats:
    """
    Class for storing statistics about the fitness of an individual (an action trace)
    """
    def __init__(self, visited_states, fitness, done):
        """
        Constructor for the storage class
        Args:
            visited_states: the abstract states visited by the individual
            fitness: its fitness
            done: Boolean value indicating if a terminal state was reached in the execution of the individual
        """
        self.visited_states = visited_states
        self.fitness = fitness
        self.done = done


def select(pool, summed_fitness):
    """
    Selection function of the genetic algorithm implementing the fuzzing. It performs a roulette-wheel-based selection.
    Args:
        pool: pool of individuals
        summed_fitness: the sum of all fitness values of all individual

    Returns: a selected individual

    """
    choice = random.random() * summed_fitness
    i = 0
    while choice > 0:
        choice -= pool[i][1].fitness
        i += 1
    return pool[i - 1]


def mutate(parent, mut_stop_prob, actions, max_duration, parent_done):
    """
    Mutation function for the genetic algorithm, which performs various types of mutation that are selected at random,
    including inserting, changing, changing, deleting, and appending action. Mutations are repeated with a configured
    probability, which determines mutation strength.
    Args:
        parent: the individual that shall be mutated
        mut_stop_prob: mutation strength, the probability with which repeated mutation is stopped
        actions: available actions
        max_duration: the maximum length of a mutation, e.g., up to max_duration many actions can be inserted
        parent_done: Boolean flag indicating if the individual under mutation reached a terminal state

    Returns: the mutated individual

    """
    offspring = copy.deepcopy(parent)
    stop = False
    # we repeat mutation until a biased coin flip tells us to stop
    while not stop:
        # if the individual reached a terminal state, we change its actions otherwise we just append actions
        # with that we favor individual that reach terminal states
        mutation_type = random.choice(range(3) if parent_done else [3])
        # select how many actions we want to mutate/insert
        frames = random.randint(1, max_duration)
        if mutation_type == 0:
            insert_action(offspring, actions, frames)
        if mutation_type == 1:
            change_action(offspring, actions, frames)
        if mutation_type == 2:
            delete_steps(offspring, frames)
        if mutation_type == 3:  # if parent was not done, it makes sense to append actions
            append_action(offspring, actions, frames)
        stop = random.random() < mut_stop_prob
    return offspring


def change_action(offspring, actions, frames):
    """
    Change actions in a trace.
    Args:
        offspring: the trace to be mutated
        actions: available actions
        frames: defines how many actions are changed

    Returns: None (change in place)

    """
    if len(offspring) - frames < 1:
        return
    # we need to make sure to not go beyond the end of the trace
    index = random.randint(0, len(offspring) - frames)
    chosen_action = random.choice(actions)
    # this could be a NOOP, but this should be ok
    for i in range(frames):
        offspring[index + i] = chosen_action


def append_action(offspring, actions, frames):
    """
    Append actions to a trace.
    Args:
        offspring: the trace to be mutated
        actions: available actions
        frames: defines how many actions are appended

    Returns: None (change in place)

    """
    chosen_action = random.choice(actions)
    for i in range(frames):
        offspring.append(chosen_action)


def delete_steps(offspring, frames):
    """
    Delete actions from a trace.
    Args:
        offspring: the trace to be mutated
        frames: defines how many actions are deleted

    Returns: None (change in place)

    """
    if len(offspring) - frames < 1:
        return
    # we need to make sure to not go beyond the end of the trace
    index = random.randint(0, len(offspring) - frames)
    del offspring[index:index + frames]


def insert_action(offspring, actions, frames):
    """
    Insert actions into a trace.
    Args:
        offspring: the trace to be mutated
        actions: available actions
        frames: defines how many actions are inserted

    Returns: None (change in place)

    """
    index = random.randint(0, len(offspring))
    chosen_action = random.choice(actions)
    for i in range(frames):
        offspring.insert(index, chosen_action)


def evaluate(unwrapped_env, env, individual, visited_states):
    """
    This function evaluates an individual by executing it in the environment and then computing the parts
    of the fitness, which are the final x position reached in the environment, the number of new states visited, and
    the length of the individual, which relates to how fast the individual reached the final x position.
    Args:
        unwrapped_env: unwrapped environment without transformation applied
        env: wrapped environment with transformation applied
        individual: individual/trace to be evaluated
        visited_states: previously visited states by others

    Returns: information for fitness computation

    """
    visited_states_ind = []
    success, done = run_trace(unwrapped_env, env, individual, do_render=False, visited_state_list=visited_states_ind,
                              do_print=False)
    final_x_position = visited_states_ind[-1][0]
    nr_new_states = 0
    # count the number of visited state
    for s in visited_states_ind:
        if s not in visited_states and s not in visited_states_ind[:1]:
            nr_new_states += 1
    # print("visited {} new states".format(nr_new_states))
    if nr_new_states == 0:  # special case that just repeats a previously found trace
        unnormalized_fitness = (0, 0, 0)
    else:
        unnormalized_fitness = (final_x_position, nr_new_states, len(visited_states_ind))
    # return all the information including whether a terminal state was reached
    return visited_states_ind, unnormalized_fitness, success, done


def normalize_fitness(pool):
    """
    This function computates a single fitness values for each individual from the information gathered by the
    evaluation.
    Args:
        pool: the set of all individual of a generation

    Returns: list of pairs comprising individuals and their fitness

    """
    # for normalization:
    # determine the minimum and maximum final x positions reached by every trace after executing them
    x_positions = [fitness_stats.fitness[0] for offspring, fitness_stats in pool]
    min_x_pos = min(x_positions)
    max_x_pos = max(x_positions)
    x_range = max_x_pos - min_x_pos
    # determine the maximum speed by individuals, i.e. the maximum final position divided by final x position
    if any([fitness_stats.fitness[0] != 0 for offspring, fitness_stats in pool]):
        max_speed = max([fitness_stats.fitness[0] / fitness_stats.fitness[2]
                         for offspring, fitness_stats in pool if fitness_stats.fitness[0] != 0])
    else:
        max_speed = 0
    # determine maximum number of newly visited states
    max_visited_states = max([fitness_stats.fitness[1] for offspring, fitness_stats in pool])
    normalized_pool = []
    # now perform the normalization for every individual by dividing the fitness components by their maximum values
    for offspring, fitness_stats in pool:
        speed_fitness = (speed_weight * (fitness_stats.fitness[0] / fitness_stats.fitness[2])
                         / max_speed) if fitness_stats.fitness[0] > 0 else 0
        if x_range == 0:
            pos_fitness = position_weight
        else:
            pos_fitness = position_weight * ((fitness_stats.fitness[0] - min_x_pos) / x_range)
        new_state_fitness = new_state_weight * (fitness_stats.fitness[1] / max_visited_states)
        # combine the weighted normalized values through summation
        full_fitness = speed_fitness + pos_fitness + new_state_fitness
        stats = FitnessStats(fitness_stats.visited_states,
                             full_fitness, fitness_stats.done)
        normalized_pool.append((offspring, stats))
    return normalized_pool


def crossover(parent1, parent2):
    """
    This function performs a point-wise crossover of two individuals, i.e. we choose a random index, cut the individuals
    at this index and concatenate the first part of one parent with second part of the second parent
    Args:
        parent1: one parent for the crossover
        parent2: one parent for the crossover

    Returns: a new individual

    """
    cross_point = random.randint(0, min(len(parent1), len(parent2)) - 1)
    offspring = parent1[:cross_point] + parent2[cross_point:]
    return offspring


def fuzz(unwrapped_env, env, seed, pool_size, generations, mut_stop_prob, actions, max_mutation_duration=15,
         truncation_selection_ratio=1, cross_over_prob=0.00,
         render_good_traces=False):
    """
    The main function for fuzzing of traces. It implements a genetic algorithm over a fixed number of generations
    with a fixed pool size.
    Args:
        unwrapped_env: unwrapped env. without transformations applied
        env:  wrapped env. with transformations applied
        seed: a seed trace which is usually the reference trace from the DFS for the goal
        pool_size: defines the number of individuals in a generation
        generations: the number of generations to be performed
        mut_stop_prob: mutation strength (inverse)
        actions:  available actions
        max_mutation_duration: defines how many actions are affected by a single mutation
        truncation_selection_ratio: defines how many of the best individuals of each generation shall be kept for
        selection to the next generation
        cross_over_prob: the probability with which crossover is performed
        render_good_traces: Boolean flag that determines if trace execution shall be rendered
         (debugging and demonstration)

    Returns: pairs containing all successful individual that were generation and the fittest of each generation

    """
    # init
    successful_traces = []
    best_traces = []
    try:
        print("Starting to fuzz")
        # evaluate the seed and initialize the pool with the seed
        visited_states_seed, fitness, cleared_level, done = evaluate(unwrapped_env, env, seed, set())
        pool = [(seed, FitnessStats(visited_states_seed, fitness, done))]
        pool = normalize_fitness(pool)
        summed_fitness = pool[0][1].fitness
        # add the seed to successful and best traces
        successful_traces = [seed] if cleared_level else []
        best_traces = [seed]
        visited_states = set(visited_states_seed)
        for i in range(generations):
            print(f"Fuzzing generation {i}")
            new_pool = []
            visited_states_in_gen = set()
            # creation of a new pool of individuals
            while len(new_pool) < pool_size:
                # first select a parent
                parent = select(pool, summed_fitness)
                # mutate/crossover the trace with a specific probability
                if len(pool) > 1 and random.random() < cross_over_prob:
                    parent1 = parent2 = None
                    while parent1 == parent2:
                        parent1 = select(pool, summed_fitness)
                        parent2 = select(pool, summed_fitness)

                    offspring = crossover(parent1[0], parent2[0])
                else:
                    parent_done = parent[1].done
                    offspring = mutate(parent[0], mut_stop_prob, actions, max_mutation_duration, parent_done)
                # evaluate the offspring from the mutation/crossover
                visited_states_off, fitness_off, cleared_level, done = evaluate(unwrapped_env, env, offspring,
                                                                                visited_states)
                # truncate to effective trace length, otherwise we get longer and longer traces
                # and the effects of mutations decrease, since uneffective parts of traces
                # may be mutated
                # visited_states_off is a list that contains all visited states, i.e., it may be shorter
                # than the offspring, if a terminal state is reached before executing all actions
                # thus we cut at reaching a terminal state
                offspring = offspring[0:len(visited_states_off)]
                new_pool.append((offspring, FitnessStats(visited_states_off, fitness_off, done)))

                # check if trace is successful and also if the trace visits any new state
                if done and new_pool[-1][1].fitness[1] > 0:
                    if cleared_level:
                        successful_traces.append(offspring)
                        print(f"Found {len(successful_traces)}-th successful trace")
                        if render_good_traces:
                            run_trace(unwrapped_env, env, offspring, do_render=True)

            # now we normalize the fitness values, i.e., we compute a single value for every traces
            pool = normalize_fitness(new_pool)
            # sort by fitness
            pool = sorted(pool, key=lambda elem: elem[1].fitness, reverse=True)
            # truncate pool
            new_pool_size = math.ceil(truncation_selection_ratio * pool_size)
            pool = pool[0:new_pool_size]
            visited_states_in_gen = set()
            # keep track of all visited states
            for _, stats in pool:
                visited_states_in_gen.update(stats.visited_states)

            visited_states.update(visited_states_in_gen)
            summed_fitness = sum([f_stat.fitness for _, f_stat in pool])
            if render_good_traces:
                run_trace(unwrapped_env, env, pool[0][0], do_render=True)
            best_traces.append(pool[0][0])
        return successful_traces, best_traces
    except e:
        # should not happen
        print(e)
        return successful_traces, best_traces
