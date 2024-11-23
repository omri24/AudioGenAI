import numpy as np
import random
from scipy.stats import bernoulli
import time

class DeterministicEnv:
    def __init__(self, states_list, actions_dict, rewards_dict, transitions_dict, initial_state):
        if not isinstance(states_list, list):
            raise TypeError("In DeterministicEnv init, states_list must be a list, but currently is: " + str(type(states_list)))
        if not isinstance(actions_dict, dict):
            raise TypeError("In DeterministicEnv init, actions_dict must be a dict, but currently is: " + str(type(actions_dict)))
        if not isinstance(rewards_dict, dict):
            raise TypeError("In DeterministicEnv init, rewards_dict must be a dict, but currently is: " + str(type(rewards_dict)))
        if not isinstance(transitions_dict, dict):
            raise TypeError("In DeterministicEnv init, transitions_dict must be a dict, but currently is: " + str(type(transitions_dict)))
        if not isinstance(initial_state, (np.ndarray, int, float)):
            raise TypeError("In DeterministicEnv init, initial_state is of wrong type: " + str(type(rewards_dict)))
        self.all_states = states_list         # All possible states (that were observed)
        self.all_actions = actions_dict       # All possible actions for each state
        self.all_rewards = rewards_dict       # The reward for (state, action) pair
        self.transitions = transitions_dict   # The following state for (state, action) pair
        self.initial_state = initial_state
        self.state = initial_state

    def observe_next_state(self, action):
        if action not in self.all_actions[self.state]:
            ret_str = "Can't choose this action in the current state of the environment"
            return ret_str
        else:
            next_state = self.transitions[(self.state, action)]
            return next_state

    def step(self, action):
        self.state = self.observe_next_state(action)
        return None

    def construct_env_from_observations_dict(self, observations_dict, arcs_for_state=5):
        """
        Construct the environment
        :param observations_dict: dict in the format of the output of the function
                                  "MIDI_coding.format_dataset_single_note_optional_modulo_encoding"
        :return: None
        """
        timer_start = time.time()
        t_rewards_dict = {}
        t_actions_dict = {}
        t_transitions_dict = {}
        t_all_states = []
        t_all_states_set = set(t_all_states)
        for outer_key in observations_dict.keys():
            if outer_key not in t_all_states_set:
                t_all_states.append(outer_key)
                t_all_states_set.add(outer_key)
            occurrences = list(observations_dict[outer_key].values())
            occurrences.sort(reverse=True)
            highest_values = occurrences[:arcs_for_state:]
            highest_keys = []
            sum_highest_keys = 0
            t_actions_dict[outer_key] = []
            for inner_key in observations_dict[outer_key]:
                if observations_dict[outer_key][inner_key] in highest_values:
                    t_actions_dict[outer_key].append(inner_key)
                    if inner_key not in t_all_states_set:
                        t_all_states.append(inner_key)
                        t_all_states_set.add(inner_key)
                    highest_keys.append(inner_key)
                    sum_highest_keys += observations_dict[outer_key][inner_key]
            for inner_key in highest_keys:
                t_rewards_dict[(outer_key, inner_key)] = observations_dict[outer_key][inner_key] / sum_highest_keys
                t_transitions_dict[(outer_key, inner_key)] = inner_key
        self.all_states = t_all_states
        self.all_actions = t_actions_dict
        self.all_rewards = t_rewards_dict
        self.transitions = t_transitions_dict
        if self.initial_state == -1:
            i = random.randint(0, len(self.all_states) - 1)
            self.state = self.all_states[i]
        else:
            self.state = self.initial_state
        timer_end = time.time()
        calc_time = timer_end - timer_start
        print("Environment constructed in " + str(round(calc_time, 2)) + " seconds")
        return None

class Agent:
    def __init__(self, env, horizon, policy_dict={}, value_func={}, q_func={}, best_q={}, state_classes={}, log_dict={}, time=0, discount_factor=1):
        self.policy = policy_dict            # Dict in the form of: {state: action}
        self.env = env                       # DeterministicEnv object
        self.V = value_func                  # Dict in the form of: {state: E[return]}
        self.Q = q_func                      # Dict in the form of: {(state, action): E[return]}
        self.greedy_Q_val = best_q           # Dict in the form of: {state: [best action, it's Q]}
        self.state_classes = state_classes   # Dict in the form of: {class: list of states}
        self.log = log_dict
        self.horizon = horizon
        self.t = time
        self.discount_factor = discount_factor

    def standard_state_classification(self, state):
        ret_lst = []
        last_item = -1
        for idx, item in enumerate(state):
            if idx == 0:
                ret_lst.append(999)
            else:
                if item != 666:
                    if item > last_item:
                        ret_lst.append(1)
                    elif item < last_item:
                        ret_lst.append(-1)
                    else:   # item == last_item
                        ret_lst.append(0)
                else:     # item == 666
                    ret_lst.append(666)
            if item != 666:  # A note is played
                last_item = item
            else:  # No note is played - choose pseudo last_item in the middle of the MIDI notes spectrum
                last_item = 64
        return tuple(ret_lst)

    def construct_agent_from_env(self):
        timer_start = time.time()
        t_policy = {}
        t_V = {}
        t_Q = {}
        t_greedy_Q_val = {}
        t_classes = {}
        for key in self.env.all_actions.keys():
            t_V[key] = random.uniform(0, 1)
            curr_class = self.standard_state_classification(key)
            if curr_class not in t_classes.keys():
                t_classes[curr_class] = []
            t_classes[curr_class].append(key)
        for key in self.env.all_rewards:
            t_Q[key] = random.uniform(0, 1)
        self.V = t_V
        self.Q = t_Q
        self.state_classes = t_classes
        for key in self.env.all_actions.keys():
            t_policy[key] = self.greedy_Q_action_from_state(key)
            greedy_action = self.greedy_Q_action_from_state(key)
            t_greedy_Q_val[key] = [greedy_action, self.Q[(key, greedy_action)]]
        self.greedy_Q_val = t_greedy_Q_val
        self.policy = t_policy
        timer_end = time.time()
        calc_time = timer_end - timer_start
        print("Agent constructed in " + str(round(calc_time, 2)) + " seconds")

    def standard_learning_rate(self):
        return 1 / (1 + self.t)

    def get_action(self, force_greedy=1):
        curr_state = self.env.state
        if force_greedy == 1:
            return self.greedy_Q_val[self.env.state][0]
        else:
            return self.policy[curr_state]

    def get_reward(self):
        curr_state = self.env.state
        action = self.get_action()
        return self.env.all_rewards[(curr_state, action)]

    def get_next_state(self):
        action = self.get_action()
        return self.env.observe_next_state(action)

    def get_next_action(self):
        action = self.get_action()
        next_state = self.env.observe_next_state(action)
        return self.policy[next_state]

    def termination_check(self):
        if self.t < self.horizon:
            return 0
        else:
            return self.policy

    def greedy_Q_action_from_state(self, state):
        possible_actions_in_state = self.env.all_actions[state]
        opt_key = -1
        opt_Q_val = -1
        for action in possible_actions_in_state:
            curr_key = (state, action)
            if self.Q[curr_key] > opt_Q_val:
                opt_key = curr_key
                opt_Q_val = self.Q[curr_key]
        return opt_key[1]

    def greedy_Q_action(self):
        possible_actions_in_state = self.env.all_actions[self.env.state]
        opt_key = -1
        opt_Q_val = -1
        for action in possible_actions_in_state:
            curr_key = (self.env.state, action)
            if self.Q[curr_key] > opt_Q_val:
                opt_key = curr_key
                opt_Q_val = self.Q[curr_key]
        return opt_key[1]

    def fast_greedy_Q_action(self):
        return self.greedy_Q_val[self.env.state][0]

    def greedy_Q_action_considering_class(self, next_class):
        possible_actions_in_state = self.env.all_actions[self.env.state]
        opt_key = -1
        opt_Q_val = -1
        for action in possible_actions_in_state:
            if self.standard_state_classification(action) != next_class:   # action is next state
                continue
            curr_key = (self.env.state, action)
            if self.Q[curr_key] > opt_Q_val:
                opt_key = curr_key
                opt_Q_val = self.Q[curr_key]
        if isinstance(opt_key, int):    # No relevant state according to class
            return -1
        else:
            return opt_key[1]

    def epsilon_greedy_Q_action(self):
        p = self.standard_learning_rate()
        is_random = bernoulli.rvs(p)
        possible_actions_in_state = self.env.all_actions[self.env.state]
        if is_random == 0:
            opt_key = -1
            opt_Q_val = -1
            for action in possible_actions_in_state:
                curr_key = (self.env.state, action)
                if self.Q[curr_key] > opt_Q_val:
                    opt_key = curr_key
                    opt_Q_val = self.Q[curr_key]
            return opt_key[1]
        else:
            i = random.randint(0, len(possible_actions_in_state) - 1)
            return possible_actions_in_state[i]

    def fast_epsilon_greedy_Q_action(self):
        p = self.standard_learning_rate()
        is_random = bernoulli.rvs(p)
        if is_random == 0:
            return self.greedy_Q_val[self.env.state][0]
        else:
            possible_actions_in_state = self.env.all_actions[self.env.state]
            i = random.randint(0, len(possible_actions_in_state) - 1)
            return possible_actions_in_state[i]


    def general_TDT_learning_step(self, temporal_difference_target, is_epsilon_greedy=1):
        if is_epsilon_greedy == 1:
            action = self.fast_epsilon_greedy_Q_action()
        else:
            action = self.fast_greedy_Q_action()
        state = self.env.state
        alpha = self.standard_learning_rate()
        self.Q[(state, action)] = (1 - alpha) * self.Q[(state, action)] + alpha * temporal_difference_target
        if self.t not in self.log.keys():
            self.log[self.t] = []
        self.log[self.t].append((state, action))
        self.log[self.t].append(self.Q[(state, action)])
        if self.Q[(state, action)] > self.greedy_Q_val[state][1]:
            self.greedy_Q_val[state] = [action, self.Q[(state, action)]]
            self.policy[state] = action
        self.t += 1
        if self.t % 100000 == 0:
            print("Completed " + str(self.t / 1000) + " * 10^3 time steps")
        self.env.step(action)

    def SARSA_step(self, is_epsilon_greedy=1):
        reward = self.get_reward()
        next_state = self.get_next_state()
        next_action = self.get_next_action()
        temporal_difference_target = reward + self.discount_factor * self.Q[(next_state, next_action)]
        self.general_TDT_learning_step(temporal_difference_target, is_epsilon_greedy)
        to_end = self.termination_check()
        if to_end != 0:
            return self.policy
        else:
            return 0

    def TD_lambda_step(self, is_epsilon_greedy=1):
        if is_epsilon_greedy == 1:
            action = self.fast_epsilon_greedy_Q_action()
        else:
            action = self.fast_greedy_Q_action()
        reward = self.get_reward()
        next_state = self.get_next_state()
        next_action = self.get_next_action()
        concat_state = (self.env.state, self)
        delta = reward + self.discount_factor * self.Q

    def move_to_random_state(self):
        i = random.randint(0, len(self.env.all_states) - 1)
        return self.env.all_states[i]

    def generate_audio_sequence(self, in_samples=0, max_history=8, enhance=1):
        ret_tuple = ()
        history = []
        if in_samples == 0:
            samples = self.horizon
        else:
            samples = in_samples
        for time_step in range(samples):
            if isinstance(self.env.state, str):
                raise ValueError("Policy learned illegal action")
            if self.policy[self.env.state] == self.env.state:
                print("State s == s' occurred in s = " + str(self.env.state))
                self.env.state = self.move_to_random_state()
            if self.env.state in history:
                print("state s in history occurred in s = " + str(self.env.state))
                self.env.state = self.move_to_random_state()

            if enhance == 1:
                mode = random.randint(0, 7)
                if mode < 2:
                    ret_tuple += self.env.state
                elif mode > 7:
                    ret_tuple += self.env.state
                    ret_tuple += self.env.state
                    ret_tuple += self.env.state
                    ret_tuple += self.env.state
                else:
                    ret_tuple += self.env.state
                    ret_tuple += self.env.state

            elif enhance == 2:
                if len(ret_tuple) % 16 == 12:
                    ret_tuple += self.env.state
                    ret_tuple += self.env.state
                    ret_tuple += self.env.state
                    ret_tuple += self.env.state
                elif len(ret_tuple) % 16 == 13:
                    ret_tuple += self.env.state
                    ret_tuple += self.env.state
                    ret_tuple += self.env.state
                elif len(ret_tuple) % 16 == 14:
                    ret_tuple += self.env.state
                    ret_tuple += self.env.state
                elif len(ret_tuple) % 16 == 15:
                    ret_tuple += self.env.state
                else:
                    ret_tuple += self.env.state
                    ret_tuple += self.env.state

            else:
                ret_tuple += self.env.state
                ret_tuple += self.env.state
            if len(history) <= max_history:
                history += [self.env.state]
            else:
                temp_var = history.pop(0)
                history += [self.env.state]
            action = self.policy[self.env.state]
            self.env.step(action)
        return ret_tuple

    def get_random_state_from_class(self, desired_class):
        if desired_class in self.state_classes.keys():
            optional_states = self.state_classes[desired_class]
            idx = random.randint(0, len(optional_states) - 1)
            return optional_states[idx]
        else:
            return -1

    def fix_audio(self, up_down_feature_lst):
        ret_tuple = ()
        len_of_env_state = len(self.env.state)

        initial_class = self.standard_state_classification(tuple(up_down_feature_lst[:len_of_env_state]))
        try_getting_random_state = self.get_random_state_from_class(initial_class)
        if not isinstance(try_getting_random_state, int):
            self.env.state = try_getting_random_state
        for idx, item in enumerate(up_down_feature_lst):
            if idx % len_of_env_state == 0 and (idx - 1 + len_of_env_state) < len(up_down_feature_lst): # Condition to generate new state
                if idx == 0:
                    ret_tuple += self.env.state
                else:
                    target_class = self.standard_state_classification(tuple(up_down_feature_lst[idx:idx + len_of_env_state]))
                    try_epsilon_greedy = self.greedy_Q_action_considering_class(target_class)
                    if isinstance(try_epsilon_greedy, int):     # No possible next state that fits target_class
                        try_getting_random_state = self.get_random_state_from_class(target_class)
                        if isinstance(try_getting_random_state, int):
                            ret_tuple += self.move_to_random_state()
                        else:
                            ret_tuple += try_getting_random_state
                    else:
                        ret_tuple += try_epsilon_greedy   # The epsilon greedy attempt succeeded
        return ret_tuple
