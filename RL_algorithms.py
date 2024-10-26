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
        :param observations_dict: dict in the format the output of the function "MIDI_coding.format_dataset_single_note_modulo_encoding"
        :return: None
        """
        timer_start = time.time()
        t_rewards_dict = {}
        t_actions_dict = {}
        t_transitions_dict = {}
        t_all_states = []
        for outer_key in observations_dict.keys():
            if outer_key not in t_all_states:
                t_all_states += [outer_key]
            occurrences = list(observations_dict[outer_key].values())
            occurrences.sort(reverse=True)
            highest_values = occurrences[:arcs_for_state:]
            highest_keys = []
            sum_highest_keys = 0
            t_actions_dict[outer_key] = []
            for inner_key in observations_dict[outer_key]:
                if observations_dict[outer_key][inner_key] in highest_values:
                    t_actions_dict[outer_key] += [inner_key]
                    if inner_key not in t_all_states:
                        t_all_states += [inner_key]
                    highest_keys += [inner_key]
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
        print("Environment constructed in " + str(calc_time) + " seconds")
        return None

class Agent:
    def __init__(self, policy_dict, env, value_func, q_func, horizon, time=0, discount_factor=1):
        self.policy = policy_dict          # Dict in the form of: {state: action}
        self.env = env                     # DeterministicEnv object
        self.V = value_func                # Dict in the form of: {state: E[return]}
        self.Q = q_func                    # Dict in the form of: {(state, action): E[return]}
        self.horizon = horizon
        self.t = time
        self.discount_factor = discount_factor

    def construct_agent_from_env(self):
        timer_start = time.time()
        t_policy = {}
        t_V = {}
        t_Q = {}
        for key in self.env.all_actions:
            t_policy[key] = self.env.all_actions[key][0]
            t_V[key] = random.uniform(0, 1)
        for key in self.env.all_rewards:
            t_Q[key] = random.uniform(0, 1)
        self.policy = t_policy
        self.V = t_V
        self.Q = t_Q
        timer_end = time.time()
        calc_time = timer_end - timer_start
        print("Agent constructed in " + str(calc_time) + " seconds")

    def standard_learning_rate(self):
        return 1 / (1 + self.t)

    def get_action(self):
        curr_state = self.env.state
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

    def epsilon_greedy_Q_action(self):
        p = self.standard_learning_rate()
        is_random = bernoulli(p)
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

    def general_TDT_learning_step(self, temporal_difference_target, is_epsilon_greedy=0):
        if is_epsilon_greedy == 1:
            action = self.epsilon_greedy_Q_action()
        else:
            action = self.greedy_Q_action()
        state = self.env.state
        alpha = self.standard_learning_rate()
        self.Q[(state, action)] = (1 - alpha) * self.Q[(state, action)] + alpha * temporal_difference_target
        self.t += 1
        self.env.step(action)

    def SARSA_step(self, is_epsilon_greedy=0):
        reward = self.get_reward()
        next_state = self.get_next_state()
        next_action = self.get_next_action()
        temporal_difference_target = reward + self.discount_factor * self.Q[(next_state, next_action)]
        self.general_TDT_learning_step(temporal_difference_target, is_epsilon_greedy)
        to_end = self.termination_check()
        if to_end != 0:
            for key in self.policy.keys():
                self.policy[key] = self.greedy_Q_action()
            return self.policy
        else:
            return 0

    def move_to_random_state(self):
        i = random.randint(0, len(self.env.all_states) - 1)
        return self.env.all_states[i]

    def generate_audio_sequence(self, in_samples=0, max_history=8):
        ret_tuple = ()
        history = []
        if in_samples == 0:
            samples = self.horizon
        else:
            samples = in_samples
        for time_step in range(samples):
            if isinstance(self.env.state, str):
                self.env.state = self.move_to_random_state()
            if self.policy[self.env.state] == self.env.state:
                self.env.state = self.move_to_random_state()
            if self.env.state in history:
                self.env.state = self.move_to_random_state()
            ret_tuple += self.env.state
            if len(history) <= max_history:
                history += [self.env.state]
            else:
                temp_var = history.pop(0)
                history += [self.env.state]
            action = self.policy[self.env.state]
            self.env.step(action)
        return ret_tuple
