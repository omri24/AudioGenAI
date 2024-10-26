import numpy as np
import random
from scipy.stats import bernoulli

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
        self.all_states = states_list         # All possible states
        self.all_actions = actions_dict       # All possible actions for each state
        self.all_rewards = rewards_dict       # The reward for (state, action) pair
        self.transitions = transitions_dict   # The following state for (state, action) pair
        if initial_state == -1:
            i = random.randint(0, len(self.all_states) - 1)
            self.state = self.all_states[i]
        else:
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


class Agent:
    def __init__(self, policy_dict, env, value_func, q_func, horizon, time=0, discount_factor=1):
        self.policy = policy_dict          # Dict in the form of: {state: action}
        self.env = env                     # DeterministicEnv object
        self.V = value_func                # Dict in the form of: {state: E[return]}
        self.Q = q_func                    # Dict in the form of: {(state, action): E[return]}
        self.horizon = horizon
        self.t = time
        self.discount_factor = discount_factor

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
            action = self.greedy_Q_action
        state = self.env.state
        alpha = self.standard_learning_rate()
        self.Q[(state, action)] = (1 - alpha) * self.Q[(state, action)] + alpha * temporal_difference_target
        self.t += 1

    def SARSA_step(self, is_epsilon_greedy=0):
        reward = self.get_reward()
        next_state = self.get_next_state()
        next_action = self.get_next_action()
        temporal_difference_target = reward + self.discount_factor * self.Q[(next_state, next_action)]
        self.general_TDT_learning_step(temporal_difference_target, is_epsilon_greedy)