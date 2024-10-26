import mido
import MIDI_IO as io
import MIDI_coding as code
import audio_tools as tools
import audio_metrics as metrics
import numpy as np
import RL_algorithms as RL
import pickle


mid_file = mido.MidiFile("piano.mid")
lst = io.vectorize_MIDI("piano.mid")
data = [code.format_dataset_single_note_modulo_encoding(item, 4, 0) for item in lst]

env = RL.DeterministicEnv([], {}, {}, {}, -1)

env.construct_env_from_observations_dict(data[0])
agent = RL.Agent({}, env, {}, {}, 100000)
agent.construct_agent_from_env()
policy = 0
while policy == 0:
    policy = agent.SARSA_step()

rrr = agent.generate_audio_sequence(100)
redo = code.decode_1d_non_modulo_vectorized_audio(rrr)
n = io.export_MIDI([redo], 180)

"""
g = tools.pseudo_scale_estimation(lst[0])
lst = [code.single_note_modulo_encoder(array) for array in lst]
d = metrics.generate_circle_of_fifth_distances()
dist = metrics.general_vector_modulo_12_metric(np.array([13,2,8,4]), [1,2,3,11])



modulo_12_states = [i for i in range(12)]
actions_5_7_for_state = {}
for state in modulo_12_states:
    actions_5_7_for_state[state] = [((state + 5) % 12), ((state - 5) % 12), ((state + 7) % 12), ((state - 7) % 12)]
rewards_for_5_7 = {}
for state in actions_5_7_for_state.keys():
    for action in actions_5_7_for_state:
       rewards_for_5_7[(state, action)] = 1
transitions_for_5_7 = {}
for state in actions_5_7_for_state.keys():
    for action in actions_5_7_for_state:
       transitions_for_5_7[(state, action)] = action
initial_state = -1
env = RL.DeterministicEnv(modulo_12_states, actions_5_7_for_state, rewards_for_5_7, transitions_for_5_7, initial_state)
a = 5
"""
