import mido
import MIDI_IO as io
import MIDI_coding as code
import audio_tools as tools
import audio_metrics as metrics
import numpy as np
import RL_algorithms as RL

mid_file = mido.MidiFile("Queen - Bohemian Rhapsody.mid")
lst = io.vectorize_MIDI("Queen - Bohemian Rhapsody.mid")
n = io.export_MIDI(lst)

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
