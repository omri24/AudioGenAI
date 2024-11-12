import mido
import MIDI_IO as io
import MIDI_coding as code
import audio_tools as tools
import audio_metrics as metrics
import numpy as np
import RL_algorithms as RL
import sys
import time


algo_group = sys.argv[1]
specific_algo = sys.argv[2]
gen_or_fix = sys.argv[3]
reference_file = sys.argv[4]
file_to_fix = sys.argv[5]

if algo_group.upper() == "RL" and specific_algo.upper() == "SARSA" and gen_or_fix.upper() == "GEN":

    lst = io.vectorize_MIDI(reference_file)
    data = [code.format_dataset_single_note_optional_modulo_encoding(item, 4, 0) for item in lst]

    env = RL.DeterministicEnv([], {}, {}, {}, -1)
    env.construct_env_from_observations_dict(data[0], arcs_for_state=7)

    agent = RL.Agent({}, env, {}, {}, {}, 10 ** 5)
    agent.construct_agent_from_env()

    timer_start = time.time()
    policy = 0
    while policy == 0:
        policy = agent.SARSA_step()

    timer_end = time.time()
    calc_time = timer_end - timer_start
    print("Agent trained in " + str(round(calc_time, 2)) + " seconds")

    generated_tuple = agent.generate_audio_sequence(100, 8, 2)
    decoded_generated_tuple = code.decode_1d_non_modulo_vectorized_audio(generated_tuple)
    n = io.export_MIDI([decoded_generated_tuple], 180)

elif algo_group.upper() == "RL" and specific_algo.upper() == "SARSA" and gen_or_fix.upper() == "FIX":

    fix_lst = io.vectorize_MIDI(file_to_fix)
    single_notes_lst = [code.get_single_note_audio_from_multi_note_audio(item) for item in fix_lst]
    up_down_feature_lst_lst = [code.get_up_down_features_from_audio(item) for item in single_notes_lst]

    ref_lst = io.vectorize_MIDI(reference_file)
    ref_data = [code.format_dataset_single_note_optional_modulo_encoding(item, 4, 0) for item in ref_lst]

    env = RL.DeterministicEnv([], {}, {}, {}, -1)
    env.construct_env_from_observations_dict(ref_data[0], arcs_for_state=7)

    agent = RL.Agent({}, env, {}, {}, {}, 10 ** 5)
    agent.construct_agent_from_env()

    timer_start = time.time()
    policy = 0
    while policy == 0:
        policy = agent.SARSA_step()

    timer_end = time.time()
    calc_time = timer_end - timer_start
    print("Agent trained in " + str(round(calc_time, 2)) + " seconds")

    generated_tuple = agent.fix_audio(up_down_feature_lst_lst[0])
    decoded_generated_tuple = code.decode_1d_non_modulo_vectorized_audio(generated_tuple)
    n = io.export_MIDI([decoded_generated_tuple], 180)


"""
else:
    lst = io.vectorize_MIDI(reference_file)
    data = [code.get_single_note_audio_from_multi_note_audio(item) for item in lst]
    n = io.export_MIDI(data, 180)
"""
