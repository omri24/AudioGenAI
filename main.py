import mido
import MIDI_IO as io
import MIDI_coding as code
import audio_tools as tools
import audio_metrics as metrics
import numpy as np
import RL_algorithms as RL
import classic_algorithms as c_algo
import sys
import time
import pandas as pd

use_command_line_parameters = 1
if use_command_line_parameters != 1:

    # Hyper-parameters:
    gen_or_fix_utils = "FIX"

    # For "GEN" and "FIX"
    algo_group = "RL"
    specific_algo = "SARSA"
    reference_file = "piano1.mid"
    horizon = 10 ** int("7")

    # For "FIX"
    file_to_fix = "single_notes_errors_piano_and_drums2.mid"
    correct_file = "single_notes_piano_and_drums2.mid"

    # For "SINGLE_NOTE":
    error_type = "1"
    input_file = "piano_and_drums2.mid"
    input_file_name = input_file[:input_file.index(".")]

    # For "STATISTICS":
    # Need to complete if wanted

else:       # use_command_line_parameters == 1
    gen_or_fix_utils = sys.argv[1]

    if gen_or_fix_utils.upper() in ["GEN", "FIX", "STATISTICS"]:
        algo_group = sys.argv[2]
        specific_algo = sys.argv[3]
        reference_file = sys.argv[4]
        if algo_group.upper() == "RL":
            horizon = 10 ** int(sys.argv[5])
        if algo_group.upper() == "CLASSIC":
            calculate_statistics = int(sys.argv[5])

    if gen_or_fix_utils.upper() in ["FIX", "STATISTICS"]:
        file_to_fix = sys.argv[6]
        correct_file = sys.argv[7]

    if gen_or_fix_utils.upper() == "SINGLE_NOTE":
        error_type = sys.argv[2]
        input_file = sys.argv[3]
        input_file_name = input_file[:input_file.index(".")]

if gen_or_fix_utils.upper() in ["GEN", "FIX"]:
    if gen_or_fix_utils.upper() == "GEN" and algo_group.upper() == "RL" and specific_algo.upper() == "SARSA":

        lst = io.vectorize_MIDI(reference_file)
        data = [code.format_dataset_single_note_optional_modulo_encoding(item, 4, 0) for item in lst]

        env = RL.DeterministicEnv([], {}, {}, {}, -1)
        env.construct_env_from_observations_dict(data[0], arcs_for_state=10)

        agent = RL.Agent(env, horizon=horizon)
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
        n = io.export_MIDI([decoded_generated_tuple], ticks_per_sixteenth=180)



    elif gen_or_fix_utils.upper() == "FIX" and algo_group.upper() == "RL" and specific_algo.upper() in ["SARSA", "SARSA_LAMBDA"]:

        fix_lst = io.vectorize_MIDI(file_to_fix, channel_filtering=0)
        single_notes_lst = [code.get_single_note_audio_from_multi_note_audio(item) for item in fix_lst]
        up_down_feature_lst_lst = [code.get_up_down_features_from_audio(item) for item in single_notes_lst]

        ref_lst = io.vectorize_MIDI(reference_file, channel_filtering=0)
        ref_data = [code.format_dataset_single_note_optional_modulo_encoding(item, 4, 0) for item in ref_lst]

        env = RL.DeterministicEnv([], {}, {}, {}, -1)
        env.construct_env_from_observations_dict(ref_data[0], arcs_for_state=10)

        agent = RL.Agent(env, horizon=horizon)
        agent.construct_agent_from_env()

        generated_tuple_untrained = agent.fix_audio(up_down_feature_lst_lst[0])
        decoded_generated_tuple = code.decode_1d_non_modulo_vectorized_audio(generated_tuple_untrained)
        n = io.export_MIDI([decoded_generated_tuple], ticks_per_sixteenth=180, file_name="output_untrained.mid")

        timer_start = time.time()
        policy = 0
        while policy == 0:
            if specific_algo.upper() == "SARSA":
                policy = agent.SARSA_step()
            else:    # specific_algo.upper() == "SARSA_LAMBDA"
                policy = agent.SARSA_lambda_step()
        timer_end = time.time()
        calc_time = timer_end - timer_start
        print("Agent trained in " + str(round(calc_time, 2)) + " seconds")

        timer_start = time.time()
        generated_tuple_trained = agent.fix_audio(up_down_feature_lst_lst[0])
        decoded_generated_tuple = code.decode_1d_non_modulo_vectorized_audio(generated_tuple_trained)
        n = io.export_MIDI([decoded_generated_tuple], ticks_per_sixteenth=180, file_name="output_trained.mid")
        timer_end = time.time()
        calc_time = timer_end - timer_start
        print("Agent generated audio in " + str(round(calc_time, 6)) + " seconds")

        correct_lst = io.vectorize_MIDI(correct_file, channel_filtering=0)
        single_notes_lst_corrected = [code.get_single_note_audio_from_multi_note_audio(item) for item in correct_lst]
        encoded_correct = [code.single_note_modulo_encoder(item) for item in single_notes_lst_corrected]
        shortest_sequence_len = min(len(encoded_correct[0]), len(generated_tuple_trained), len(generated_tuple_untrained))
        delta_correct_untrained = metrics.general_vector_modulo_12_metric(
            list(encoded_correct[0])[:shortest_sequence_len], list(generated_tuple_untrained)[:shortest_sequence_len])
        delta_correct_trained = metrics.general_vector_modulo_12_metric(
            list(encoded_correct[0])[:shortest_sequence_len], list(generated_tuple_trained)[:shortest_sequence_len])
        mean_delta_untrained = delta_correct_untrained / shortest_sequence_len
        mean_delta_trained = delta_correct_trained / shortest_sequence_len
        print("Mean delta between correct and untrained = " + str(mean_delta_untrained))
        print("Mean delta between correct and trained = " + str(mean_delta_trained))
        # Positive value of 'improvement' is what we want
        improvement = (mean_delta_untrained - mean_delta_trained) / mean_delta_untrained
        print("Improvement in % = " + str(round(improvement * 100, 2)))

    elif gen_or_fix_utils.upper() == "FIX" and algo_group.upper() == "CLASSIC" and specific_algo.upper() == "LIN_OPT":

        fix_lst = io.vectorize_MIDI(file_to_fix, channel_filtering=0)
        single_notes_lst = [code.get_single_note_audio_from_multi_note_audio(item) for item in fix_lst]
        up_down_feature_lst_lst = [code.get_up_down_features_from_audio(item, False) for item in single_notes_lst]

        ref_lst = io.vectorize_MIDI(reference_file, channel_filtering=0)
        ref_data = [code.format_dataset_single_note_optional_modulo_encoding(item, 4, 0) for item in ref_lst]

        lin_opt_obj = c_algo.LinOpt()
        if calculate_statistics == 1:
            lin_opt_obj.construct_basic_mat_from_dict_for_linear_optimal_estimator(ref_data[0], 0)
            lin_opt_obj.construct_data_statistics()
            lin_opt_obj.dump_statistics()
        else:
            lin_opt_obj.load_statistics()



if gen_or_fix_utils.upper() == "SINGLE_NOTE":
    if error_type.upper() == "NO_ERRORS":
        lst_raw = io.vectorize_MIDI(input_file, channel_filtering=0)
        data = [code.get_single_note_audio_from_multi_note_audio(item) for item in lst_raw]
        n = io.export_MIDI(data, ticks_per_sixteenth=180, file_name="single_notes_" + input_file_name +".mid")



    elif error_type.upper() == "1":
        lst_raw = io.vectorize_MIDI(input_file, channel_filtering=0)
        lst_single = [code.get_single_note_audio_from_multi_note_audio(item) for item in lst_raw]
        lst_errors = [code.apply_errors_for_single_note_audio(item, error_type=1) for item in lst_single]
        n = io.export_MIDI(lst_errors, ticks_per_sixteenth=180, file_name="single_notes_errors_" + input_file_name +".mid")

if gen_or_fix_utils.upper() == "STATISTICS":
    if algo_group.upper() == "RL" and specific_algo.upper() in ["SARSA", "SARSA_LAMBDA"]:

        helping_dict = {}

        # For SARSA_lambda - assumption is that convergence takes 10 ** 5 steps
        horizons = [10 ** 4, 2 * (10 ** 4), 4 * (10 ** 4), 6 * (10 ** 4), 8 * (10 ** 4), 10 ** 5]
        max_horizon = max(horizons)
        num_of_iterations = 1

        for item in horizons:
            helping_dict[item] = []

        for i in range(num_of_iterations):
            print("\nIteration " + str(i + 1) + "\n")

            fix_lst = io.vectorize_MIDI(file_to_fix, channel_filtering=0)
            single_notes_lst = [code.get_single_note_audio_from_multi_note_audio(item) for item in fix_lst]
            up_down_feature_lst_lst = [code.get_up_down_features_from_audio(item) for item in single_notes_lst]

            ref_lst = io.vectorize_MIDI(reference_file, channel_filtering=0)
            ref_data = [code.format_dataset_single_note_optional_modulo_encoding(item, 4, 0) for item in ref_lst]

            env = RL.DeterministicEnv([], {}, {}, {}, -1)
            env.construct_env_from_observations_dict(ref_data[0], arcs_for_state=10)

            agent = RL.Agent(env, horizon=(max_horizon + 1))  # Plus 1 - make sure max horizon executed
            agent.construct_agent_from_env()

            generated_tuple_untrained = agent.fix_audio(up_down_feature_lst_lst[0])
            decoded_generated_tuple = code.decode_1d_non_modulo_vectorized_audio(generated_tuple_untrained)

            correct_lst = io.vectorize_MIDI(correct_file, channel_filtering=0)
            single_notes_lst_corrected = [code.get_single_note_audio_from_multi_note_audio(item) for item in correct_lst]
            encoded_correct = [code.single_note_modulo_encoder(item) for item in single_notes_lst_corrected]

            policy = 0
            while policy == 0:
                if specific_algo.upper() == "SARSA":
                    policy = agent.SARSA_step()
                else:  # specific_algo.upper() == "SARSA_LAMBDA"
                    policy = agent.SARSA_lambda_step()
                if (agent.t + 1) in horizons:
                    generated_tuple_trained = agent.fix_audio(up_down_feature_lst_lst[0])
                    decoded_generated_tuple = code.decode_1d_non_modulo_vectorized_audio(generated_tuple_trained)
                    shortest_sequence_len = min(len(encoded_correct[0]), len(generated_tuple_trained),
                                                len(generated_tuple_untrained))
                    delta_correct_untrained = metrics.general_vector_modulo_12_metric(
                        list(encoded_correct[0])[:shortest_sequence_len],
                        list(generated_tuple_untrained)[:shortest_sequence_len])
                    delta_correct_trained = metrics.general_vector_modulo_12_metric(
                        list(encoded_correct[0])[:shortest_sequence_len],
                        list(generated_tuple_trained)[:shortest_sequence_len])
                    mean_delta_untrained = delta_correct_untrained / shortest_sequence_len
                    mean_delta_trained = delta_correct_trained / shortest_sequence_len
                    print("Mean delta between correct and untrained = " + str(mean_delta_untrained))
                    print("Mean delta between correct and trained = " + str(mean_delta_trained))
                    # Positive value of 'improvement' is what we want
                    improvement = (mean_delta_untrained - mean_delta_trained) / mean_delta_untrained
                    print("Improvement in % = " + str(round(improvement * 100, 2)))
                    helping_dict[agent.t + 1] = []
                    helping_dict[agent.t + 1].append(improvement)

            helping_df = pd.DataFrame.from_dict(helping_dict)
            helping_df.to_excel("output.xlsx")


