import numpy as np


def single_note_modulo_encoder(vectorized_midi):
    """
    Encodes a song according to note names, omitting data about the octave. Must 1 or 0 notes at each time slot
    :param vectorized_midi: one array from the output of "MIDI_IO.vectorize_MIDI"
    :return: np array of the encoding
    """
    sum_array_axis_0 = np.sum(vectorized_midi, axis=0)
    mask = sum_array_axis_0 > 1
    two_notes_in_one_time_slot = np.any(mask)
    if two_notes_in_one_time_slot:
        raise ValueError("In the midi input there are time slots in which 2 notes are being played - not supported")
    argmax_array = np.argmax(vectorized_midi, axis=0)
    mask = argmax_array == 0
    mask = mask * (-1)
    modulo_array = argmax_array % 12
    modulo_array = modulo_array + mask      # if at time slot t there are no notes, place -1 at this point
    return modulo_array


def vectorized_MIDI_to_modulu_array(vectorized_midi):
    """
    Apply modulo 12 logic (note name - omitting octave data) to the output of "MIDI_IO.vectorize_MIDI"
    :param vectorized_midi: one array from the output of "MIDI_IO.vectorize_MIDI"
    :return: np array, after applying modulo 12 logic to each column
    """
    modulo_12_array = np.zeros(shape=(12, vectorized_midi.shape[1]))
    for col in range(vectorized_midi.shape[1]):
        for row in range(vectorized_midi.shape[0]):
            if vectorized_midi[row, col] == 1:
                modulo_row = row % 12
                modulo_12_array[modulo_row, col] = 1
    return modulo_12_array


def format_dataset_single_note_modulo_encoding(vectorized_midi, samples_for_algo=4):
    """
    generate dataset, for generative model that generates 1 note for time slot (1/16)
    note: this function formats the MIDI to "feature vectors" and labels that are with the same dimensions
    :param vectorized_midi: one array from the output of "MIDI_IO.vectorize_MIDI"
    :param samples_for_algo: number of time slots (1/16) that are passed to the generative algorithm
    :return: a dictionary, each key is a tuple with n = look_back entries (curr state), and the value is a dict
             that maps next state and observed amount of occurrence
    """
    modulo_12_array = vectorized_MIDI_to_modulu_array(vectorized_midi)
    sum_ax_0 = np.sum(modulo_12_array, axis=0)
    ret_dict = {}
    for col in range(modulo_12_array.shape[1]):
        if col + 7 < modulo_12_array.shape[1]:
            curr_states = []
            next_states = []
            for offset in range(2 * samples_for_algo):
                t_curr_states = []
                t_next_states = []
                if sum_ax_0[col + offset] == 0:    # Handle the situation that no notes are played
                    if (offset > 0) and (offset < samples_for_algo):
                        t_curr_states = [item + [666] for item in curr_states]
                    elif (offset > samples_for_algo) and (offset < (2 * samples_for_algo)):
                        t_next_states = [item + [666] for item in next_states]
                    elif offset == 0:
                        t_curr_states += [[666]]
                    else:   # offset is samples_for_algo
                        t_next_states += [[666]]
                else:
                    for row in range(modulo_12_array.shape[0]):
                        if modulo_12_array[row, col + offset] == 1:
                            if offset == 0:
                                t_curr_states += [[row]]
                            elif offset == samples_for_algo:
                                t_next_states += [[row]]
                            elif offset > 0 and offset < samples_for_algo:
                                for item in curr_states:
                                    t_curr_states += [item + [row]]
                            else:    # offset > samples_for_algo and offset < 2 * samples_for_algo
                                for item in next_states:
                                    t_next_states += [item + [row]]
                if offset < samples_for_algo:
                    curr_states = t_curr_states
                else:
                    next_states = t_next_states
            next_states_dict = {}
            for item_list in next_states:
                item_tuple = tuple(item_list)
                if item_tuple not in next_states_dict.keys():
                    next_states_dict[item_tuple] = 1
                else:
                    next_states_dict[item_tuple] += 1
            for list_key in curr_states:
                tuple_key = tuple(list_key)
                if tuple_key not in ret_dict.keys():
                    ret_dict[tuple_key] = next_states_dict
                else:
                    for inner_key in next_states_dict.keys():
                        if inner_key not in ret_dict[tuple_key].keys():
                            ret_dict[tuple_key][inner_key] = next_states_dict[inner_key]
                        else:
                            ret_dict[tuple_key][inner_key] += next_states_dict[inner_key]
    return ret_dict





