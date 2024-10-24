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

