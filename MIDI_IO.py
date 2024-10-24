import numpy as np
import mido


def vectorize_MIDI(file_name, num_of_MIDI_notes=128):
    """
    Assuming only 4/4 songs in the MIDI file and that resolution of 1/16 is small enough
    :param file_name: the midi file to process
    :param num_of_MIDI_notes: number of MIDI notes in the format (usually 128)
    :return: list of numpy arrays, each represent a song from the MIDI file
    """
    ret_lst = []
    midi_file = mido.MidiFile(file_name)
    bpm = midi_file.ticks_per_beat / 4   # assuming the song is 4/4
    for i, track in enumerate(midi_file.tracks):
        track_array = -1
        note_messages = []
        for message in track:
            if message.type in ["note_on", "note_off"]:
                note_messages += [message]
        if len(note_messages) == 0:
            print("Track " + str(i) + " doesn't contain notes")
        else:
            list_for_array = [0 for i in range(num_of_MIDI_notes)]
            for message in note_messages:
                if message.time == 0:
                    if message.type == "note_on":
                        list_for_array[message.note] = 1
                    if message.type == "note_off":
                        list_for_array[message.note] = 0
                else:
                    block_length = message.time / bpm
                    if int(block_length) != block_length:
                        raise ValueError("Block_length must be a natural number")
                    vec_for_array = np.array(list_for_array)
                    vec_for_array = np.reshape(vec_for_array, (num_of_MIDI_notes, 1))
                    block_to_concat = np.concatenate([vec_for_array for i in range(int(block_length))], axis=1)
                    if isinstance(track_array, int):
                        track_array = block_to_concat
                    else:
                        track_array = np.concatenate([track_array, block_to_concat], axis=1)
                    if message.type == "note_on":
                        list_for_array[message.note] = 1
                    if message.type == "note_off":
                        list_for_array[message.note] = 0
        if isinstance(track_array, np.ndarray):
            ret_lst += [track_array]
    return ret_lst
