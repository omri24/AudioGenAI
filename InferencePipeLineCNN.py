import random
import re
import time

from stable_baselines3 import PPO
import MIDI_IO as io
import MIDI_coding as code
import RL_algorithms as RL
from RL_algorithms import FiniteHorizonDDPEnv, get_top_actions, calculate_entropy
import mido
from mido import Message
from copy import deepcopy
from music21 import converter, key, note, stream, analysis, chord
from audio_tools import get_scale_notes, closest_note, remove_risky_notes, move_note_to_correct_octave, handle_semi_tones
from DL_TRAIN_TEST import MODEL_TYPE
# Configurations
reference_file = "piano1_ref.mid"
#file_to_fix = "single_notes_errors_piano_and_drums6.mid"
#correct_file = "single_notes_piano_and_drums6.mid"
#gen_or_fix = "fix"
model_to_use = "ppo_finite_horizon_500_steps_500000"
path_to_midi_files = "midi_files"

# Function to extract note tuple from midi - for CNN
def extract_note_pitches(midi_path, target_track =0, target_channel=3):
    note_pitches = []
    # Load the MIDI file
    try:
        mid = mido.MidiFile(midi_path)
    except IOError:
        print(f"Could not open MIDI file: {midi_path}")
        return []
    # Iterate through all messages in all tracks
    for track_idx, track in enumerate(mid.tracks):
        if track_idx == target_track:
            for msg in track:
                if msg.type == 'note_on' and msg.velocity > 0:
                    if msg.channel == target_channel:
                        note_pitches.append(msg.note)
    return tuple(note_pitches)

class BACK_MIDI():
    def __init__(self , generated_tuple, path, error_file, save_to_file, target_channel=3):
        # Audio generation params
        self.manual_offset = 0  # Semi-tone shifts up or down to the entire generated audio
        self.target_channel = target_channel     # Set this to the channel of the lead instrument
        self.file_name_for_inference = path
        self.file_name_back = r"C:\Users\DELL\Downloads\back (1).mid"
        self.p = 1 / 2   # Probability to change a note (error)
        self.error_range = 2   # Range of error
        self.squeeze_notes_to_same_octave = False
        self.move_notes_to_input_octave = True
        self.consider_user_playing_direction = True
        self.use_generated_tuple_for_correction = True  # not relevant for CNN
        self.auto_set_manual_offset = False  # ?
        self.force_key = False  # moves by one-two to stay in the key
        self.remove_risky_notes_from_key = False  # even though inside same key, less choice.
        self.consider_distance_from_back = True  # to make it sound good with back tracks
        self.relevant_channels_for_currently_playing = [11]  # Set this to the main instrument that present in the back
        self.generated_tuple = generated_tuple
        # Define a string that will be included in the (fixed) output file name
        self.fix_config_str = "-configs"
        if self.use_generated_tuple_for_correction:
            self.fix_config_str += "-gen1"
        else:
            self.fix_config_str += "-gen0"
        if self.move_notes_to_input_octave:
            self.fix_config_str += "-movToIn1"
        else:
            self.fix_config_str += "-movToIn0"
        if self.consider_user_playing_direction:
            self.fix_config_str += "-direction1"
        else:
            self.fix_config_str += "-direction0"
        if self.force_key:
            self.fix_config_str += "-forceKey1"
        else:
            self.fix_config_str += "-forceKey0"
        if self.remove_risky_notes_from_key:
            self.fix_config_str += "-delRisk1"
        else:
            self.fix_config_str += "-delRisk0"
        if self.consider_distance_from_back:
            self.fix_config_str += "-distBack1"
        else:
            self.fix_config_str += "-distBack0"
        if self.squeeze_notes_to_same_octave:
            self.fix_config_str += "-squeeze1"
        else:
            self.fix_config_str += "-squeeze0"
        if self.auto_set_manual_offset:
            self.fix_config_str += "-autoManualOffset1"
        else:
            self.fix_config_str += "-autoManualOffset0"
        # If 'force_key' and 'consider_distance_from_back', use 'consider_distance_from_back'
        if self.force_key and self.consider_distance_from_back:
            self.force_key = False
            print("!!! Can't use both 'force_key' and 'consider_distance_from_back' - 'force_key' disabled !!!")
        # Inference
        timer_start = time.time()
        print("Starting inference run")
        self.mid = mido.MidiFile(self.file_name_for_inference)

        # Estimate keys
        midi_input_only_back = converter.parse(self.file_name_back)
        # Filter only elements that are Notes or Chords (which have pitch)
        pitched_elements = midi_input_only_back.recurse().getElementsByClass([note.Note, chord.Chord])

        # Now analyze the key
        key_estimate_input_only_back = pitched_elements.analyze('key').name
        key_estimate_input_only_back_relative_eq = pitched_elements.analyze('key').relative.name

        midi_input_used_in_training = converter.parse(error_file)
        key_estimate_input_used_in_training = midi_input_used_in_training.analyze('key').name
        key_estimate_input_used_in_training_relative_eq = midi_input_used_in_training.analyze('key').relative.name
        #midi_generated = converter.parse("mid_saved_midi_file.mid")


        #key_estimate_generated = midi_generated.analyze('key').name

        key_notes_input_only_back = get_scale_notes(key_estimate_input_only_back)
        if self.remove_risky_notes_from_key:
            key_notes_input_only_back = remove_risky_notes(key_notes_input_only_back)

        # If selected - set the manual_offset automatically to eliminate scale difference
        if self.auto_set_manual_offset:
            lst_options_for_training_scale = [key_estimate_input_used_in_training, key_estimate_input_used_in_training_relative_eq]
            lst_options_for_user_backing_track = [key_estimate_input_only_back, key_estimate_input_only_back_relative_eq]
            for item in lst_options_for_training_scale:
                if re.search("major", item):
                    training_scale_major = item
                else:
                    training_scale_minor = item
            for item in lst_options_for_user_backing_track:
                if re.search("major", item):
                    backing_track_scale_major = item
                else:
                    backing_track_scale_minor = item
            note_map = {"C": 0, "C#": 1, "D-": 1, "D": 2,
                        "E-": 3, "E": 4, "F": 5, "F#": 6,
                        "G-": 6, "G": 7, "A-": 8, "A": 9,
                        "B-": 10, "B": 11, "C-": 11}

            training_scale_note_name = training_scale_major[:training_scale_major.index(" ")]
            backing_track_note_name = backing_track_scale_major[:backing_track_scale_major.index(" ")]

            distance_back_train = note_map[backing_track_note_name] - note_map[training_scale_note_name]
            if distance_back_train > 6:   # Case need to reduce instead of increase
                distance_back_train += -12

            self.manual_offset = distance_back_train

        # Remove unnecessary items from tuple (666) and make sure the distances are less than an octave
        generated_tuple_lst = list(generated_tuple)
        for idx, item in enumerate(generated_tuple_lst):
            if item > 127:   # Remove 666 and 999
                if idx > 0:
                    generated_tuple_lst[idx] = generated_tuple_lst[idx - 1]
                else:   # idx == 0
                    generated_tuple_lst[idx] = generated_tuple[0]
            if idx > 0 and self.squeeze_notes_to_same_octave:
                last_note = generated_tuple_lst[idx -1]
                if abs(last_note - item) > 11:
                    if item > last_note:
                        generated_tuple_lst[idx] += -12
                    else:  # last_note >= item
                        generated_tuple_lst[idx] += 12

        generated_tuple = tuple(generated_tuple_lst)

        # Fix generated tuple
        mid_fixed = deepcopy(self.mid)
        pointer_on_generated_tuple = 0

        for i, track in enumerate(self.mid.tracks):

            currently_playing = []
            backup_memory = []
            last_note_from_user = -1
            last_selected_note = -1

            # Backup memory must not be empty - fill it with some relevant note
            for temp_msg in enumerate(track):
                if temp_msg == "note_on" and msg.channel in self.relevant_channels_for_currently_playing:
                    backup_memory.append(msg.note)

            # Iterate over all messages
            for j, msg in enumerate(track):
                if msg.type == "note_on" and msg.channel == target_channel:
                    if msg.velocity > 0:
                        ending_message = Message('note_off', channel=msg.channel, note=msg.note,
                                                      velocity=msg.velocity, time=msg.time)
                        curr_end_idx = "not found"
                        for k, _msg in enumerate(track):
                            if _msg.type == "note_off":
                                #ending_message.time = _msg.time
                                #ending_message.velocity = _msg.velocity

                                if k > j and _msg.channel == target_channel and _msg.note == msg.note:
                                    curr_end_idx = k
                                    break
                        if isinstance(curr_end_idx, str):
                            print(f"During error correction found a note without 'note_off' message - will not be changed")
                            continue
                        if self.use_generated_tuple_for_correction:
                            note_to_use =  generated_tuple[pointer_on_generated_tuple]
                        else:
                            note_to_use = msg.note
                        # Handle direction
                        if self.consider_user_playing_direction:
                            if msg.note > last_note_from_user:
                                direction = 1
                            elif msg.note < last_note_from_user:
                                direction = -1
                            else:  # msg.note == last_note_from_user
                                direction = 0
                            # Find offset
                            note_to_use_mod_12 = note_to_use % 12
                            if note_to_use_mod_12 not in key_notes_input_only_back:
                                fixed_note_mod_12 = closest_note(note_to_use_mod_12, key_notes_input_only_back)
                                if not self.remove_risky_notes_from_key:
                                    fixed_note_mod_12 = handle_semi_tones(fixed_note_mod_12, [i[1] for i in currently_playing])
                                scale_offset = fixed_note_mod_12 - note_to_use_mod_12
                            else:  # Note in scale
                                scale_offset = 0
                            if note_to_use_mod_12 not in [i[1] for i in backup_memory]:
                                fixed_note_mod_12 = closest_note(note_to_use_mod_12, [i[1] for i in backup_memory])
                                back_offset = fixed_note_mod_12 - note_to_use_mod_12
                            else:  # Note in back
                                back_offset = 0
                            calc_offset = self.manual_offset + scale_offset * int(self.force_key) + back_offset * int(self.consider_distance_from_back)

                            if direction == 1 and calc_offset + note_to_use < last_selected_note:  # User went up but generated audio will go down
                                note_to_use += 12
                            elif direction == -1 and calc_offset + note_to_use > last_selected_note:  # User went down but generated audio will go ip
                                note_to_use += -12
                            else:  # direction == 0
                                note_to_use = last_selected_note
                        else:
                            direction = 1  # to not affect the offset

                        if self.move_notes_to_input_octave:
                            note_to_use = move_note_to_correct_octave(note_to_use, msg.note)   # Move to correct octave
                        else:
                            print("Inference without 'move_notes_to_input_octave' set to True (as currently executed) is not safe!")

                        note_to_use_mod_12 = note_to_use % 12
                        if note_to_use_mod_12 not in key_notes_input_only_back:
                            fixed_note_mod_12 = closest_note(note_to_use_mod_12, key_notes_input_only_back)
                            if not self.remove_risky_notes_from_key:
                                fixed_note_mod_12 = handle_semi_tones(fixed_note_mod_12, [i[1] for i in currently_playing])
                            scale_offset = fixed_note_mod_12 - note_to_use_mod_12
                        else:  # Note in scale
                            scale_offset = 0
                        if note_to_use_mod_12 not in [i[1] for i in backup_memory]:
                            fixed_note_mod_12 = closest_note(note_to_use_mod_12, [i[1] for i in backup_memory])
                            back_offset = fixed_note_mod_12 - note_to_use_mod_12
                        else:  # Note in back
                            back_offset = 0

                        if not self.consider_user_playing_direction or (self.consider_user_playing_direction and direction != 0):
                            pointer_on_generated_tuple += 1
                        msg_to_fixed = deepcopy(msg)
                        #msg_to_fixed.note += scale_offset * int(force_key)
                        #ending_message.note += scale_offset * int(force_key)
                        total_offset = (self.manual_offset + scale_offset * int(self.force_key) + back_offset * int(self.consider_distance_from_back)) * abs(direction)
                        new_note = int(round(note_to_use + total_offset))
                        new_note = max(0, min(127, new_note))  # Ensure it's within valid MIDI range
                        mid_fixed.tracks[i][j].note = new_note
                        mid_fixed.tracks[i][curr_end_idx].note = new_note
                        last_note_from_user = msg.note
                        last_selected_note = new_note
                        #key = (target_channel, mid_fixed.tracks[i][j].note % 12)
                        #if key not in currently_playing:
                        #    currently_playing.append(key)

                # Not in target channel, but it's note_on/note_off, also, avoid drums
                elif msg.type in ['note_on', 'note_off'] and msg.channel in self.relevant_channels_for_currently_playing:
                    key = (msg.channel, msg.note % 12)
                    if msg.type == 'note_on' and msg.velocity > 0:
                        if key not in currently_playing:
                            currently_playing.append(key)
                    else:  # note_off or note_on with velocity 0
                        if key in currently_playing:
                            currently_playing.remove(key)
                    if len(currently_playing) > 0:   # There are notes to relate to - update the memory
                        backup_memory = deepcopy(currently_playing)

        mid_fixed.save(f'{save_to_file}_inference_output_fixed{self.fix_config_str}.mid')
        timer_end = time.time()
        print(f"Inference succeed files saved, in {round(timer_end - timer_start, 2)} seconds")




