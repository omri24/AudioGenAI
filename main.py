import mido
import MIDI_IO as io
import MIDI_coding as code
import audio_tools as tools
import audio_metrics as metrics

mid_file = mido.MidiFile("arctic.mid")
lst = io.vectorize_MIDI("arctic.mid")
g = tools.pseudo_scale_estimation(lst[0])
lst = [code.single_note_modulo_encoder(array) for array in lst]
d = metrics.generate_circle_of_fifth_distances()
a = 5
