import os
import MIDI_IO as io
import MIDI_coding as mc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split
from collections import defaultdict
import random


def one_hot_encode_output(output_vector, file_name, vector_shape = 3):
    if(vector_shape==3):
        if isinstance(output_vector, torch.Tensor):
            out = output_vector.to("cpu").numpy().transpose(0, 2, 1)[0]
        elif isinstance(output_vector, np.ndarray):
            out = output_vector.transpose(0, 2, 1)[0]
        elif isinstance(output_vector, list):
            out = np.array(output_vector).transpose(0, 2, 1)[0]
        else:
            raise TypeError(f"Unsupported type: {type(output_vector)}")
        decoded_array = []
        for a in out:
            argmax = np.argmax(a)
            one_hot_vect = [1 if argmax == i else 0 for i in range(128)]
            decoded_array.append(one_hot_vect)
    elif(vector_shape==1):
        decoded_array = [[1 if note == i else 0 for i in range(128)] for note in output_vector]
    else:
        raise ValueError(f"Unsupported vector shape: {vector_shape}")
    decoded_array = np.array(decoded_array).T
    io.export_MIDI(np.array([decoded_array]), f"{file_name}.mid")

class create_dataset():
    def __init__(self, path, length, output_file):
        i = 0
        self.dataset_data = []
        self.dataset_label=[]
        self.song_name=[]
        for fname in os.scandir(path):
            if (os.path.splitext(fname)[1] in (".midi", ".MID", ".mid") ):
                song_name = os.path.splitext(fname)[0].split('\\')[-1].replace("_","")
                print(f"in new song - {song_name}")
                np_array = np.array(io.vectorize_MIDI(fname))
                for t in np_array:
                    for i in range(0,t.shape[1]-length, length//20):
                        song = t[:, i:i + length]
                        if (len(np.unique(np.argmax(song, axis=0))) > 3):
                            for j in range(5):
                                new_song = mc.apply_errors_for_single_note_audio(song, p=0.5)
                                self.dataset_data.append(new_song)
                                self.dataset_label.append(song.argmax(axis=0))
                                self.song_name.append(f"{song_name}_{i}_{j}")
        print("dataset length", len(self.dataset_data))
        self.save_dataset("dataset", output_file, self.dataset_data, self.dataset_label, self.song_name)
    def save_dataset(self, folder_path, file_name, data, label, song_name):
        os.makedirs(folder_path, exist_ok=True)
        torch.save({"data": data, "label": label, "song_name":song_name}, os.path.join(folder_path, f"{file_name}.pth"))
    def save_dataset_as_midi(self, folder_path):
        for i, song_name in enumerate(self.song_name):
            io.export_MIDI(np.array([self.dataset_data[i]]), f"{folder_path}{song_name}_error.midi")
            one_hot_encode_output(output_vector=self.dataset_label[i], file_name=f"{folder_path}{song_name}_original.midi", vector_shape=1)

class load_dataset(Dataset):
    def __init__(self, file_path):
        checkpoint = torch.load(file_path, weights_only=False)
        self.data=checkpoint["data"]
        self.label=checkpoint["label"]
        self.song_name = checkpoint["song_name"]

    def split_train_validation(self, train_percent, validation_percent=0, seed=None):
        song_to_indices = defaultdict(list)
        for idx, name in enumerate(self.song_name):
            song_to_indices[name.split("_")[0]].append(idx)
        all_song_names = list(song_to_indices.keys())
        print(all_song_names)
        if seed is not None:
            random.seed(seed)
        random.shuffle(all_song_names)
        train_size = int(len(all_song_names) * train_percent)
        validation_size=int(len(all_song_names) * validation_percent)
        train_songs = set(all_song_names[:train_size])
        val_songs = set(all_song_names[train_size:train_size + validation_size])
        train_indices = [idx for name in train_songs for idx in song_to_indices[name]]
        val_indices = [idx for name in val_songs for idx in song_to_indices[name]]
        self.train_data = torch.utils.data.Subset(self, train_indices)
        self.validation_data = torch.utils.data.Subset(self, val_indices)
        return self.train_data, self.validation_data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.label[idx], dtype=torch.uint8)

class load_test_dataset(Dataset):
    def __init__(self, file_path):
        checkpoint = torch.load(file_path, weights_only=False)
        self.data=checkpoint["data"]
        self.label=checkpoint["label"]
        self.song_name = checkpoint["song_name"]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.label[idx], dtype=torch.uint8), self.song_name[idx]



