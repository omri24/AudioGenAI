import os
import MIDI_IO as io
import MIDI_coding as mc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split



class create_dataset():
    def __init__(self, path, length):
        i = 0
        self.dataset_data = []
        self.dataset_label=[]
        for fname in os.scandir(path):
            if (os.path.splitext(fname)[1] == ".midi"):
                print("in new song")
                np_array = np.array(io.vectorize_MIDI(fname))
                for t in np_array:
                    n = int(len(t[0])/length)
                    for i in range(n):
                        print("adding new file to array")
                        for j in range(3):
                            song = t[:, i:i+length]
                            new_song = mc.apply_errors_for_single_note_audio(song)
                            self.dataset_data.append(new_song)
                            self.dataset_label.append(song.argmax(axis=0))
        self.save_dataset("dataset", 1, self.dataset_data, self.dataset_label)
    def save_dataset(self, folder_path, i, data, label):
        os.makedirs(folder_path, exist_ok=True)
        torch.save({"data": data, "label": label}, os.path.join(folder_path, f"data_with_labels{i}.pth"))


class load_dataset(Dataset):
    def __init__(self, file_path):
        checkpoint = torch.load(file_path)
        self.data=checkpoint["data"]
        self.label=checkpoint["label"]

    def split_train_test(self, train_percent, validation_percent=0):
        train_size = int(len(self) * train_percent)
        validation_size=int(len(self)*validation_percent)
        test_size = int(len(self) - train_size-validation_size)
        self.train_data, self.validation_data, self.test_data = random_split(self, [train_size, validation_size, test_size])
        return  self.train_data, self.validation_data, self.test_data
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.label[idx], dtype=torch.uint8)


def create_weight_loss_matrix():
    """
    :return: Weights matrix 128*128
    Notes inside octave:
    1 - the notes are the same
    0- The worst option (triton)
    0.2 - notes outside of scale
    0.4 - notes inside of scale
    0.6 - notes inside of chord
    0.7 - Tonica
    0.55 - notes within chord, 7th degree
    TODO - Notes outside octave
    """

    target, prediction = np.mgrid[0:128, 0:128]
    weights=np.zeros(shape=(128,128))
    weights_values = [1, 0, 0.2, 0.4, 0.6, 0.7, 0.55]
    relations = [[0], [7], [1,3,8,10], [2,5,9], [4], [6], [11]]
    for i,r in enumerate(relations):
        for d in r:
            weights=np.where((target+d)%12==prediction%12, weights_values[i], weights)
    return weights


