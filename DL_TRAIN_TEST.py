import numpy as np
import torch
from scipy.stats import stats
from torch import nn
from collections import OrderedDict
import torch.optim as optim
import math
import pandas as pd
import torch.nn.functional as F
import MIDI_DATASET
import MIDI_IO as io
import MIDI_coding as mc
import audio_metrics
from enum import Enum
import matplotlib.pyplot as plt
import DL_algorithms
import os
import time
import TRANFORMERS_DL_ALGORITHMS as transformers


class IM_DISTANCE_FROM(Enum):
    TARGET_DISTANCE = 1
    PREV_NOTE_OUTPUT_DISTANCE = 2
    PREV_NOTE_TARGET_DISTANCE = 3


class IM_DISTANCE_TYPE(Enum):
    L1 = 1
    Wasserstein = 2
    Harmonic_Distance = 3


class RETURN_TYPE(Enum):
    MEAN = 1
    SUM = 2


class MODEL_TYPE(Enum):
    CNN = 1
    TRANSFORMER = 2


def create_distance_matrix():
    distance_matrix = np.zeros((128, 128), dtype=np.float32)
    for i in range(128):
        for j in range(128):
            distance_matrix[i, j] = (1 + np.abs(i // 12 - j // 12)) * audio_metrics.scalar_harmonic_metric(i, j)
    return torch.tensor(distance_matrix)


# Cross Entropy Loss
class CrossEntropyWeightedDistanceLoss(nn.Module):
    def __init__(self, device, weights=None):
        """
        :param weights: in tensor.torch format. matrix 128*128 for each note what is the wighted distance from a different note. The weight is multiplied by the distance.
        If the wight in type None, simple CrossEntropy distance.
        """
        super(CrossEntropyWeightedDistanceLoss, self).__init__()
        self.weights = create_distance_matrix().to(device) if weights == None else weights.to(device)
        self.weights = self.weights.clamp(min=0, max=5)

    def forward(self, predictions, targets, targets_weights=1, prev_notes_weight=1):
        """
        :param predictions: in shape (batch_size, 128, sequence_len) - the output from the network
        :param targets: shape (batch_size, sequence_len) - actual note (between 0-128)
        :param targets_weights: how much weight to give the distance between prediction to target
        :param prev_notes_weight:how much weight to give the distance between prediction to previous target
        :return: mean loss of batch
        """
        targets = targets.long()
        predictions_loss = torch.flatten(predictions.transpose(2, 1), start_dim=0, end_dim=1)
        # calculate cross entropy loss
        cross_entropy = F.cross_entropy(predictions_loss, targets.flatten(),
                                        reduction='none')  # Shape: [batch_size*sequence_len]
        predictions_norm = F.softmax(predictions_loss, dim=1)
        """effective weights:
         self.weights[targets.flatten()].shape = [batch_size*seq_len, 128]
         targets_weights = int that is given in the function
         predictions_norm.shape = [batch_size*seq_len, 128] 
         after torch.mean/ torch.shape - we get [batch_size*seq_len]
        """
        effective_weights = -targets_weights * torch.sum(self.weights[targets.flatten()] * torch.log(1.0 - predictions_norm + 1e-5),
                                                        dim=1) #if targets_weights!=0 else torch.ones_like(cross_entropy)
        prev_targets = torch.zeros_like(targets)
        prev_targets[:, 1:] = targets[:, :-1]  # Shift right
        prev_weights = self.weights[prev_targets]
        prev_weights[:, 0, :] = 0
        distances_between_notes_weight = -prev_notes_weight * torch.sum(prev_weights.view(-1, 128) * torch.log(1.0 - predictions_norm + 1e-5),
                                                                       dim=1) #if prev_notes_weight!=0 else torch.ones_like(cross_entropy)
        #print("cross entropy - ", cross_entropy.mean())
        #print("effective_weights - ", effective_weights.mean())
        #print("distances_between_notes_weight - ", distances_between_notes_weight.mean())
        loss = cross_entropy + effective_weights + distances_between_notes_weight
        return torch.mean(loss)


def total_harmonic_distance(predictions, targets=None, im_distance_type=IM_DISTANCE_TYPE.Harmonic_Distance,
                            im_distance_from=IM_DISTANCE_FROM.TARGET_DISTANCE,
                            return_type=RETURN_TYPE.MEAN,
                            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """
    :param predictions: output from model - shape (batch_size, sequence_len, 128)
    :param targets: original sequence - shape (batch_size, sequence_len)
    :param IM_DISTANCE_FROM: Options:
    DEFAULT: IM_DISTANCE_FROM.TARGET_DISTANCE - returns distance from original song
    IM_DISTANCE_FROM.PREV_NOTE_OUTPUT_DISTANCE - returns distance from previous note from predictions
    IM_DISTANCE_FROM.PREV_NOTE_TARGET_DISTANCE - returns distance from previous note target
    :param return_type: Options:
    DEFAULT: MEAN - returns mean distance over all batch and sequence
    SUM - returns sum of all distances
    :return: distance
    """
    distance_matrix = create_distance_matrix().to(device)
    predictions_max = torch.argmax(predictions, dim=1)
    if (im_distance_from == IM_DISTANCE_FROM.TARGET_DISTANCE):
        if targets is None:
            raise ValueError("Input 'targets' cannot be None for TARGET_DISTANCE.")
        else:
            if im_distance_type == IM_DISTANCE_TYPE.Harmonic_Distance:
                distance = distance_matrix[predictions_max.flatten().long(), targets.flatten().long()]
            elif im_distance_type == IM_DISTANCE_TYPE.L1:
                distance = np.abs(predictions_max.flatten().long() - targets.flatten().long())
            elif im_distance_type == IM_DISTANCE_TYPE.Wasserstein:
                u = predictions_max[:, 1:].flatten().long().cpu().numpy()
                v = targets[:, :-1].flatten().long().cpu().numpy()
                distance = stats.wasserstein_distance(u, v)
                return distance
            else:
                raise ValueError(f"im_distance_type cant be {im_distance_type}")
    elif im_distance_from == IM_DISTANCE_FROM.PREV_NOTE_TARGET_DISTANCE:
        if targets is None:
            raise ValueError("Input 'targets' cannot be None for TARGET_DISTANCE.")
        else:
            if (im_distance_type == IM_DISTANCE_TYPE.Harmonic_Distance):
                distance = distance_matrix[
                    predictions_max[:, 1:].flatten().long(), targets[:, :-1].flatten().long()]
            elif (im_distance_type == IM_DISTANCE_TYPE.L1):
                distance = np.abs(predictions_max[:, 1:].flatten().long() - targets[:, :-1].flatten().long())
            elif (im_distance_type == IM_DISTANCE_TYPE.Wasserstein):
                u = predictions_max[:, 1:].flatten().long().cpu().numpy()
                v = targets[:, :-1].flatten().long().cpu().numpy()
                distance = stats.wasserstein_distance(u, v)
                return distance
            else:
                raise ValueError(f"im_distance_type cant be {im_distance_type}")
    elif (im_distance_from == IM_DISTANCE_FROM.PREV_NOTE_OUTPUT_DISTANCE):
        if im_distance_type == IM_DISTANCE_TYPE.Harmonic_Distance:
            distance = distance_matrix[
                predictions_max[:, 1:].flatten().long(), predictions_max[:, :-1].flatten().long()]
        elif im_distance_type == IM_DISTANCE_TYPE.L1:
            distance = np.abs(predictions_max[:, 1:].flatten().long() - predictions_max[:, :-1].flatten().long())
        elif im_distance_type == IM_DISTANCE_TYPE.Wasserstein:
            u = predictions_max[:, 1:].flatten().long().cpu().numpy()
            v = predictions_max[:, :-1].flatten().long().cpu().numpy()
            distance = stats.wasserstein_distance(u, v)
            return distance
        else:
            raise ValueError(f"im_distance_type cant be {im_distance_type}")
    else:
        raise ValueError(f"no distance type entered - {im_distance_from}")
    if (return_type == RETURN_TYPE.MEAN):
        return torch.mean(distance.float()).detach().cpu().item()
    elif (return_type == RETURN_TYPE.SUM):
        return torch.sum(distance.float()).detach().cpu().item()
    else:
        raise ValueError("no return type entered")


def calculate_im(distance_trained, distance_not_trained):
    """
    :param distance_trained: the distance for IM calculations after trained model
    :param distance_not_trained:  the distance for IM calculations before trained model
    :return: returns the IM calculations
    """
    return 100 * (distance_not_trained - distance_trained) / distance_not_trained


def im(output_trained, output_not_trained, original=None, im_type=IM_DISTANCE_TYPE.Harmonic_Distance,
       device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """
    :param output_trained:
    :param output_not_trained:
    :param original: if NONE only calculates the distance for PREV_NOTE_OUTPUT
    :return:
    """
    return_dict = {}
    if (original != None):
        distance_not_trained_target = total_harmonic_distance(predictions=output_not_trained, targets=original,
                                                              im_distance_type=im_type,
                                                              im_distance_from=IM_DISTANCE_FROM.TARGET_DISTANCE,
                                                              device=device)
        distance_trained_target = total_harmonic_distance(predictions=output_trained, targets=original,
                                                          im_distance_type=im_type,
                                                          im_distance_from=IM_DISTANCE_FROM.TARGET_DISTANCE,
                                                          device=device)
        return_dict["TARGET_DISTANCE"] = calculate_im(distance_trained_target,
                                                      distance_not_trained_target)
        distance_not_trained_prev_target = total_harmonic_distance(predictions=output_not_trained, targets=original,
                                                                   im_distance_type=im_type,
                                                                   im_distance_from=IM_DISTANCE_FROM.PREV_NOTE_TARGET_DISTANCE,
                                                                   device=device)
        distance_trained_prev_target = total_harmonic_distance(predictions=output_trained, targets=original,
                                                               im_distance_type=im_type,
                                                               im_distance_from=IM_DISTANCE_FROM.PREV_NOTE_TARGET_DISTANCE,
                                                               device=device)
        return_dict["PREV_NOTE_TARGET"] = calculate_im(distance_trained_prev_target,
                                                       distance_not_trained_prev_target)
    distance_not_trained_prev_prediction = total_harmonic_distance(predictions=output_not_trained,
                                                                   im_distance_type=im_type,
                                                                   im_distance_from=IM_DISTANCE_FROM.PREV_NOTE_OUTPUT_DISTANCE,
                                                                   device=device)
    distance_trained_prev_prediction = total_harmonic_distance(predictions=output_trained, im_distance_type=im_type,
                                                               im_distance_from=IM_DISTANCE_FROM.PREV_NOTE_OUTPUT_DISTANCE,
                                                               device=device)
    return_dict["PREV_NOTE_OUTPUT"] = calculate_im(distance_trained_prev_prediction,
                                                   distance_not_trained_prev_prediction)
    return return_dict


class Train():
    def __init__(self, model_type, training_set, validation_set, path, device, model=None):
        self.model_type = model_type
        self.training_set = training_set
        self.validation_set = validation_set
        self.path = path
        self.device = device
        self.model = model if model is not None else None

    def set_loss_function(self, weights_target, weights_before):
        self.loss_function = CrossEntropyWeightedDistanceLoss(self.device)
        self.weights_before = weights_before
        self.weights_target = weights_target

    def set_training_params(self, **args):
        if (self.model is not None):
            self.batch_size = args.get("batch_size", 16)
            self.lr_start = args.get("lr_start", 1e-3)
            self.lr_step = args.get("lr_step", 4)
            self.lr_mul = args.get("lr_mul", 0.7)
            self.with_eval_set = args.get("with_eval_set", True)
            self.load_train = torch.utils.data.DataLoader(dataset=self.training_set, batch_size=self.batch_size,
                                                          shuffle=True, drop_last=True)
            self.load_validation = torch.utils.data.DataLoader(dataset=self.validation_set, batch_size=self.batch_size,
                                                               shuffle=False, drop_last=True)
            print("finished setting up parameters")
        else:
            raise Exception("need to set model before setting training params")

    def set_model(self, **args):
        self.seq_len = args.get("seq_len", 64)
        self.dropout = args.get("dropout", 0.5)
        self.with_eval_set = args.get("with_eval_set", True)
        if (self.model_type == MODEL_TYPE.CNN):
            self.dropout_depth = args.get("dropout_depth", 0.1)
            self.model = DL_algorithms.CNN_MODEL(sequence_len=self.seq_len, dropout=self.dropout,
                                                 dropout_depth=self.dropout_depth, device=self.device)
        if (self.model_type == MODEL_TYPE.TRANSFORMER):
            self.hidden_dim = args.get("hidden_dim", 512)
            self.num_encoder_layers = args.get("num_encoder_layers", 4)
            self.num_decoder_layers = args.get("num_decoder_layers", 2)
            self.model = transformers.Transformers_Model(vocab_size=128,
                                                         num_decoder_layers=self.num_decoder_layers,
                                                         num_encoder_layers=self.num_encoder_layers,
                                                         hidden_dim=self.hidden_dim,
                                                         output_dim=128,
                                                         nhead=8).to(self.device)

    def plot_training_table(self, training_table=None, save=False, show=False):
        # Convert all torch tensors to CPU floats
        if training_table is None:
            training_table = self.training_table
        training_table = training_table.applymap(
            lambda x: x.detach().cpu().item() if torch.is_tensor(x) else x
        )

        epochs = training_table.index

        plt.figure(figsize=(15, 10))

        # Plot Loss
        plt.subplot(2, 2, 1)
        plt.plot(epochs, training_table["loss train"], label='Train Loss', marker='o')
        plt.plot(epochs, training_table["loss validation"], label='Validation Loss', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Over Epochs')
        plt.legend()
        plt.grid(True)

        # Plot IM Target
        plt.subplot(2, 2, 2)
        plt.plot(epochs, training_table["im train target"], label='IM Train Target', marker='o')
        plt.plot(epochs, training_table["im valid target"], label='IM Validation Target', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('IM Target')
        plt.title('IM Target harmonic Over Epochs')
        plt.legend()
        plt.grid(True)

        # Plot IM Previous Target
        plt.subplot(2, 2, 3)
        plt.plot(epochs, training_table["im train prev target"], label='IM Train Prev Target', marker='o')
        plt.plot(epochs, training_table["im valid prev target"], label='IM Validation Prev Target', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('IM Prev Target')
        plt.title('IM Previous Target harmonic Over Epochs')
        plt.legend()
        plt.grid(True)

        # Plot IM Previous Output
        plt.subplot(2, 2, 4)
        plt.plot(epochs, training_table["im train prev output"], label='IM Train Prev Output', marker='o')
        plt.plot(epochs, training_table["im valid prev output"], label='IM Validation Prev Output', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('IM Prev Output')
        plt.title('IM Previous Output harmonic Over Epochs')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        if save:
            plt.savefig(f'{self.path}.png')
        if show:
            plt.show()

    def train(self, epoch_size):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr_start, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_step, gamma=self.lr_mul)
        return_table = pd.DataFrame(index=np.arange(epoch_size),
                                    columns=["lr", "loss train", "loss validation", "im train target",
                                             "im valid target", "im train prev target", "im valid prev target",
                                             "im train prev output", "im valid prev output"])
        for epoch in range(epoch_size):
            print(f"running {epoch}/{epoch_size}")
            print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
            run_loss = 0
            im_total = {"TARGET_DISTANCE": 0, "PREV_NOTE_TARGET": 0, "PREV_NOTE_OUTPUT": 0}
            total = 0
            self.model.train()
            for X_train, t_train in self.load_train:
                X_train = X_train.to(self.device)
                t_train = t_train.to(self.device)
                optimizer.zero_grad()
                if (self.model_type == MODEL_TYPE.CNN):
                    y = self.model(X_train)
                if (self.model_type == MODEL_TYPE.TRANSFORMER):
                    x = torch.argmax(X_train, dim=1).to(self.device)
                    y = self.model(x,x)
                loss = self.loss_function(y, t_train, targets_weights=self.weights_target,
                                          prev_notes_weight=self.weights_before)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                run_loss += loss
                batch_im = im(output_trained=y, output_not_trained=X_train, original=t_train)
                im_total["TARGET_DISTANCE"] += batch_im["TARGET_DISTANCE"]
                im_total["PREV_NOTE_OUTPUT"] += batch_im["PREV_NOTE_OUTPUT"]
                im_total["PREV_NOTE_TARGET"] += batch_im["PREV_NOTE_TARGET"]
                total += 1
            current_lr = optimizer.param_groups[0]['lr']
            return_table.at[epoch, "lr"] = current_lr
            return_table.at[epoch, "loss train"] = run_loss / total
            return_table.at[epoch, "im train target"] = im_total["TARGET_DISTANCE"] / total
            return_table.at[epoch, "im train prev output"] = im_total["PREV_NOTE_OUTPUT"] / total
            return_table.at[epoch, "im train prev target"] = im_total["PREV_NOTE_TARGET"] / total
            print(f"Train: Loss: {run_loss / total} for epoch {epoch}")
            scheduler.step()
            with torch.no_grad():
                if (self.with_eval_set):
                    self.model.eval()
                    eval_loss = 0
                    im_total = {"TARGET_DISTANCE": 0, "PREV_NOTE_TARGET": 0, "PREV_NOTE_OUTPUT": 0}
                    total = 0
                    for x_valid, t_valid in self.load_validation:
                        if (self.model_type == MODEL_TYPE.CNN):
                            y = self.model(x_valid.to(self.device))
                        if (self.model_type == MODEL_TYPE.TRANSFORMER):
                            x = torch.argmax(x_valid, dim=1).to(self.device)
                            y = self.model(x, x)
                        loss = self.loss_function(y, t_valid.to(self.device), targets_weights=self.weights_target,
                                                  prev_notes_weight=self.weights_before)
                        eval_loss += loss
                        batch_im = im(output_trained=y, output_not_trained=x_valid,
                                      original=t_valid)
                        im_total["TARGET_DISTANCE"] += batch_im["TARGET_DISTANCE"]
                        im_total["PREV_NOTE_OUTPUT"] += batch_im["PREV_NOTE_OUTPUT"]
                        im_total["PREV_NOTE_TARGET"] += batch_im["PREV_NOTE_TARGET"]
                        total += 1
                    print(f"Eval: Loss: {eval_loss / total} for epoch {epoch}")
                    return_table.at[epoch, "loss validation"] = eval_loss / total
                    return_table.at[epoch, "im valid target"] = im_total["TARGET_DISTANCE"] / total
                    return_table.at[epoch, "im valid prev output"] = im_total["PREV_NOTE_OUTPUT"] / total
                    return_table.at[epoch, "im valid prev target"] = im_total["PREV_NOTE_TARGET"] / total
        print(return_table)
        self.training_table = return_table.copy()
        self.plot_training_table(show=True)

    def save_model_weights(self):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'weights_target': self.weights_target,
            'weights_before': self.weights_before,
            'epoch_size': self.training_table.shape[0]
            # Add more if needed
        }, f'{self.path}.pth')

    def save_training(self):
        self.plot_training_table(save=True)
        self.training_table.to_csv(f'{self.path}.csv')


class Test():
    def __init__(self, device, path):
        self.device = device
        self.path = path
        self.testing_results = pd.DataFrame(columns=["file_name",
                                                     "cross_entropy_no_weights",
                                                     "cross_entropy_weights_target",
                                                     "cross_entropy_weights_prev",
                                                     "im target L1",
                                                     "im prev target L1",
                                                     "im prev output L1",
                                                     "im target harmonic",
                                                     "im prev target harmonic",
                                                     "im prev output harmonic",
                                                     "im target warrestein",
                                                     "im prev target warrestein",
                                                     "im prev output warrestein",
                                                     "inference time"])

    def load_model(self, model_type, sequence_len, hidden_dim, decoder_layers, encoder_layers):
        self.model_type = model_type
        if (model_type == MODEL_TYPE.CNN):
            self.model = DL_algorithms.CNN_MODEL(sequence_len=sequence_len, device=self.device)
            print(f"finished loading model from path {self.path}.pth")
        if (model_type == MODEL_TYPE.TRANSFORMER):
            self.model = transformers.Transformers_Model(vocab_size=128,
                                                         output_dim=128,
                                                         hidden_dim=hidden_dim,
                                                         num_decoder_layers=decoder_layers,
                                                         num_encoder_layers=encoder_layers,
                                                         nhead=8)
        self.model.load_state_dict(torch.load(f'{self.path}.pth', map_location=self.device, weights_only=False))
        self.model.to(self.device)

    def test_single_file(self, corrupt_midi_file_path, correct_midi_file_path, output_path=None,
                         add_to_results=True):
        print("testing song", output_path.split('\\')[-1])
        output_path = output_path if output_path is not None else self.path
        error = np.array(io.vectorize_MIDI(corrupt_midi_file_path))
        if (error.shape[2] <= 64):
            print("error shape:", error.shape)
            return 0
        else:
            error = error[:, :, 0:64]
        error = torch.tensor(error, dtype=torch.float).to(self.device)
        original = np.array(io.vectorize_MIDI(correct_midi_file_path))
        original = torch.tensor(original.argmax(axis=1)).long().to(self.device)
        if (original.shape[1] <= 64):
            print("original shape:", original.shape)
            return 0
        else:
            original = original[:, 0:64]
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        start_time = time.time()
        with torch.no_grad():
            if (self.model_type == MODEL_TYPE.CNN):
                audio_train = self.model(error)
            elif (self.model_type == MODEL_TYPE.TRANSFORMER):
                error_ind = torch.argmax(error, dim=1).to(self.device)
                audio_train = self.model(error_ind, error_ind)
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        end_time = time.time()
        MIDI_DATASET.one_hot_encode_output(audio_train, output_path)
        loss_function = CrossEntropyWeightedDistanceLoss(device=self.device)
        im_results_L1 = im(output_trained=audio_train, output_not_trained=error, original=original,
                           im_type=IM_DISTANCE_TYPE.L1)
        im_results_harmonic = im(output_trained=audio_train, output_not_trained=error, original=original,
                                 im_type=IM_DISTANCE_TYPE.Harmonic_Distance)
        im_results_warrestein = im(output_trained=audio_train, output_not_trained=error, original=original,
                                   im_type=IM_DISTANCE_TYPE.Wasserstein)
        u = original.flatten().cpu().numpy()
        v = torch.argmax(audio_train, dim=1).flatten().cpu().numpy()
        if add_to_results:
            new_results = {
                "file_name": output_path.split('\\')[-1],
                "cross_entropy_no_weights": loss_function(predictions=audio_train, targets=original, targets_weights=0,
                                                          prev_notes_weight=0).detach().cpu().item(),
                "cross_entropy_weights_target": loss_function(predictions=audio_train, targets=original,
                                                              targets_weights=1,
                                                              prev_notes_weight=0).detach().cpu().item(),
                "cross_entropy_weights_prev": loss_function(predictions=audio_train, targets=original,
                                                            targets_weights=0,
                                                            prev_notes_weight=1).detach().cpu().item(),
                "im target L1": im_results_L1["TARGET_DISTANCE"],
                "im prev target L1": im_results_L1["PREV_NOTE_TARGET"],
                "im prev output L1": im_results_L1["PREV_NOTE_OUTPUT"],
                "im target harmonic": im_results_harmonic["TARGET_DISTANCE"],
                "im prev target harmonic": im_results_harmonic["PREV_NOTE_TARGET"],
                "im prev output harmonic": im_results_harmonic["PREV_NOTE_OUTPUT"],
                "im target warrestein": im_results_warrestein["TARGET_DISTANCE"],
                "im prev target warrestein": im_results_warrestein["PREV_NOTE_TARGET"],
                "im prev output warrestein": im_results_warrestein["PREV_NOTE_OUTPUT"],
                "inference time": end_time - start_time}
            print(new_results)
            self.testing_results.loc[len(self.testing_results)] = new_results
        return 0

    def test_multiple_from_dataset(self, test_dataset, output_path, test_name):
        song_to_show = np.random.randint(low=0, high=len(test_dataset), size=3)
        i=0
        load_test = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
        for error_song, original_song, song_name in load_test:
            error_song = error_song.to(self.device)
            original_song = original_song.to(self.device)
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            start_time = time.time()
            with torch.no_grad():
                if (self.model_type == MODEL_TYPE.CNN):
                    audio_train = self.model(error_song)
                elif (self.model_type == MODEL_TYPE.TRANSFORMER):
                    error_ind = torch.argmax(error_song, dim=1).to(self.device)
                    audio_train = self.model(error_ind, error_ind)
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()
            if(i in song_to_show):
                MIDI_DATASET.one_hot_encode_output(audio_train, f"{output_path}{song_name}_{test_name}")
                MIDI_DATASET.one_hot_encode_output(error_song, f"{output_path}{song_name}_error")
                MIDI_DATASET.one_hot_encode_output(original_song, f"{output_path}{song_name}_original")
            loss_function = CrossEntropyWeightedDistanceLoss(device=self.device)
            im_results_L1 = im(output_trained=audio_train, output_not_trained=error_song, original=original_song,
                               im_type=IM_DISTANCE_TYPE.L1)
            im_results_harmonic = im(output_trained=audio_train, output_not_trained=error_song, original=original_song,
                                     im_type=IM_DISTANCE_TYPE.Harmonic_Distance)
            im_results_warrestein = im(output_trained=audio_train, output_not_trained=error_song, original=original_song,
                                       im_type=IM_DISTANCE_TYPE.Wasserstein)
            new_results = {
                "file_name": song_name,
                "cross_entropy_no_weights": loss_function(predictions=audio_train, targets=original_song,
                                                          targets_weights=0,
                                                          prev_notes_weight=0).detach().cpu().item(),
                "cross_entropy_weights_target": loss_function(predictions=audio_train, targets=original_song,
                                                              targets_weights=1,
                                                              prev_notes_weight=0).detach().cpu().item(),
                "cross_entropy_weights_prev": loss_function(predictions=audio_train, targets=original_song,
                                                            targets_weights=0,
                                                            prev_notes_weight=1).detach().cpu().item(),
                "im target L1": im_results_L1["TARGET_DISTANCE"],
                "im prev target L1": im_results_L1["PREV_NOTE_TARGET"],
                "im prev output L1": im_results_L1["PREV_NOTE_OUTPUT"],
                "im target harmonic": im_results_harmonic["TARGET_DISTANCE"],
                "im prev target harmonic": im_results_harmonic["PREV_NOTE_TARGET"],
                "im prev output harmonic": im_results_harmonic["PREV_NOTE_OUTPUT"],
                "im target warrestein": im_results_warrestein["TARGET_DISTANCE"],
                "im prev target warrestein": im_results_warrestein["PREV_NOTE_TARGET"],
                "im prev output warrestein": im_results_warrestein["PREV_NOTE_OUTPUT"],
                "inference time": end_time - start_time}
            print(new_results)
            i+=1
            self.testing_results.loc[len(self.testing_results)] = new_results

    def test_multiple_from_folder(self, test_name, folder_path, with_original_songs=True, original_key="_orig",
                                  error_key="_error"):
        file_name = {}
        sname = ""
        for fname in os.scandir(folder_path):
            if (os.path.splitext(fname)[1] == ".midi"):
                if (with_original_songs):
                    if original_key in os.path.splitext(fname)[0]:
                        sname = os.path.splitext(fname)[0][:os.path.splitext(fname)[0].find(original_key)]
                        with_error = False
                if error_key in os.path.splitext(fname)[0]:
                    sname = os.path.splitext(fname)[0][:os.path.splitext(fname)[0].find(error_key)]
                    with_error = True
                if sname not in file_name.keys() and sname != "":
                    file_name[sname] = {"original": "", "error": ""} if with_original_songs else {"error"}
                if (with_error):
                    file_name[sname]["error"] = fname
                else:
                    file_name[sname]["original"] = fname
        print("list of songs: ", file_name.keys())
        for song_name, files in file_name.items():
            print("starting - ", song_name)
            print("checking files - ", files)
            if (files["error"] == '' or files["original"] == '' and with_original_songs):
                raise Exception(f"song {song_name} doesnt have original or error")
            if (with_original_songs):
                self.test_single_file(corrupt_midi_file_path=files["error"],
                                      correct_midi_file_path=files["original"],
                                      output_path=f"{song_name}_{test_name}")
            else:
                self.test_single_file(corrupt_midi_file_path=files["error"],
                                      output_path=f"{song_name}_after_correction")

    def save_test_results(self, test_name, results_file_path):
        results = self.testing_results.mean(numeric_only=True)
        results["test name"] = test_name
        results["test size"] = len(self.testing_results)
        results_df = pd.DataFrame([results])
        print(results_df)
        if os.path.isfile(results_file_path):
            # Append without writing the header again
            results_df.to_csv(results_file_path, mode='a', header=False, index=False)
        else:
            # Create a new file with header
            results_df.to_csv(results_file_path, mode='w', header=True, index=False)


def create_test_folder(song_path, output_path, error_song=None, l=64, original_key="_orig", error_key="_error",
                       test_size=None):
    original = np.array(io.vectorize_MIDI(song_path))[0]
    if (error_song == None):
        error_song = mc.apply_errors_for_single_note_audio(original)
    song_name = os.path.splitext(song_path)[0].split('\\')[-1]
    os.makedirs(output_path, exist_ok=True)
    if (test_size != None):
        step = original.shape[1] // test_size
    else:
        step = l
    index = 0
    for i in range(0, original.shape[1] - step - 3, step):
        new_original = original[:, i:i + l + 3]
        new_error = error_song[:, i:i + l + 3]
        io.export_MIDI(np.array([new_original]), f"{output_path}{song_name}_{index}{original_key}.midi")
        io.export_MIDI(np.array([new_error]), f"{output_path}{song_name}_{index}{error_key}.midi")
        index += 1
    print("finished creating test")
