import os
import random
import argparse
import numpy as np
import pandas as pd
import ast

import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class PSMSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = MinMaxScaler()
        data = pd.read_csv(data_path + '/train.csv')
        data = data.values[:, 1:]

        data = np.nan_to_num(data)

        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(data_path + '/test.csv')

        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)

        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test

        self.test_labels = pd.read_csv(data_path + '/test_label.csv').values[:, 1:]

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), \
                   np.float32(self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class MSLSegLoader(object):
    def __init__(self, root_path, win_size, step=1, fname="A-1", mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.fname = fname
        self.scaler = MinMaxScaler()
        
        # Load labeled anomalies metadata
        labeled_anomalies_path = os.path.join(root_path, 'labeled_anomalies.csv')
        labeled_anomalies = pd.read_csv(labeled_anomalies_path, delimiter=',')
        data_info = labeled_anomalies[labeled_anomalies['chan_id'] == fname]
        
        data_path = os.path.join(root_path, "train", f"{fname}.npy")
        data = np.load(data_path)
        self.scaler.fit(data)
        
        data_path = os.path.join(root_path, "test", f"{fname}.npy")
        test_data = np.load(data_path)
            
        self.train = self.scaler.transform(data)
        self.test = self.scaler.transform(test_data)
        self.val = self.test
            
        # Generate test labels
        labels = []
        for _, row in data_info.iterrows():
            anomalies = ast.literal_eval(row['anomaly_sequences'])
            length = row.iloc[-1]
            label = np.zeros([length], dtype=bool)
            for anomaly in anomalies:
                label[anomaly[0]:anomaly[1] + 1] = True
            labels.extend(label)
        self.targets = np.asarray(labels)

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.targets[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.targets[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(self.targets[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.targets[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMAPSegLoader(object):
    def __init__(self, root_path, win_size, step=1, fname="A-1", mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.fname = fname
        self.scaler = MinMaxScaler()
        
        # Load labeled anomalies metadata
        labeled_anomalies_path = os.path.join(root_path, 'labeled_anomalies.csv')
        labeled_anomalies = pd.read_csv(labeled_anomalies_path, delimiter=',')
        data_info = labeled_anomalies[labeled_anomalies['chan_id'] == fname]
        
        data_path = os.path.join(root_path, "train", f"{fname}.npy")
        data = np.load(data_path)
        self.scaler.fit(data)
        
        data_path = os.path.join(root_path, "test", f"{fname}.npy")
        test_data = np.load(data_path)
            
        self.train = self.scaler.transform(data)
        self.test = self.scaler.transform(test_data)
        self.val = self.test
            
        # Generate test labels
        labels = []
        for _, row in data_info.iterrows():
            anomalies = ast.literal_eval(row['anomaly_sequences'])
            length = row.iloc[-1]
            label = np.zeros([length], dtype=bool)
            for anomaly in anomalies:
                label[anomaly[0]:anomaly[1] + 1] = True
            labels.extend(label)
        self.targets = np.asarray(labels)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.targets[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.targets[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.targets[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.targets[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMDSegLoader(object):
    def __init__(self, root_path, win_size, step=1, fname="machine-1-1.txt", mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.fname = fname
        self.scaler = MinMaxScaler()
        
        data_path = os.path.join(root_path, "train", fname)
        data = pd.read_csv(data_path)
        data = np.asarray(data)
        self.scaler.fit(data)
        
        data_path = os.path.join(root_path, "test", fname)
        test_data = pd.read_csv(data_path)
        test_data = np.asarray(test_data)
        
        # Load labeled anomalies metadata        
        file_path = os.path.join(root_path, "test_label", fname)
        labeled_anomalies = pd.read_csv(file_path)
        self.targets = np.asarray(labeled_anomalies)
            
        self.train = self.scaler.transform(data)
        self.test = self.scaler.transform(test_data)
        self.val = self.test

        
    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.targets[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.targets[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.targets[index:index + self.win_size])
        else:
            return np.float32(
                self.test[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.targets[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


def get_loader_segment(data_path, batch_size, window_size=100, step=100, mode='train', dataset='KDD', fname=None):
    if (dataset == 'SMD'):
        dataset = SMDSegLoader(data_path, window_size, step, fname, mode)
    elif (dataset == 'MSL'):
        dataset = MSLSegLoader(data_path, window_size, step, fname, mode)
    elif (dataset == 'SMAP'):
        dataset = SMAPSegLoader(data_path, window_size, step, fname, mode)
    elif (dataset == 'PSM'):
        dataset = PSMSegLoader(data_path, window_size, step, mode)
        
    shuffle = False
    if mode == 'train':
        shuffle = True
    data_loader =DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=0,
                            pin_memory=True)
    return data_loader


def he_init_normal(m):
    if isinstance(m, (torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.Linear)):
        torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0)


class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if torch.isnan(metrics):
            return True
            # return False

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)

def set_device(config_device):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Arrange GPU devices starting from 0
    os.environ["CUDA_VISIBLE_DEVICES"] = config_device
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    print('Device: %s' % device)
    if (device.type == 'cuda') or (torch.cuda.device_count() > 1):
        print('GPU activate --> Count of using GPUs: %s' % torch.cuda.device_count())
    
    return device


def set_feature_size(dset):
    if dset == 'MSL':
        feature_size = 55
    elif dset == 'SMD':
        feature_size = 38
    elif dset == 'SMAP':
        feature_size = 25
    elif dset == 'PSM':
        feature_size = 25
    elif dset == 'SWaT':
        feature_size = 51
    
    return feature_size
    
