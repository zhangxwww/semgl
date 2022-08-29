import os
import numpy as np
import torch


def load_data(data_dir, dataset_name='cub', no_val=False):
    features = np.load(os.path.join(data_dir, 'features.npy'))
    labels = np.load(os.path.join(data_dir, 'labels.npy'))
    train_val_idx = np.load(os.path.join(data_dir, 'train_idx.npy'))
    test_idx = np.load(os.path.join(data_dir, 'test_idx.npy'))

    if no_val:
        train_idx = val_idx = train_val_idx
    else:
        train_idx = np.where(train_val_idx % 5 != 0)[0]
        val_idx = np.where(train_val_idx % 5 == 0)[0]

    features = torch.from_numpy(features)
    labels = torch.from_numpy(labels)
    train_idx = torch.from_numpy(train_idx)
    val_idx = torch.from_numpy(val_idx)
    test_idx = torch.from_numpy(test_idx)
    return features, labels, train_idx, val_idx, test_idx
