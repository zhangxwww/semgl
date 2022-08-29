import os
import scipy.io as scio
import numpy as np
import torch


def load_data(data_dir, feature_name, dataset_name):
    if dataset_name == 'ntu':
        filename = 'NTU2012_mvcnn_gvcnn.mat'
    elif dataset_name == 'modelnet':
        filename = 'ModelNet40_mvcnn_gvcnn.mat'
    else:
        raise Exception('Wrong dataset name: {}'.format(dataset_name))
    filename = os.path.join(data_dir, filename)
    data = scio.loadmat(filename)
    labels = data['Y'].astype(np.long)
    if labels.min() == 1:
        labels = labels - 1
    idx = data['indices'].item()

    if feature_name == 'MVCNN':
        features = data['X'][0].item().astype(np.float32)
    elif feature_name == 'GVCNN':
        features = data['X'][1].item().astype(np.float32)
    else:
        raise Exception('Wrong feature name {}'.format(feature_name))

    idx_train_val = np.where(idx == 1)[0]
    idx_train = np.where(idx_train_val % 5 != 0)[0]
    idx_val = np.where(idx_train_val % 5 == 0)[0]
    idx_test = np.where(idx == 0)[0]

    features = torch.from_numpy(features)
    labels = torch.from_numpy(labels).reshape(-1)
    idx_train = torch.from_numpy(idx_train)
    idx_val = torch.from_numpy(idx_val)
    idx_test = torch.from_numpy(idx_test)
    return features, labels, idx_train, idx_val, idx_test
