import os
import torch
import torch.nn.functional as F


def load_data(data_dir, config):

    

    if config['large']:
        features = torch.load(os.path.join(data_dir, 'SUN_features_1w.pt'))
        labels = torch.load(os.path.join(data_dir, 'SUN_labels_1w.pt'))

        idx_train = torch.load(os.path.join(data_dir, 'SUN_idx_train_1w.pt'))
        idx_val = torch.load(os.path.join(data_dir, 'SUN_idx_val_1w.pt'))
        idx_test = torch.load(os.path.join(data_dir, 'SUN_idx_test_1w.pt'))

    else:
        features = torch.load(os.path.join(data_dir, 'SUN_features.pt'))
        labels = torch.load(os.path.join(data_dir, 'SUN_labels.pt'))

        idx_train = torch.load(os.path.join(data_dir, 'SUN_idx_train.pt'))
        idx_val = torch.load(os.path.join(data_dir, 'SUN_idx_val.pt'))
        idx_test = torch.load(os.path.join(data_dir, 'SUN_idx_test.pt'))

    features = F.normalize(features, p=2, dim=1)
    return features, labels, idx_train, idx_val, idx_test
