# coding=utf-8
import numpy as np
from datautil.util import Nmax
from scipy.io import loadmat


class EEGDataset(object):
    def __init__(self, dataset, channels, points, task, root_dir, domain_name, domain_label=-1, labels=None, indices=None, test_envs=[]):
        data = np.empty((channels,points,0))
        labels = []
        for i in range(len(domain_name)):
            data_mat = loadmat(root_dir+domain_name[i])
            data = np.concatenate([data, data_mat['x']], axis=2) # (channels, trial_length, samples)
            labels = np.concatenate([labels, np.squeeze(data_mat['y'])], axis=0)
        self.domain_num = 0
        self.task = task
        self.dataset = dataset
        self.labels = np.array(labels)
        self.x = data[np.newaxis, :]

        if indices is None:
            self.indices = np.arange(data.shape[2])
        else:
            self.indices = indices
        self.dlabels = np.ones(self.labels.shape) * \
            (domain_label-Nmax(test_envs, domain_label))

    def set_labels(self, tlabels=None, label_type='domain_label'):
        assert len(tlabels) == len(self.x)
        if label_type == 'domain_label':
            self.dlabels = tlabels
        elif label_type == 'class_label':
            self.labels = tlabels


    def __getitem__(self, index):
        index = self.indices[index]
        data = self.x[:,:,:,index]
        ctarget = self.labels[index]
        dtarget = self.dlabels[index]
        return data, ctarget, dtarget

    def __len__(self):
        return len(self.indices)
