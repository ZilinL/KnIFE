# coding=utf-8
import numpy as np
import sklearn.model_selection as ms
from torch.utils.data import DataLoader

from datautil.EEGdataload import EEGDataset
from datautil.mydataloader import InfiniteDataLoader

def get_eeg_dataloader(args):
    rate = 0.2
    trdatalist, tedatalist = [], []

    names = args.eeg_dataset[args.dataset]
    args.domain_num = len(names)
    for i in range(len(names)):
        if i in args.test_envs:  # test_envs represents target domain data
            tedatalist.append(EEGDataset(args.dataset, args.channels, args.points, args.task, args.data_dir,
                                           names[i], i, test_envs=args.test_envs)) # if target, put in tedatalist
        else:
            tmpdatay = EEGDataset(args.dataset, args.channels, args.points, args.task, args.data_dir,
                                    names[i], i, test_envs=args.test_envs).labels
            l = len(tmpdatay)
            if args.split_style == 'strat':
                lslist = np.arange(l)
                stsplit = ms.StratifiedShuffleSplit(
                    2, test_size=rate, train_size=1-rate, random_state=args.seed)
                stsplit.get_n_splits(lslist, tmpdatay)
                indextr, indexte = next(stsplit.split(lslist, tmpdatay))
            else:
                indexall = np.arange(l)
                np.random.seed(args.seed)
                np.random.shuffle(indexall)
                ted = int(l*rate)
                indextr, indexte = indexall[:-ted], indexall[-ted:]

            trdatalist.append(EEGDataset(args.dataset, args.channels, args.points, args.task, args.data_dir,
                                           names[i], i, indices=indextr, test_envs=args.test_envs))
            tedatalist.append(EEGDataset(args.dataset, args.channels, args.points, args.task, args.data_dir,
                                           names[i], i, indices=indexte, test_envs=args.test_envs)) # tedatalist includes evl data and test data

    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=None,
        batch_size=args.batch_size,
        num_workers=args.N_WORKERS)
        for env in trdatalist]

    eval_loaders = [DataLoader(
        dataset=env,
        batch_size=64,
        num_workers=args.N_WORKERS,
        drop_last=False,
        shuffle=False)
        for env in trdatalist+tedatalist]

    return train_loaders, eval_loaders
