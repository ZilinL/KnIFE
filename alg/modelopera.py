# coding=utf-8
import torch
from network import EEG_network


def get_fea(args):
    if args.net == 'EEGNet':
        net = EEG_network.EEGNet(args.channels, args.points)
    elif args.net == 'DeepConveNet':
        net = EEG_network.DeepConveNet(args.channels, args.points)
    elif args.net == 'DeepConveNet_4':
        net = EEG_network.DeepConveNet_4(args.channels, args.points)
    elif args.net == 'DeepConveNet_6':
        net = EEG_network.DeepConveNet_6(args.channels, args.points)
    elif args.net == 'ShallowConveNet':
        net = EEG_network.ShallowConveNet(args.channels, args.points)
    return net

def get_teach_fea(args, teachNet):
    if teachNet == 'DeepConveNet':
        net = EEG_network.DeepConveNet(args.channels, args.points)
    elif args.net == 'EEGNet':
        net = EEG_network.EEGNet(args.channels, args.points)
    elif args.net == 'DeepConveNet':
        net = EEG_network.DeepConveNet(args.channels, args.points)
    elif args.net == 'DeepConveNet_4':
        net = EEG_network.DeepConveNet_4(args.channels, args.points)
    elif args.net == 'ShallowConveNet':
        net = EEG_network.ShallowConveNet(args.channels, args.points)
    return net


def accuracy(network, loader, args, item):

    dataset_list={'OpenBMI':[54,200], 'BCICIV-2a':[9,288], 'BCICIV-2b':[9,140], 'BCICIII':[1,280]}
    correct = 0
    total = 0
    all_y = torch.empty(0).cuda()
    all_p = torch.empty(0,args.num_classes).cuda()
    correct_subjects = []
    network.eval()
    with torch.no_grad():
        for data in loader:
            x = data[0].cuda().float()
            y = data[1].cuda().long()
            p = network.predict(x)
            all_p = torch.cat((all_p, p), dim=0)
            all_y = torch.cat((all_y, y), dim=0)
            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float()).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float()).sum().item()
            total += len(x)
    if item == 'target' :
        if 'BCICIV-2a' in args.dataset:
            subjects = dataset_list['BCICIV-2a'][0]
            data_per_subject = dataset_list['BCICIV-2a'][1]
        elif 'OpenBMI' in args.dataset:
            subjects = dataset_list['OpenBMI'][0]
            data_per_subject = dataset_list['OpenBMI'][1]
        elif 'BCICIII' in args.dataset:
            subjects = dataset_list['BCICIII'][0]
            data_per_subject = dataset_list['BCICIII'][1]
        elif 'BCICIV-2b' in args.dataset:
            subjects = dataset_list['BCICIV-2b'][0]
            data_per_subject = [320,280,320,320,320,320,320,320,320] # 每个被试个数不一样，因此导致总的精度和被试的平均精度值不一样
        else:
            print('No such dataset exists!')

        if 'BCICIV-2b' in args.dataset:
            for i in range(subjects):
                start_idx = sum(data_per_subject[:i])
                end_idx = sum(data_per_subject[:i+1])
                subject_p = all_p[start_idx:end_idx]
                subject_y = all_y[start_idx:end_idx]
                correct_subject = (subject_p.argmax(1).eq(subject_y).float()).sum().item() / len(subject_y)
                correct_subjects.append(correct_subject)
        else:
            for i in range(subjects):
                start_idx = i * data_per_subject
                end_idx = (i + 1) * data_per_subject
                subject_p = all_p[start_idx:end_idx]
                subject_y = all_y[start_idx:end_idx]
                correct_subject = (subject_p.argmax(1).eq(subject_y).float()).sum().item() / len(subject_y)
                correct_subjects.append(correct_subject)

    network.train()
    return correct / total, correct_subjects
