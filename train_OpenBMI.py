# coding=utf-8

import os
import sys
import time
import numpy as np
import argparse

import datetime
import random
import string

from alg.opt import *
from alg import alg, modelopera
from utils.util import set_random_seed, save_checkpoint, print_args, train_valid_target_eval_names, alg_loss_dict, Tee, eeg_param_init, print_environ
from datautil.getdataloader import get_eeg_dataloader

current_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

def get_args():
    parser = argparse.ArgumentParser(description='DG')
    parser.add_argument('--algorithm', type=str, default="Knife") 
    parser.add_argument('--alpha', type=float, 
                        default=1, help='DANN dis alpha')
    parser.add_argument('--anneal_iters', type=int,
                        default=500, help='Penalty anneal iters used in VREx')
    parser.add_argument('--batch_size', type=int,
                        default=32, help='batch_size')
    parser.add_argument('--beta', type=float,
                        default=0.1, help='beta')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='Adam hyper-param')
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--checkpoint_freq', type=int,
                        default=3, help='Checkpoint every N epoch')
    parser.add_argument('--classifier', type=str,
                        default="linear", choices=["linear", "wn"])
    parser.add_argument('--data_file', type=str, default='',
                        help='root_dir')
    parser.add_argument('--dataset', type=str, default='OpenBMI-6domain')
    parser.add_argument('--data_dir', type=str, default='./data/OpenBMI/filteredMat/', help='data dir')
    parser.add_argument('--dis_hidden', type=int,
                        default=256, help='dis hidden dimension')
    parser.add_argument('--disttype', type=str, default='cos',
                        choices=['1-norm', '2-norm', 'cos', 'norm-2-norm', 'norm-1-norm'])
    parser.add_argument('--gpu_id', type=str, nargs='?',
                        default='6', help="device id to run")
    parser.add_argument('--groupdro_eta', type=float,
                        default=1, help="groupdro eta")
    parser.add_argument('--inner_lr', type=float,
                        default=1e-2, help="learning rate used in MLDG")
    parser.add_argument('--L', type=float,
                        default=0.2, help="tradeoff hyperparameter used in data augment")
    parser.add_argument('--lam', type=float,
                        default=1, help="tradeoff hyperparameter used in VREx")
    parser.add_argument('--layer', type=str, default="bn",
                        choices=["ori", "bn"])
    parser.add_argument('--lr', type=float, default=5e-3, help="learning rate")
    parser.add_argument('--lr_decay', type=float, default=0.75, help='for sgd')
    parser.add_argument('--lr_decay1', type=float,
                        default=1.0, help='for pretrained featurizer')
    parser.add_argument('--lr_decay2', type=float, default=1.0,
                        help='inital learning rate decay of network')
    parser.add_argument('--lr_gamma', type=float,
                        default=0.0003, help='for optimizer')
    parser.add_argument('--max_epoch', type=int,
                        default=200, help="max iterations")
    parser.add_argument('--mixupalpha', type=float,
                        default=0.2, help='mixup hyper-param')
    parser.add_argument('--mldg_beta', type=float,
                        default=1, help="mldg hyper-param")
    parser.add_argument('--mmd_gamma', type=float,
                        default=1, help='MMD, CORAL hyper-param')
    parser.add_argument('--momentum', type=float,
                        default=0.9, help='for optimizer')
    parser.add_argument('--net', type=str, default='EEGNet',  # DeepConveNet _4 _6
                        help="featurizer: vgg16, resnet50, resnet101, DTNBase")
    parser.add_argument('--N_WORKERS', type=int, default=4)
    parser.add_argument('--rsc_f_drop_factor', type=float,
                        default=1/3, help='rsc hyper-param')
    parser.add_argument('--rsc_b_drop_factor', type=float, 
                        default=1/3, help='rsc hyper-param')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    parser.add_argument('--schuse', action='store_true')
    parser.add_argument('--schusech', type=str, default='cos')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--split_style', type=str, default='strat',
                        help="the style to split the train and eval datasets")
    parser.add_argument('--task', type=str, default="img_dg",
                        choices=["img_dg"], help='now only support image tasks')
    parser.add_argument('--tau', type=float, default=1, help="andmask tau")
    parser.add_argument('--test_envs', type=int, nargs='+',
                        default=[6], help='target domains') # OpenBMI-4 -6 -9
    parser.add_argument('--output', type=str,
                        default="train_output/", help='result output path')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    args = parser.parse_args()
    args.steps_per_epoch = 100
    args.data_dir = args.data_file+args.data_dir
    current_date = datetime.datetime.now().strftime('%Y%m%d')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    os.makedirs(args.output+current_date , exist_ok=True)
 
    random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    filename_out = f"{'out'}_{current_time}_{args.net}_{args.dataset}_{args.algorithm}_{random_string}_{'.txt'}"
    filename_err = f"{'err'}_{current_time}_{args.net}_{args.dataset}_{args.algorithm}_{random_string}_{'.txt'}"
    sys.stdout = Tee(os.path.join(args.output+current_date, filename_out))
    sys.stderr = Tee(os.path.join(args.output+current_date, filename_err))

    args = eeg_param_init(args)
    print_environ()
    print(filename_out) 
    return args


if __name__ == '__main__':
    args = get_args()
    set_random_seed(args.seed)

    loss_list = alg_loss_dict(args)
    train_loaders, eval_loaders = get_eeg_dataloader(args)
    eval_name_dict = train_valid_target_eval_names(args) # dict including train domain, eval domain and test domain.
    algorithm_class = alg.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(args).cuda()
    algorithm.train()
    opt = get_optimizer(algorithm, args)
    sch = get_scheduler(opt, args)

    s = print_args(args, [])
    print('=======hyper-parameter used========')
    print(s)

    if 'Knife' in args.algorithm:
        ms = time.time()
        n_steps = args.max_epoch*args.steps_per_epoch
        print('start training fft teacher net')
        opt1 = get_optimizer(algorithm.teaNet, args, isteacher=True)
        sch1 = get_scheduler(opt1, args)
        algorithm.teanettrain(train_loaders, n_steps, opt1, sch1)
        print('complet time:%.4f' % (time.time()-ms))

    acc_record = {}
    acc_type_list = ['train', 'valid', 'target']
    train_minibatches_iterator = zip(*train_loaders)
    best_valid_acc, target_acc = 0, 0
    print('===========start training===========')
    sss = time.time()
    for epoch in range(args.max_epoch):
        for iter_num in range(args.steps_per_epoch):
            minibatches_device = [(data)
                                  for data in next(train_minibatches_iterator)]
            if args.algorithm == 'VREx' and algorithm.update_count == args.anneal_iters:
                opt = get_optimizer(algorithm, args)
                sch = get_scheduler(opt, args)
            step_vals = algorithm.update(minibatches_device, opt, sch)

        if (epoch in [int(args.max_epoch*0.7), int(args.max_epoch*0.9)]) and (not args.schuse):
            print('manually descrease lr')
            for params in opt.param_groups:
                params['lr'] = params['lr']*0.1

        if (epoch == (args.max_epoch-1)) or (epoch % args.checkpoint_freq == 0):
            print('===========epoch %d===========' % (epoch))
            s = ''
            for item in acc_type_list:
                if item!='target':

                    acc_record[item] = np.mean(np.array( [modelopera.accuracy(algorithm, eval_loaders[i], args, item)[0] for i in eval_name_dict[item]]))
                    s += (item+'_acc:%.4f,' % acc_record[item])
                else:
                    acc_tmp =[modelopera.accuracy(algorithm, eval_loaders[i], args, item) for i in eval_name_dict[item]]
                    acc_record[item] = acc_tmp[0][0]
                    acc_subjects = acc_tmp[0][1]
                    s += (item+'_acc:%.4f,' % acc_record[item])
            print(s[:-1])
            if acc_record['valid'] > best_valid_acc:
                best_valid_acc = acc_record['valid']
                target_acc = acc_record['target']
                top_acc_subjects = acc_subjects
            if args.save_model_every_checkpoint:
                save_checkpoint(f'model_epoch{epoch}.pkl', algorithm, args)
            print('total cost time: %.4f' % (time.time()-sss))
            algorithm_dict = algorithm.state_dict()

    mean_acc_subjects = np.mean(top_acc_subjects)
    var_acc_subjects = np.std(top_acc_subjects)
    save_checkpoint('model.pkl', algorithm, args)

    print('valid acc: %.4f' % best_valid_acc)
    print('DG result: %.4f' % target_acc)
    print('mean_acc_subjects: %.4f' % mean_acc_subjects)
    print('var_acc_subjects: %.4f' % var_acc_subjects)
    print('acc_per_subjects:', top_acc_subjects)

    random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    filename = f"{'done'}_{current_time}_{random_string}_{'.txt'}"

    with open(os.path.join(args.output, filename), 'w') as f:
        f.write('done\n')
        f.write('total cost time:%s\n' % (str(time.time()-sss)))
        f.write('valid acc:%.4f\n' % (best_valid_acc))
        f.write('target acc:%.4f' % (target_acc))
