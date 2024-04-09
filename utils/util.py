# coding=utf-8
import random
import numpy as np
import torch
import sys
import os
import torchvision
import PIL


def set_random_seed(seed=0):
    # seed setting
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(filename, alg, args):
    save_dict = {
        "args": vars(args),
        "model_dict": alg.cpu().state_dict()
    }
    torch.save(save_dict, os.path.join(args.output, filename))


def train_valid_target_eval_names(args):
    eval_name_dict = {'train': [], 'valid': [], 'target': []}
    t = 0
    for i in range(args.domain_num):
        if i not in args.test_envs:
            eval_name_dict['train'].append(t)
            t += 1
    for i in range(args.domain_num):
        if i not in args.test_envs:
            eval_name_dict['valid'].append(t)
        else:
            eval_name_dict['target'].append(t)
        t += 1
    return eval_name_dict


def alg_loss_dict(args):
    loss_dict = {'ANDMask': ['total'],
                 'CORAL': ['class', 'coral', 'total'],
                 'DANN': ['class', 'dis', 'total'],
                 'ERM': ['class'],
                 'ERM_fft_test': ['class'],
                 'Mixup': ['class'],
                 'MLDG': ['total'],
                 'MMD': ['class', 'mmd', 'total'],
                 'GroupDRO': ['group'],
                 'RSC': ['class'],
                 'VREx': ['loss', 'nll', 'penalty'],
                 'Knife': ['class', 'dist', 'aug', 'align', 'total'],
                 }
    return loss_dict[args.algorithm]


def print_args(args, print_list):
    s = "==========================================\n"
    l = len(print_list)
    for arg, content in args.__dict__.items():
        if l == 0 or arg in print_list:
            s += "{}:{}\n".format(arg, content)
    return s


def print_environ():
    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))


class Tee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def eeg_param_init(args):
    dataset = args.dataset
    if dataset == 'BCICIV-2a-3domain':
        domains = [['A1', 'A2', 'A3'], ['A4', 'A5', 'A6'], ['A7', 'A8', 'A9'],['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9']]
    elif dataset == 'BCICIII-IVa-5domain':
        domains = [['A1'], ['A2'], ['A3'], ['A4'], ['A5']]
    elif dataset == 'BCICIV-2a-9domain':
        domains = [['A1'], ['A2'], ['A3'], ['A4'], ['A5'], ['A6'], ['A7'], ['A8'], ['A9'], ['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9']]
    elif dataset == 'BCICIV-2a-5domain':
        domains = [['A1', 'A2'], ['A3', 'A4'], ['A5', 'A6'], ['A7', 'A8'], ['A9'], ['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9']]
    if dataset == 'BCICIV-2b-3domain':
        domains = [['A1', 'A2', 'A3'], ['A4', 'A5', 'A6'], ['A7', 'A8', 'A9'],['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9']]
    elif dataset == 'BCICIV-2b-9domain':
        domains = [['A1'], ['A2'], ['A3'], ['A4'], ['A5'], ['A6'], ['A7'], ['A8'], ['A9'], ['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9']]
    elif dataset == 'BCICIV-2a-3domain-EA':
        domains = [['A1'], ['A2'], ['A3'], ['A4'], ['A5'], ['A6'], ['A7'], ['A8'], ['A9'], ['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9']]
    elif dataset == 'OpenBMI-4domain':
        domains = [['s001' 's002' 's003' 's004' 's005' 's006' 's007' 's008' 's009' 's010' 's011' 's012' 's013' 's014'],  # domain 1
                   ['s015' 's016' 's017' 's018' 's019' 's020' 's021' 's022' 's023' 's024' 's025' 's026' 's027' 's028'],  # domain 2   
                   ['s029' 's030' 's031' 's032' 's033' 's034' 's035' 's036' 's037' 's038' 's039' 's040' 's041' 's042'],  # domain 3
                   ['s043' 's044' 's045' 's046' 's047' 's048' 's049' 's050' 's051' 's052' 's053' 's054'],                # domain 4
                   ['se001' 'se002' 'se003' 'se004' 'se005' 'se006' 'se007' 'se008' 'se009' 'se010' 'se011' 'se012' 'se013' 'se014'    # target domain 5
                    'se015' 'se016' 'se017' 'se018' 'se019' 'se020' 'se021' 'se022' 'se023' 'se024' 'se025' 'se026' 'se027' 'se028'
                    'se029' 'se030' 'se031' 'se032' 'se033' 'se034' 'se035' 'se036' 'se037' 'se038' 'se039' 'se040' 'se041' 'se042'
                    'se043' 'se044' 'se045' 'se046' 'se047' 'se048' 'se049' 'se050' 'se051' 'se052' 'se053' 'se054']]
    elif dataset == 'OpenBMI-6domain':
        domains = [['s001', 's002', 's003', 's004', 's005', 's006', 's007', 's008', 's009'],   # domain 1
                   ['s010', 's011', 's012', 's013', 's014', 's015', 's016', 's017', 's018'],   # domain 2
                   ['s019', 's020', 's021', 's022', 's023', 's024', 's025', 's026', 's027'],   # domain 3
                   ['s028', 's029', 's030', 's031', 's032', 's033', 's034', 's035', 's036'],   # domain 4
                   ['s037', 's038', 's039', 's040', 's041', 's042', 's043', 's044', 's045'],   # domain 5
                   ['s046', 's047', 's048', 's049', 's050', 's051', 's052', 's053', 's054'],   # domain 6           
                   ['se001', 'se002', 'se003', 'se004', 'se005', 'se006', 'se007', 'se008','se009', 'se010', 'se011', 'se012', 'se013', 'se014',    # target domain 7
                   'se015', 'se016', 'se017', 'se018', 'se019', 'se020', 'se021', 'se022', 'se023', 'se024', 'se025', 'se026', 'se027', 'se028',
                    'se029', 'se030', 'se031', 'se032', 'se033', 'se034', 'se035', 'se036', 'se037', 'se038', 'se039', 'se040', 'se041', 'se042',
                    'se043', 'se044', 'se045', 'se046', 'se047', 'se048', 'se049', 'se050', 'se051', 'se052', 'se053', 'se054']]
    elif dataset == 'OpenBMI-9domain':
        domains = [['s001', 's002', 's003', 's004', 's005', 's006'],   # domain 1
                   ['s007', 's008', 's009', 's010', 's011', 's012'],   # domain 2
                   ['s013', 's014', 's015', 's016', 's017', 's018'],   # domain 3
                   ['s019', 's020', 's021', 's022', 's023', 's024'],   # domain 4
                   ['s025', 's026', 's027', 's028', 's029', 's030'],   # domain 5
                   ['s031', 's032', 's033', 's034', 's035', 's036'],   # domain 6
                   ['s037', 's038', 's039', 's040', 's041', 's042'],   # domain 7
                   ['s043', 's044', 's045', 's046', 's047', 's048'],   # domain 8 
                   ['s049', 's050', 's051', 's052', 's053', 's054'],   # domain 9          
                   ['se001', 'se002', 'se003', 'se004', 'se005', 'se006', 'se007', 'se008','se009', 'se010', 'se011', 'se012', 'se013', 'se014',    # target domain 10
                   'se015', 'se016', 'se017', 'se018', 'se019', 'se020', 'se021', 'se022', 'se023', 'se024', 'se025', 'se026', 'se027', 'se028',
                    'se029', 'se030', 'se031', 'se032', 'se033', 'se034', 'se035', 'se036', 'se037', 'se038', 'se039', 'se040', 'se041', 'se042',
                    'se043', 'se044', 'se045', 'se046', 'se047', 'se048', 'se049', 'se050', 'se051', 'se052', 'se053', 'se054']]
    elif dataset == 'SEEDIV-3domain':
        domains = [['s1', 's2', 's3', 's4', 's5'], 
                   ['s6', 's7', 's8', 's9', 's10'], 
                   ['s11', 's12', 's13', 's14', 's15'],   
                   ['t1', 't2', 't3', 't4', 't5', 't6', 't7', 't8','t9', 't10', 't11', 't12', 't13', 't14', 't15']]
    else:
        print('No such dataset exists!')
    args.domains = domains
    args.eeg_dataset = {
        'BCICIV-2a-3domain': [['A1', 'A2', 'A3'], ['A4', 'A5', 'A6'], ['A7', 'A8', 'A9'],['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9']],
        'BCICIV-2a-9domain': [['A1'], ['A2'], ['A3'], ['A4'], ['A5'], ['A6'], ['A7'], ['A8'], ['A9'], ['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9']],
        'BCICIV-2b-3domain': [['A1', 'A2', 'A3'], ['A4', 'A5', 'A6'], ['A7', 'A8', 'A9'],['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9']],
        'BCICIV-2b-9domain': [['A1'], ['A2'], ['A3'], ['A4'], ['A5'], ['A6'], ['A7'], ['A8'], ['A9'], ['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9']],
        'BCICIII-IVa-5domain': [['A1'], ['A2'], ['A3'], ['A4'], ['A5']],
        'BCICIV-2a-3domain-EA': [['A1'], ['A2'], ['A3'], ['A4'], ['A5'], ['A6'], ['A7'], ['A8'], ['A9'], ['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9']],
        'BCICIV-2a-5domain': [['A1', 'A2'], ['A3', 'A4'], ['A5', 'A6'], ['A7', 'A8'], ['A9'], ['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9']],
        'OpenBMI-4domain' : [['s001', 's002', 's003', 's004', 's005', 's006', 's007', 's008', 's009', 's010', 's011', 's012', 's013', 's014'],  # domain 1
                   ['s015', 's016', 's017', 's018', 's019', 's020', 's021', 's022', 's023', 's024', 's025', 's026', 's027', 's028'],    # domain 2   
                   ['s029', 's030', 's031', 's032', 's033', 's034', 's035', 's036', 's037', 's038', 's039', 's040', 's041', 's042'],    # domain 3
                   ['s043', 's044', 's045', 's046', 's047', 's048', 's049', 's050', 's051', 's052', 's053', 's054'],                    # domain 4
                   ['se001', 'se002', 'se003', 'se004', 'se005', 'se006', 'se007', 'se008', 'se009', 'se010', 'se011', 'se012', 'se013', 'se014',    # target domain 5
                    'se015', 'se016', 'se017', 'se018', 'se019', 'se020', 'se021', 'se022', 'se023', 'se024', 'se025', 'se026', 'se027', 'se028',
                    'se029', 'se030', 'se031', 'se032', 'se033', 'se034', 'se035', 'se036', 'se037', 'se038', 'se039', 'se040', 'se041', 'se042',
                    'se043', 'se044', 'se045', 'se046', 'se047', 'se048', 'se049', 'se050', 'se051', 'se052', 'se053', 'se054']],
        'OpenBMI-6domain':
                  [['s001', 's002', 's003', 's004', 's005', 's006', 's007', 's008', 's009'],   # domain 1
                   ['s010', 's011', 's012', 's013', 's014', 's015', 's016', 's017', 's018'],   # domain 2
                   ['s019', 's020', 's021', 's022', 's023', 's024', 's025', 's026', 's027'],   # domain 3
                   ['s028', 's029', 's030', 's031', 's032', 's033', 's034', 's035', 's036'],   # domain 4
                   ['s037', 's038', 's039', 's040', 's041', 's042', 's043', 's044', 's045'],   # domain 5
                   ['s046', 's047', 's048', 's049', 's050', 's051', 's052', 's053', 's054'],   # domain 6           
                   ['se001', 'se002', 'se003', 'se004', 'se005', 'se006', 'se007', 'se008','se009', 'se010', 'se011', 'se012', 'se013', 'se014',    # target domain 7
                   'se015', 'se016', 'se017', 'se018', 'se019', 'se020', 'se021', 'se022', 'se023', 'se024', 'se025', 'se026', 'se027', 'se028',
                    'se029', 'se030', 'se031', 'se032', 'se033', 'se034', 'se035', 'se036', 'se037', 'se038', 'se039', 'se040', 'se041', 'se042',
                    'se043', 'se044', 'se045', 'se046', 'se047', 'se048', 'se049', 'se050', 'se051', 'se052', 'se053', 'se054']],
        'OpenBMI-9domain':
                  [['s001', 's002', 's003', 's004', 's005', 's006'],   # domain 1
                   ['s007', 's008', 's009', 's010', 's011', 's012'],   # domain 2
                   ['s013', 's014', 's015', 's016', 's017', 's018'],   # domain 3
                   ['s019', 's020', 's021', 's022', 's023', 's024'],   # domain 4
                   ['s025', 's026', 's027', 's028', 's029', 's030'],   # domain 5
                   ['s031', 's032', 's033', 's034', 's035', 's036'],   # domain 6
                   ['s037', 's038', 's039', 's040', 's041', 's042'],   # domain 7
                   ['s043', 's044', 's045', 's046', 's047', 's048'],   # domain 8 
                   ['s049', 's050', 's051', 's052', 's053', 's054'],   # domain 9          
                   ['se001', 'se002', 'se003', 'se004', 'se005', 'se006', 'se007', 'se008','se009', 'se010', 'se011', 'se012', 'se013', 'se014',    # target domain 10
                   'se015', 'se016', 'se017', 'se018', 'se019', 'se020', 'se021', 'se022', 'se023', 'se024', 'se025', 'se026', 'se027', 'se028',
                    'se029', 'se030', 'se031', 'se032', 'se033', 'se034', 'se035', 'se036', 'se037', 'se038', 'se039', 'se040', 'se041', 'se042',
                    'se043', 'se044', 'se045', 'se046', 'se047', 'se048', 'se049', 'se050', 'se051', 'se052', 'se053', 'se054']],
        'SEEDIV-3domain':
                  [['s1', 's2', 's3', 's4', 's5'], 
                   ['s6', 's7', 's8', 's9', 's10'], 
                   ['s11', 's12', 's13', 's14', 's15'],   
                   ['t1', 't2', 't3', 't4', 't5', 't6', 't7', 't8','t9', 't10', 't11', 't12', 't13', 't14', 't15']]
    }
    if dataset == 'BCICIV-2a-3domain':
        args.input_shape = (22, 750)
        args.num_classes = 4
        args.channels = 22
        args.points = 750
    elif dataset == 'BCICIV-2a-9domain':
        args.input_shape = (22, 750)
        args.num_classes = 4
        args.channels = 22
        args.points = 750
    elif dataset == 'BCICIV-2b-9domain':
        args.input_shape = (3, 750)
        args.num_classes = 2
        args.channels = 3
        args.points = 750
    elif dataset == 'BCICIV-2b-3domain':
        args.input_shape = (3, 750)
        args.num_classes = 2
        args.channels = 3
        args.points = 750
    elif dataset == 'BCICIII-IVa-5domain':
        args.input_shape = (118, 250)
        args.num_classes = 2
        args.channels = 118
        args.points = 250
    elif dataset == 'BCICIV-2a-5domain':
        args.input_shape = (22, 750)
        args.num_classes = 4
        args.channels = 22
        args.points = 750
    elif dataset == 'BCICIV-2a-3domain-EA':
        args.input_shape = (22, 750)
        args.num_classes = 4
        args.channels = 22
        args.points = 750
    elif dataset == 'OpenBMI-4domain':
        args.input_shape = (20, 1000)
        args.num_classes = 2
        args.channels = 20
        args.points = 1000
    elif dataset == 'OpenBMI-6domain':
        args.input_shape = (20, 1000)
        args.num_classes = 2
        args.channels = 20
        args.points = 1000
    elif dataset == 'OpenBMI-9domain':
        args.input_shape = (20, 1000)
        args.num_classes = 2
        args.channels = 20
        args.points = 1000
    elif dataset == 'SEEDIV-3domain':
        args.input_shape = (62, 800)
        args.num_classes = 4
        args.channels = 62
        args.points = 800
    else:
        print('No such dataset exists!')

    return args


def extract_ampl_phase(fft_im):
    # fft_im: size should be bx3xhxwx2
    fft_amp = fft_im[:,:,:,:,0]**2 + fft_im[:,:,:,:,1]**2
    fft_amp = torch.sqrt(fft_amp)
    fft_pha = torch.atan2( fft_im[:,:,:,:,1], fft_im[:,:,:,:,0] )
    return fft_amp, fft_pha

def low_freq_mutate( amp_src, amp_trg, L=0.1 ):
    _, _, h, w = amp_src.size()
    b = (  np.floor(np.amin((h,w))*L)  ).astype(int)     # get b
    amp_src[:,:,0:b,0:b]     = amp_trg[:,:,0:b,0:b]      # top left
    amp_src[:,:,0:b,w-b:w]   = amp_trg[:,:,0:b,w-b:w]    # top right
    amp_src[:,:,h-b:h,0:b]   = amp_trg[:,:,h-b:h,0:b]    # bottom left
    amp_src[:,:,h-b:h,w-b:w] = amp_trg[:,:,h-b:h,w-b:w]  # bottom right
    return amp_src

def FDA_source_to_target(src_img, trg_img, L=0.1):
    # exchange magnitude
    # input: src_img, trg_img

    # get fft of both source and target
    fft_src = torch.rfft( src_img.clone(), signal_ndim=2, onesided=False ) 
    fft_trg = torch.rfft( trg_img.clone(), signal_ndim=2, onesided=False )

    # extract amplitude and phase of both ffts
    amp_src, pha_src = extract_ampl_phase( fft_src.clone())
    amp_trg, pha_trg = extract_ampl_phase( fft_trg.clone())

    # replace the low frequency amplitude part of source with that from target
    amp_src_ = low_freq_mutate( amp_src.clone(), amp_trg.clone(), L=L )

    # recompose fft of source
    fft_src_ = torch.zeros( fft_src.size(), dtype=torch.float )
    fft_src_[:,:,:,:,0] = torch.cos(pha_src.clone()) * amp_src_.clone()
    fft_src_[:,:,:,:,1] = torch.sin(pha_src.clone()) * amp_src_.clone()

    # get the recomposed image: source content, target style
    _, _, imgH, imgW = src_img.size()
    src_in_trg = torch.irfft( fft_src_, signal_ndim=2, onesided=False, signal_sizes=[imgH,imgW] )

    return src_in_trg
