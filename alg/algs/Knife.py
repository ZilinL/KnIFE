# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import numpy as np

from alg.modelopera import get_fea, get_teach_fea
from network import common_network
from alg.algs.base import Algorithm


class Knife(Algorithm):
    def __init__(self, args):
        super(Knife, self).__init__(args)
        self.args = args
        self.featurizer = get_fea(args)
        self.classifier = common_network.feat_classifier(
            args.num_classes, self.featurizer.output_feature_dim(), args.classifier)

        self.teaf = get_fea(args)
        self.teac = common_network.feat_classifier(
            args.num_classes, self.teaf.output_feature_dim(), args.classifier)
        self.teaNet = nn.Sequential(
            self.teaf,
            self.teac
        )

    def teanettrain(self, dataloaders, epochs, opt1, sch1):
        self.teaNet.train()
        minibatches_iterator = zip(*dataloaders)
        for epoch in range(epochs):
            minibatches = [(tdata) for tdata in next(minibatches_iterator)]
            all_x = torch.cat([data[0].cuda().float() for data in minibatches])
            all_y = torch.cat([data[1].cuda().long() for data in minibatches])
            all_z = torch.angle(torch.fft.fftn(all_x, dim=(2, 3)))
            all_p = self.teaNet(all_z)
            loss = F.cross_entropy(all_p, all_y, reduction='mean')
            opt1.zero_grad()
            loss.backward()
            if ((epoch+1) % (int(self.args.steps_per_epoch*self.args.max_epoch*0.7)) == 0 or (epoch+1) % (int(self.args.steps_per_epoch*self.args.max_epoch*0.9)) == 0) and (not self.args.schuse):
                for param_group in opt1.param_groups:
                    param_group['lr'] = param_group['lr']*0.1
            opt1.step()
            if sch1:
                sch1.step()

            if epoch % int(self.args.steps_per_epoch) == 0 or epoch == epochs-1:
                print('epoch: %d, cls loss: %.4f' % (epoch, loss))
        self.teaNet.eval()

    def coral(self, x, y):
        mean_x = x.mean(0, keepdim=True)
        mean_y = y.mean(0, keepdim=True)
        cent_x = x - mean_x
        cent_y = y - mean_y
        cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
        cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

        mean_diff = (mean_x - mean_y).pow(2).mean()
        cova_diff = (cova_x - cova_y).pow(2).mean()

        return mean_diff + cova_diff

    def update(self, minibatches, opt, sch):
        all_x = torch.cat([data[0].cuda().float() for data in minibatches])
        all_y = torch.cat([data[1].cuda().long() for data in minibatches])       
        with torch.no_grad():
            all_x1 = torch.angle(torch.fft.fftn(all_x, dim=(2, 3)))
            tfea = self.teaf(all_x1).detach()

        all_z = self.featurizer(all_x)
        loss1 = F.cross_entropy(self.classifier(all_z), all_y)
        loss2 = F.mse_loss(all_z, tfea)*self.args.alpha
        loss3 = 0
        loss4 = 0
        loss4_1 = 0
        loss4_2 = 0
        if len(minibatches) > 1:
            for i in range(len(minibatches)-1):
                for j in range(i+1, len(minibatches)):
                    domain1_amp, domain1_pha = self.extract_ampl_phase(all_x[i*self.args.batch_size:(i+1)*self.args.batch_size,:,:,:])
                    domain2_amp, domain2_pha = self.extract_ampl_phase(all_x[j*self.args.batch_size:(j+1)*self.args.batch_size,:,:,:])
                    domain1_aug, domain2_aug = self.spectrum_mix(domain1_amp, domain1_pha, domain2_amp, domain2_pha)
                    domain1_y = all_y[i*self.args.batch_size:(i+1)*self.args.batch_size]
                    domain2_y = all_y[j*self.args.batch_size:(j+1)*self.args.batch_size]
                    domain1_fea = self.featurizer(domain1_aug)
                    domain1_pre = self.classifier(domain1_fea)
                    domain2_fea = self.featurizer(domain2_aug)
                    domain2_pre = self.classifier(domain2_fea)
                    loss_aug = F.cross_entropy(domain1_pre, domain1_y) + F.cross_entropy(domain2_pre, domain2_y)
                    loss3 += loss_aug
                    
                    loss4_1 += self.coral(all_z[i*self.args.batch_size:(i+1)*self.args.batch_size, :],
                                        all_z[j*self.args.batch_size:(j+1)*self.args.batch_size, :])
                    loss4_2 += self.coral(domain1_fea, domain2_fea)
                    

            loss3 = loss3*2/(len(minibatches) *
                             (len(minibatches)-1))
            loss4 = 0.5*(loss4_1+loss4_2)
            loss4 = loss4*2/(len(minibatches) *
                             (len(minibatches)-1))*self.args.lam
        else:
            loss4 = self.coral(all_z[:self.args.batch_size//2, :],
                               all_z[self.args.batch_size//2:, :])
            loss4 = loss4*self.args.lam

        loss = 0.5*loss1+loss2+0.5*loss3+loss4
        opt.zero_grad()
        loss.backward()
        opt.step()
        if sch:
            sch.step()
        return {'class': loss1.item(), 'dist': (loss2).item(), 'aug': (loss3).item(), 'align': loss4.item(), 'total': loss.item()}

    def extract_ampl_phase(self, x):
        fft_x = torch.fft.rfft(x)
        # fft_x: size should be b x 1 x channels x points       
        fft_x_amp = fft_x.real**2 + fft_x.imag**2
        fft_x_amp = torch.sqrt(fft_x_amp)
        fft_x_pha = torch.atan2( fft_x.imag, fft_x.real ) 
        return fft_x_amp, fft_x_pha
    
    def spectrum_mix(self, domain1_amp, domain1_pha, domain2_amp, domain2_pha):
        # swap
        L = self.args.L
        _, _, _, p = domain1_amp.size()
        b = (  np.floor(p*L)  ).astype(int)     # get b
        p_start = p // 2 - b // 2
        tmp = domain1_amp.clone()
        domain1_amp[:,:,:,p_start:p_start+b] = domain2_amp[:,:,:,p_start:p_start+b]  
        domain2_amp[:,:,:,p_start:p_start+b] = tmp[:,:,:,p_start:p_start+b]

        # domain1_real, domain1_imag, domain2_real, domain2_imag = torch.zeros( domain1_amp.size(), dtype=torch.float )
        domain1_real = torch.cos(domain1_pha.clone()) * domain1_amp.clone()
        domain1_imag = torch.sin(domain1_pha.clone()) * domain1_amp.clone()

        domain2_real = torch.cos(domain2_pha.clone()) * domain2_amp.clone()
        domain2_imag = torch.sin(domain2_pha.clone()) * domain2_amp.clone()

        domain1_fft = torch.complex(domain1_real, domain1_imag)
        domain2_fft = torch.complex(domain2_real, domain2_imag)

        domain1_new = torch.fft.irfft(domain1_fft) 
        domain2_new = torch.fft.irfft(domain2_fft)

        return domain1_new, domain2_new

    def predict(self, x):
        return self.classifier(self.featurizer(x))