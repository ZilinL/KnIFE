# coding=utf-8
import torch
import torch.nn as nn
from torchvision import models

vgg_dict = {"vgg11": models.vgg11, "vgg13": models.vgg13, "vgg16": models.vgg16, "vgg19": models.vgg19,
            "vgg11bn": models.vgg11_bn, "vgg13bn": models.vgg13_bn, "vgg16bn": models.vgg16_bn, "vgg19bn": models.vgg19_bn}


# ========================================================================================        

class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)
        self.max_norm = max_norm


    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithConstraint, self).forward(x)
    
    
class LazyLinearWithConstraint(nn.LazyLinear):
    def __init__(self, *args, max_norm=1., **kwargs):
        super(LazyLinearWithConstraint, self).__init__(*args, **kwargs)
        self.max_norm = max_norm
        
    
    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return self(x)
    
#Depthwise separable convolution
class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwiseconv = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=bias)
        self.pointwiseconv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=padding, dilation=dilation, groups=1, bias=bias)

    def forward(self, x):
        x = self.depthwiseconv(x)
        x = self.pointwiseconv(x)
        return x

# Depthwise convolution, need to understand.
class DepthwiseConv2D(nn.Conv2d):
    def __init__(self, *arg, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(DepthwiseConv2D, self).__init__(*arg, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(DepthwiseConv2D, self).forward(x)

# ======================================================================================

class EEGNet(nn.Module):  
    def __init__(self, channels, points, kernel_length=128, kernel_length2=64, F1=8, F2=16, D=2, dropout_rate=0.5):
        super(EEGNet, self).__init__()
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.channels = channels
        self.points = points
        self.kernel_length = kernel_length
        self.kernel_length2 = kernel_length2
        self.dropout_rate = dropout_rate

        # Block1
        self.conv1 = nn.Conv2d(1, self.F1, (1,self.kernel_length), padding=0, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(self.F1 )
        self.depthwiseconv = DepthwiseConv2D(self.F1, self.F1*self.D, (self.channels,1), max_norm=1, groups=self.F1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(self.F1*self.D)
        self.activate1 = nn.ELU()
        self.pooling1 = nn.AvgPool2d((1,4), stride=4) #MI和ERN都采用4
        self.dropout1 = nn.Dropout(p=self.dropout_rate)

        #Block2
        self.separableconv = SeparableConv2d(self.F1*self.D, self.F2, (1, self.kernel_length2))
        self.batchnorm3 = nn.BatchNorm2d(self.F2)
        self.activate2 = nn.ELU()
        self.pooling2 = nn.AvgPool2d((1,8), stride=8) #MI和ERN都采用8
        self.dropout2 = nn.Dropout(p=self.dropout_rate)
   
    def forward(self, x):

        x = self.conv1(x) # [1,1,59,300] -> [1,8,59,237]
        x = self.batchnorm1(x)
        x = self.depthwiseconv(x) # -> [1,16,1,237]
        x = self.batchnorm2(x)
        x = self.activate1(x)
        x = self.pooling1(x) # -> [1,16,1,59]
        x = self.dropout1(x)
        
        x = self.separableconv(x) # -> [1,16,1,44]
        x = self.batchnorm3(x)
        x = self.activate2(x)
        x = self.pooling2(x) # -> [1,16,1,5]
        x = self.dropout2(x)
        x = x.view(x.size(0),-1)
        # output feature

        return x
    
    def output_feature_dim(self):
        data_tmp = torch.rand(1, 1, self.channels, self.points)
        EEGNet_tmp = EEGNet(self.channels, self.points, kernel_length=self.kernel_length, kernel_length2=self.kernel_length2, F1=self.F1, F2=self.F2, D=self.D, dropout_rate=0.5)
        EEGNet_tmp.eval()
        _feature_dim = EEGNet_tmp(data_tmp).view(-1,1).shape[0]
        return _feature_dim


##############################################################################################################################################################################################################
from network.spd import SPDTransform, SPDTangentSpace, SPDRectified


class signal2spd(nn.Module):
    # convert signal epoch to SPD matrix
    def __init__(self):
        super().__init__()
        self.dev = torch.device('cpu')
    def forward(self, x):
        
        x = x.squeeze()
        mean = x.mean(axis=-1).unsqueeze(-1).repeat(1, 1, x.shape[-1])
        x = x - mean
        cov = x@x.permute(0, 2, 1)
        cov = cov.to(self.dev)
        cov = cov/(x.shape[-1]-1)
        tra = cov.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
        tra = tra.view(-1, 1, 1)
        cov /= tra
        identity = torch.eye(cov.shape[-1], cov.shape[-1], device=self.dev).to(self.dev).repeat(x.shape[0], 1, 1)
        cov = cov+(1e-5*identity)
        return cov 

class E2R(nn.Module):
    def __init__(self, epochs):
        super().__init__()
        self.epochs = epochs
        self.signal2spd = signal2spd()
    def patch_len(self, n, epochs):
        list_len=[]
        base = n//epochs
        for i in range(epochs):
            list_len.append(base)
        for i in range(n - base*epochs):
            list_len[i] += 1

        if sum(list_len) == n:
            return list_len
        else:
            return ValueError('check your epochs and axis should be split again')
    
    def forward(self, x):
        # x with shape[bs, ch, time]
        list_patch = self.patch_len(x.shape[-1], int(self.epochs))
        x_list = list(torch.split(x, list_patch, dim=-1))
        for i, item in enumerate(x_list):
            x_list[i] = self.signal2spd(item)
        x = torch.stack(x_list).permute(1, 0, 2, 3)
        return x


class AttentionManifold(nn.Module):
    def __init__(self, in_embed_size, out_embed_size):
        super(AttentionManifold, self).__init__()
        
        self.d_in = in_embed_size
        self.d_out = out_embed_size
        self.q_trans = SPDTransform(self.d_in, self.d_out).cpu()
        self.k_trans = SPDTransform(self.d_in, self.d_out).cpu()
        self.v_trans = SPDTransform(self.d_in, self.d_out).cpu()

    def tensor_log(self, t):#4dim
        u, s, v = torch.svd(t)
        return u @ torch.diag_embed(torch.log(s)) @ v.permute(0, 1, 3, 2)
        
    def tensor_exp(self, t):#4dim
        # condition: t is symmetric!
        s, u = torch.linalg.eigh(t)
        return u @ torch.diag_embed(torch.exp(s)) @ u.permute(0, 1, 3, 2)
    def log_euclidean_distance(self, A, B):
        inner_term = self.tensor_log(A) - self.tensor_log(B)
        inner_multi = inner_term @ inner_term.permute(0, 1, 3, 2)
        _, s, _= torch.svd(inner_multi)
        final = torch.sum(s, dim=-1)
        return final

    def LogEuclideanMean(self, weight, cov):
        # cov:[bs, #p, s, s]
        # weight:[bs, #p, #p]
        bs = cov.shape[0]
        num_p = cov.shape[1]
        size = cov.shape[2]
        cov = self.tensor_log(cov).view(bs, num_p, -1)
        output = weight @ cov#[bs, #p, -1]
        output = output.view(bs, num_p, size, size)
        return self.tensor_exp(output)
        
    def forward(self, x, shape=None):
        if len(x.shape)==3 and shape is not None:
            x = x.view(shape[0], shape[1], self.d_in, self.d_in)
        x = x.to(torch.float)# patch:[b, #patch, c, c]
        q_list = []; k_list = []; v_list = []  
        # calculate Q K V
        bs = x.shape[0]
        m = x.shape[1]
        x = x.reshape(bs*m, self.d_in, self.d_in)
        Q = self.q_trans(x).view(bs, m, self.d_out, self.d_out)
        K = self.k_trans(x).view(bs, m, self.d_out, self.d_out)
        V = self.v_trans(x).view(bs, m, self.d_out, self.d_out)

        # calculate the attention score
        Q_expand = Q.repeat(1, V.shape[1], 1, 1)
    
        K_expand = K.unsqueeze(2).repeat(1, 1, V.shape[1], 1, 1 )
        K_expand = K_expand.view(K_expand.shape[0], K_expand.shape[1] * K_expand.shape[2], K_expand.shape[3], K_expand.shape[4])
        
        atten_energy = self.log_euclidean_distance(Q_expand, K_expand).view(V.shape[0], V.shape[1], V.shape[1])
        atten_prob = nn.Softmax(dim=-2)(1/(1+torch.log(1 + atten_energy))).permute(0, 2, 1)#now row is c.c.
        
        # calculate outputs(v_i') of attention module
        output = self.LogEuclideanMean(atten_prob, V)

        output = output.view(V.shape[0], V.shape[1], self.d_out, self.d_out)

        shape = list(output.shape[:2])
        shape.append(-1)

        output = output.contiguous().view(-1, self.d_out, self.d_out)
        return output, shape

class mAtt_bci(nn.Module):
    def __init__(self, epochs):
        super().__init__()
        #FE
        # bs, 1, channel, sample
        self.conv1 = nn.Conv2d(1, 22, (22, 1))
        self.Bn1 = nn.BatchNorm2d(22)
        # bs, 22, 1, sample
        self.conv2 = nn.Conv2d(22, 20, (1, 12), padding=(0, 6))
        self.Bn2   = nn.BatchNorm2d(20)
        
        # E2R
        self.ract1 = E2R(epochs=epochs)
        # riemannian part
        self.att2 = AttentionManifold(20, 18)
        self.ract2  = SPDRectified()
        
        # R2E
        self.tangent = SPDTangentSpace(18)
        self.flat = nn.Flatten()
        # fc
        self.linear = nn.Linear(9*19*epochs, 4, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.Bn1(x)
        x = self.conv2(x)
        x = self.Bn2(x)
        
        x = self.ract1(x)
        x, shape = self.att2(x)
        x = self.ract2(x)
        
        x = self.tangent(x)
        x = x.view(shape[0], shape[1], -1)
        x = self.flat(x)
        x = self.linear(x)
        return x


class mAtt_mamem(nn.Module):
    def __init__(self, epochs):
        super().__init__()
        # FE
        # bs, 1, channel, sample
        self.conv1 = nn.Conv2d(1, 125, (8, 1))
        self.Bn1 = nn.BatchNorm2d(125)
        # bs, 8, 1, sample
        self.conv2 = nn.Conv2d(125, 15, (1, 36), padding=(0, 18))
        self.Bn2   = nn.BatchNorm2d(15)
        
        #E2R
        self.ract1 = E2R(epochs)
        # riemannian part
        self.att2 = AttentionManifold(15, 12)
        self.ract2  = SPDRectified()
        # R2E
        self.tangent = SPDTangentSpace(12)
        self.flat = nn.Flatten()
        # fc
        self.linear = nn.Linear(6*13*epochs, 5, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.Bn1(x)
        x = self.conv2(x)
        x = self.Bn2(x)

        x = self.ract1(x)
        x, shape = self.att2(x)
        x = self.ract2(x)
        
        x = self.tangent(x)
        x = x.view(shape[0], shape[1], -1)
        x = self.flat(x)
        x = self.linear(x)
        return x

class mAtt_cha(nn.Module):
    def __init__(self, epochs):
        super().__init__()
        #FE
        # bs, 1, channel, sample
        self.conv1 = nn.Conv2d(1, 22, (56, 1))
        self.Bn1 = nn.BatchNorm2d(22)
        # bs, 56, 1, sample
        self.conv2 = nn.Conv2d(22, 16, (1, 64), padding=(0, 32))
        self.Bn2   = nn.BatchNorm2d(16)
        
        # E2R
        self.ract1 = E2R(epochs=epochs)
        # riemannian part
        self.att2 = AttentionManifold(16, 8)
        self.ract2  = SPDRectified()
        
        # R2E
        self.tangent = SPDTangentSpace(8)
        self.flat = nn.Flatten()
        # fc
        self.linear = nn.Linear(4*9*epochs, 2, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.Bn1(x)
        x = self.conv2(x)
        x = self.Bn2(x)
        
        x = self.ract1(x)
        x, shape = self.att2(x)
        x = self.ract2(x)
        
        x = self.tangent(x)
        x = x.view(shape[0], shape[1], -1)
        x = self.flat(x)
        x = self.linear(x)
        return x


####################################################################################################################################################################

res_dict = {"resnet18": models.resnet18, "resnet34": models.resnet34, "resnet50": models.resnet50,
            "resnet101": models.resnet101, "resnet152": models.resnet152, "resnext50": models.resnext50_32x4d, "resnext101": models.resnext101_32x8d}


class ResBase(nn.Module):
    def __init__(self, res_name):
        super(ResBase, self).__init__()
        model_resnet = res_dict[res_name](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class DTNBase(nn.Module):
    def __init__(self):
        super(DTNBase, self).__init__()
        self.conv_params = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.3),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.5),
            nn.ReLU()
        )
        self.in_features = 256*4*4

    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        return x


class LeNetBase(nn.Module):
    def __init__(self):
        super(LeNetBase, self).__init__()
        self.conv_params = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.in_features = 50*4*4

    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        return x
