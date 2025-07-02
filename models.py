import torch
from torch import nn
import tsai
from tsai.models.layers import ConvBlock, Add, BN1d, Squeeze, ConvBN, Conv1d, Concat, GAP1d
import torch.nn.functional as F


### Model from Singstad et al.: https://ieeexplore.ieee.org/document/10081878
def inceptiontime_singstad_d3_TS(num_channels=4, num_classes=2,):
    return inceptime_singstad_d3(c_in=num_channels, c_out=num_classes)

def inceptiontime_singstad_d6_TS(num_channels=4, num_classes=2,):
    return inceptime_singstad_d6(c_in=num_channels, c_out=num_classes)

def inceptiontime_singstad_d10_TS(num_channels=4, num_classes=2,):
    return inceptime_singstad_d10(c_in=num_channels, c_out=num_classes)

class inception_module(nn.Module):
    def __init__(self, c_in, stride=1, use_bottleneck=True, kernel_size=40, bottleneck_size=32, nb_filters=32):
        super().__init__()
        self.c_in_original = c_in
        self.c_in = c_in
        self.use_bottleneck = use_bottleneck
        self.conv1 = nn.Conv1d(in_channels=self.c_in_original, out_channels=bottleneck_size, kernel_size=1, stride=stride, padding='same', bias=False)
        if use_bottleneck and c_in > 1:
            self.c_in = bottleneck_size
        self.kernel_size_s = [kernel_size // (2 ** i) for i in range(3)]
        self.conv_s1 = nn.Conv1d(in_channels=self.c_in, out_channels=nb_filters, kernel_size=self.kernel_size_s[0], stride=stride, padding='same', bias=False)
        self.conv_s2 = nn.Conv1d(in_channels=nb_filters, out_channels=nb_filters, kernel_size=self.kernel_size_s[1], stride=stride, padding='same', bias=False)
        self.conv_s3 = nn.Conv1d(in_channels=nb_filters, out_channels=nb_filters, kernel_size=self.kernel_size_s[2], stride=stride, padding='same', bias=False)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=stride, padding=1)
        self.conv6 = nn.Conv1d(in_channels=self.c_in_original, out_channels=nb_filters, kernel_size=1, stride=stride, padding='same', bias=False)
        self.batchnorm = nn.BatchNorm1d(nb_filters*4)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        #print(f'{x.shape=}')
        if self.use_bottleneck and self.c_in_original > 1:
            xc1 = self.conv1(x)
        else:
            xc1 = x
        #print(f'{xc1.shape=}')
        xs1 = self.conv_s1(xc1)
        #print(f'{xs1.shape=}')
        xs2 = self.conv_s2(xc1)
        #print(f'{xs2.shape=}')
        xs3 = self.conv_s3(xc1)
        #print(f'{xs3.shape=}')
        xp = self.maxpool(x)
        #print(f'{xp.shape=}')
        xc6 = self.conv6(xp)
        #print(f'{xc6.shape=}')
        x = torch.cat((xs1, xs2, xs3, xc6), dim=1)
        #print(f'after cat {x.shape=}')
        x = self.batchnorm(x)
        #print(f'after batchnorm {x.shape=}')
        x = self.relu(x)
        #print(f'after relu {x.shape=}')
        return x
    
class Lambda(nn.Module):
    def __init__(self, f):
        super(Lambda, self).__init__()
        self.f = f
    def forward(self, x):
        return self.f(x)
    
class inceptime_singstad_d10(nn.Module):
    def __init__(self, c_in, c_out, stride=1, use_bottleneck=True, kernel_size=40, bottleneck_size=32, nb_filters=32, hid_layers=10):
        super().__init__()
        self.deep1 = inception_module(c_in, stride, use_bottleneck, kernel_size, bottleneck_size, nb_filters)
        self.deep2 = inception_module(nb_filters*4, stride, use_bottleneck, kernel_size, bottleneck_size, nb_filters)
        self.shortcut1 = nn.Sequential(
            nn.Conv1d(in_channels=c_in, out_channels=4*nb_filters, kernel_size=1, padding='same', bias=False),
            nn.BatchNorm1d(4*nb_filters)           
            )
        self.shortcut2 = nn.Sequential(
            nn.Conv1d(in_channels=4*nb_filters, out_channels=4*nb_filters, kernel_size=1, padding='same', bias=False),
            nn.BatchNorm1d(4*nb_filters)           
            )
        self.globalpool = Lambda(f=lambda x: torch.mean(x, dim=-1))
        self.linear = nn.Linear(4*nb_filters, c_out)

    def forward(self, x, depth=None, pass_part=None):
        '''
        #print(f'Input {x.shape=}')
        z = self.deep1(x)
        z = self.deep2(z)
        z = self.deep2(z)
        z = z + self.shortcut1(x)
        z = F.relu(z)
        #print(f'First {z.shape=}')
        w = self.deep2(z)
        w = self.deep2(w)
        w = self.deep2(w)
        w = w + self.shortcut2(z)
        w = F.relu(w)
        #print(f'Second {w.shape=}')
        v = self.globalpool(w)
        #print(f'Global pool {v.shape=}')
        v = self.linear(v)
        #print(f'Output {v.shape=}')
        return v
        '''
        #'''
        if pass_part=='first':
            if depth==0:
                return x
            z = self.deep1(x)
            z = self.deep2(z)
            z = self.deep2(z)
            z = z + self.shortcut1(x)
            z = F.relu(z)
            if depth==1:
                return z
            w = self.deep2(z)
            w = self.deep2(w)
            w = self.deep2(w)
            w = w + self.shortcut2(z)
            w = F.relu(w)
            if depth==2:
                return w
            y = self.deep2(w)
            y = self.deep2(y)
            y = self.deep2(y)
            y = y + self.shortcut2(w)
            y = F.relu(y)
            if depth==3:
                return y
            v = self.deep2(y)
            v = self.globalpool(v)
            v = self.linear(v)
            return v
        elif pass_part=='second':
            if depth<=0:
                z = self.deep1(x)
                z = self.deep2(z)
                z = self.deep2(z)
                z = z + self.shortcut1(x)
                z = F.relu(z)
            if depth<=1:
                w = self.deep2(z)
                w = self.deep2(w)
                w = self.deep2(w)
                w = w + self.shortcut2(z)
                w = F.relu(w)
            if depth<=2:
                y = self.deep2(w)
                y = self.deep2(y)
                y = self.deep2(y)
                y = y + self.shortcut2(w)
                y = F.relu(y)
            if depth <=3:
                v = self.deep2(y)
                v = self.globalpool(v)
                v = self.linear(v)
            return v
        else:
            #print(f'Input {x.shape=}')
            z = self.deep1(x)
            z = self.deep2(z)
            z = self.deep2(z)
            z = z + self.shortcut1(x)
            z = F.relu(z)
            #print(f'First {z.shape=}')
            w = self.deep2(z)
            w = self.deep2(w)
            w = self.deep2(w)
            w = w + self.shortcut2(z)
            w = F.relu(w)
            #print(f'Second {w.shape=}')
            y = self.deep2(w)
            y = self.deep2(y)
            y = self.deep2(y)
            y = y + self.shortcut2(w)
            y = F.relu(y)
            #print(f'Third {y.shape=}')
            v = self.deep2(y)
            #print(f'Forth {v.shape=}')
            v = self.globalpool(v)
            #print(f'Global pool {v.shape=}')
            v = self.linear(v)
            #print(f'Output {v.shape=}')
            return v
            #'''
    
class inceptime_singstad_d6(nn.Module):
    def __init__(self, c_in, c_out, stride=1, use_bottleneck=True, kernel_size=40, bottleneck_size=32, nb_filters=32, hid_layers=10):
        super().__init__()
        self.deep1 = inception_module(c_in, stride, use_bottleneck, kernel_size, bottleneck_size, nb_filters)
        self.deep2 = inception_module(nb_filters*4, stride, use_bottleneck, kernel_size, bottleneck_size, nb_filters)
        self.shortcut1 = nn.Sequential(
            nn.Conv1d(in_channels=c_in, out_channels=4*nb_filters, kernel_size=1, padding='same', bias=False),
            nn.BatchNorm1d(4*nb_filters)           
            )
        self.shortcut2 = nn.Sequential(
            nn.Conv1d(in_channels=4*nb_filters, out_channels=4*nb_filters, kernel_size=1, padding='same', bias=False),
            nn.BatchNorm1d(4*nb_filters)           
            )
        self.globalpool = Lambda(f=lambda x: torch.mean(x, dim=-1))
        self.linear = nn.Linear(4*nb_filters, c_out)

    def forward(self, x, depth=None, pass_part=None):
        #'''
        #print(f'Input {x.shape=}')
        z = self.deep1(x)
        z = self.deep2(z)
        z = self.deep2(z)
        z = z + self.shortcut1(x)
        z = F.relu(z)
        #print(f'First {z.shape=}')
        w = self.deep2(z)
        w = self.deep2(w)
        w = self.deep2(w)
        w = w + self.shortcut2(z)
        w = F.relu(w)
        #print(f'Second {w.shape=}')
        v = self.globalpool(w)
        #print(f'Global pool {v.shape=}')
        v = self.linear(v)
        #print(f'Output {v.shape=}')
        return v
        #'''
        '''
        if pass_part=='first':
            if depth==0:
                return x
            z = self.deep1(x)
            z = self.deep2(z)
            z = self.deep2(z)
            z = z + self.shortcut1(x)
            z = F.relu(z)
            if depth==1:
                return z
            w = self.deep2(z)
            w = self.deep2(w)
            w = self.deep2(w)
            w = w + self.shortcut2(z)
            w = F.relu(w)
            if depth==2:
                return w
            y = self.deep2(w)
            y = self.deep2(y)
            y = self.deep2(y)
            y = y + self.shortcut2(w)
            y = F.relu(y)
            if depth==3:
                return y
            v = self.deep2(y)
            v = self.globalpool(v)
            v = self.linear(v)
            return v
        elif pass_part=='second':
            if depth<=0:
                z = self.deep1(x)
                z = self.deep2(z)
                z = self.deep2(z)
                z = z + self.shortcut1(x)
                z = F.relu(z)
            if depth<=1:
                w = self.deep2(z)
                w = self.deep2(w)
                w = self.deep2(w)
                w = w + self.shortcut2(z)
                w = F.relu(w)
            if depth<=2:
                y = self.deep2(w)
                y = self.deep2(y)
                y = self.deep2(y)
                y = y + self.shortcut2(w)
                y = F.relu(y)
            if depth <=3:
                v = self.deep2(y)
                v = self.globalpool(v)
                v = self.linear(v)
            return v
        else:
            #print(f'Input {x.shape=}')
            z = self.deep1(x)
            z = self.deep2(z)
            z = self.deep2(z)
            z = z + self.shortcut1(x)
            z = F.relu(z)
            #print(f'First {z.shape=}')
            w = self.deep2(z)
            w = self.deep2(w)
            w = self.deep2(w)
            w = w + self.shortcut2(z)
            w = F.relu(w)
            #print(f'Second {w.shape=}')
            y = self.deep2(w)
            y = self.deep2(y)
            y = self.deep2(y)
            y = y + self.shortcut2(w)
            y = F.relu(y)
            #print(f'Third {y.shape=}')
            v = self.deep2(y)
            #print(f'Forth {v.shape=}')
            v = self.globalpool(v)
            #print(f'Global pool {v.shape=}')
            v = self.linear(v)
            #print(f'Output {v.shape=}')
            return v
            '''
        
class inceptime_singstad_d3(nn.Module):
    def __init__(self, c_in, c_out, stride=1, use_bottleneck=True, kernel_size=40, bottleneck_size=32, nb_filters=32, hid_layers=10):
        super().__init__()
        self.deep1 = inception_module(c_in, stride, use_bottleneck, kernel_size, bottleneck_size, nb_filters)
        self.deep2 = inception_module(nb_filters*4, stride, use_bottleneck, kernel_size, bottleneck_size, nb_filters)
        self.shortcut1 = nn.Sequential(
            nn.Conv1d(in_channels=c_in, out_channels=4*nb_filters, kernel_size=1, padding='same', bias=False),
            nn.BatchNorm1d(4*nb_filters)           
            )
        self.shortcut2 = nn.Sequential(
            nn.Conv1d(in_channels=4*nb_filters, out_channels=4*nb_filters, kernel_size=1, padding='same', bias=False),
            nn.BatchNorm1d(4*nb_filters)           
            )
        self.globalpool = Lambda(f=lambda x: torch.mean(x, dim=-1))
        self.linear = nn.Linear(4*nb_filters, c_out)

    def forward(self, x, depth=None, pass_part=None):
        #'''
        #print(f'Input {x.shape=}')
        z = self.deep1(x)
        z = self.deep2(z)
        z = self.deep2(z)
        z = z + self.shortcut1(x)
        z = F.relu(z)
        #print(f'First {z.shape=}')
        v = self.globalpool(z)
        #print(f'Global pool {v.shape=}')
        v = self.linear(v)
        #print(f'Output {v.shape=}')
        return v
        #'''
    
### Potes...
def CNN_potes_big128and64_TS(num_channels=4, num_classes=2, dataset='PhysioNet', dropout=0.25):
    return CNN_potes(c_in=num_channels, c_out=num_classes, layers=[128,64], linear=159488, dropout=dropout)

def CNN_potes_big64and32_TS(num_channels=4, num_classes=2, dataset='PhysioNet', dropout=0.25):
    return CNN_potes(c_in=num_channels, c_out=num_classes, layers=[64,32], linear=79744, dropout=dropout)

def CNN_potes_TS(num_channels=4, num_classes=2, dataset='PhysioNet', dropout=0.25):
    if dataset == 'PhysioNet':
        linear = 9968
    elif dataset == 'UMC':
        linear = 7968
    return CNN_potes(c_in=num_channels, c_out=num_classes, layers=[8,4], linear=linear, dropout=dropout)

def CNN_potes_tenpercent_TS(num_channels=4, num_classes=2):
    return CNN_potes(c_in=num_channels, c_out=num_classes, layers=[2,1], linear=2492)

def CNN_potes_twopercent_TS(num_channels=4, num_classes=2):
    return CNN_potes(c_in=num_channels, c_out=num_classes, layers=[1,1], linear=2492)

### Model from Potes et al.: https://www.cinc.org/archives/2016/pdf/182-399.pdf
def conv_block_1d(in_channels, out_channels, ks=3, pad=1, pool=False, dropout=0.):
    layers = [nn.Conv1d(in_channels, out_channels, kernel_size=ks, padding=pad), 
              #nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool1d(2))
    if dropout: layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)

class CNN_potes(nn.Module):
    def __init__(self, c_in, c_out, layers, linear, dropout=0.25):
        super().__init__()
        self.cnn1 = nn.Sequential(conv_block_1d(1, layers[0], ks=5, pad=1, pool=True), conv_block_1d(layers[0], layers[1], ks=5, pad=1, pool=True, dropout=dropout))
        self.cnn2 = nn.Sequential(conv_block_1d(1, layers[0], ks=5, pad=1, pool=True), conv_block_1d(layers[0], layers[1], ks=5, pad=1, pool=True, dropout=dropout))
        self.cnn3 = nn.Sequential(conv_block_1d(1, layers[0], ks=5, pad=1, pool=True), conv_block_1d(layers[0], layers[1], ks=5, pad=1, pool=True, dropout=dropout))
        self.cnn4 = nn.Sequential(conv_block_1d(1, layers[0], ks=5, pad=1, pool=True), conv_block_1d(layers[0], layers[1], ks=5, pad=1, pool=True, dropout=dropout))
        # nn.Flatten() has no learnable parameters, yet it has to propagate the gradients to previous layers
        self.flat1 = nn.Flatten()
        self.flat2 = nn.Flatten() 
        self.flat3 = nn.Flatten() 
        self.flat4 = nn.Flatten() 
        self.dimreduc = nn.Linear(linear, 20)
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(20, c_out)

    def forward(self, x, depth=None, pass_part=None):
        if pass_part=='first':
            if depth==0:
                return x
            x1 = x[:, 0, :][:, None, :]
            x1 = self.cnn1(x1)
            x1 = self.flat1(x1)
            x2 = x[:, 1, :][:, None, :]
            x2 = self.cnn1(x2)
            x2 = self.flat2(x2)
            x3 = x[:, 2, :][:, None, :]
            x3 = self.cnn1(x3)
            x3 = self.flat3(x3)
            x4 = x[:, 3, :][:, None, :]
            x4 = self.cnn1(x4)
            x4 = self.flat4(x4)
            # Concat the flattened tensors
            x = torch.cat((x1, x2, x3, x4), dim=1)
            x = F.relu(self.dimreduc(x))
            x = self.dropout(x)
            if depth==1:
                return x
        elif pass_part=='second':
            if depth <= 0:
                x1 = x[:, 0, :][:, None, :]
                x1 = self.cnn1(x1)
                x1 = self.flat1(x1)
                x2 = x[:, 1, :][:, None, :]
                x2 = self.cnn1(x2)
                x2 = self.flat2(x2)
                x3 = x[:, 2, :][:, None, :]
                x3 = self.cnn1(x3)
                x3 = self.flat3(x3)
                x4 = x[:, 3, :][:, None, :]
                x4 = self.cnn1(x4)
                x4 = self.flat4(x4)
                x = torch.cat((x1, x2, x3, x4), dim=1)
                x = F.relu(self.dimreduc(x))
                x = self.dropout(x)
            if depth <= 1:
                x = self.linear(x)
            return x
        elif pass_part=='latent_space':
            x1 = x[:, 0, :][:, None, :]
            x1 = self.cnn1(x1)
            x1 = self.flat1(x1)
            x2 = x[:, 1, :][:, None, :]
            x2 = self.cnn1(x2)
            x2 = self.flat2(x2)
            x3 = x[:, 2, :][:, None, :]
            x3 = self.cnn1(x3)
            x3 = self.flat3(x3)
            x4 = x[:, 3, :][:, None, :]
            x4 = self.cnn1(x4)
            x4 = self.flat4(x4)
            x = torch.cat((x1, x2, x3, x4), dim=1)
            x = F.relu(self.dimreduc(x))
            x = self.dropout(x)
            return x
        else:
            #print(f'Input {x.shape=}')
            x1 = x[:, 0, :][:, None, :]
            x1 = self.cnn1(x1)
            x1 = self.flat1(x1)
            x2 = x[:, 1, :][:, None, :]
            x2 = self.cnn1(x2)
            x2 = self.flat2(x2)
            x3 = x[:, 2, :][:, None, :]
            x3 = self.cnn1(x3)
            x3 = self.flat3(x3)
            x4 = x[:, 3, :][:, None, :]
            x4 = self.cnn1(x4)
            x4 = self.flat4(x4)
            # Concat the flattened tensors
            x = torch.cat((x1, x2, x3, x4), dim=1)
            #print(f'Flattened {x.shape=}')
            x = F.relu(self.dimreduc(x))
            #print(f'Dimreduc {x.shape=}')
            x = self.dropout(x)
            #print(f'Hidden {x.shape=}')
            x = self.linear(x)
            #print(f'Classif {x.shape=}')
            return x

# myrtle resnet9, 1d convolutions instead of 2d
def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm1d(out_channels), 
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool1d(2))
    return nn.Sequential(*layers)

class ResNet9_myrtle(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        
        #self.classifier = nn.Sequential(nn.MaxPool2d(4), 
        #                                nn.Flatten(), 
        #                                nn.Linear(512, num_classes))
        self.pool1d = nn.MaxPool1d(4)
        self.flat = nn.Flatten()
        self.linear = nn.Linear(39936, num_classes) # for 128x128

    def forward(self, out, depth=None, pass_part=None):
            #print('Input:', out.shape)
            out = self.conv1(out)
            #print('First conv:', out.shape)
            out = self.conv2(out)
            #print('Second conv:', out.shape)
            out = self.res1(out) + out
            #print('First res:', out.shape)
            out = self.conv3(out)
            #print('Third conv:', out.shape)
            out = self.conv4(out)
            #print('Forth conv:', out.shape)
            out = self.res2(out) + out
            #print('Second res:', out.shape)
            out = self.pool1d(out)
            #print('Pool before flat:', out.shape)
            out = self.flat(out)
            #print('Flat:', out.shape)
            out = self.linear(out)
            #print('Output:', out.shape)
            return out
    
def ResNet9(in_channels, num_classes):
    return ResNet9_myrtle(in_channels=in_channels, num_classes=num_classes)

### Resnet9 5k to 10m parameters
class ResNet9_myrtle(nn.Module):
    def __init__(self, in_channels, num_classes, filters, linear):
        super().__init__()
        self.conv1 = conv_block(in_channels, filters[0])
        self.conv2 = conv_block(filters[0], filters[1], pool=True)
        self.res1 = nn.Sequential(conv_block(filters[1], filters[1]), conv_block(filters[1], filters[1]))
        self.conv3 = conv_block(filters[1], filters[2], pool=True)
        self.conv4 = conv_block(filters[2], filters[3], pool=True)
        self.res2 = nn.Sequential(conv_block(filters[3], filters[3]), conv_block(filters[3], filters[3]))
        self.pool1d = nn.MaxPool1d(4)
        self.flat = nn.Flatten()
        self.linear = nn.Linear(linear, num_classes) 
    def forward(self, out, depth=None, pass_part=None):
        if pass_part=='first':
            if depth==0:
                return out
            out = self.conv1(out)
            out = self.conv2(out)
            out = self.res1(out) + out
            if depth==1:
                return out
            out = self.conv3(out)
            out = self.conv4(out)
            out = self.res2(out) + out
            if depth==2:
                return out
            out = self.pool1d(out)
            out = self.flat(out)
            if depth==3:
                return out
            out = self.linear(out)
            return out
        elif pass_part=='second':
            if depth<=0:
                out = self.conv1(out)
                out = self.conv2(out)
                out = self.res1(out) + out
            if depth<=1:
                out = self.conv3(out)
                out = self.conv4(out)
                out = self.res2(out) + out
            if depth<=2:
                out = self.pool1d(out)
                out = self.flat(out)
            if depth<=3:
                out = self.linear(out)
            return out
        else:
            #print('Input:', out.shape)
            out = self.conv1(out)
            #print('First conv:', out.shape)
            out = self.conv2(out)
            #print('Second conv:', out.shape)
            out = self.res1(out) + out
            #print('First res:', out.shape)
            out = self.conv3(out)
            #print('Third conv:', out.shape)
            out = self.conv4(out)
            #print('Forth conv:', out.shape)
            out = self.res2(out) + out
            #print('Second res:', out.shape)
            out = self.pool1d(out)
            #print('Pool before flat:', out.shape)
            out = self.flat(out)
            #print('Flat:', out.shape)
            out = self.linear(out)
            #print('Output:', out.shape)
            return out
def ResNet9(in_channels, num_classes, filters=[64, 128, 256, 512], linear=39936):
    return ResNet9_myrtle(in_channels=in_channels, num_classes=num_classes, filters=filters, linear=linear)

### taken from: https://github.com/timeseriesAI/tsai/blob/main/tsai/models/FCN.py
def FCN_TS_custom(num_channels=4, num_classes=2):
    return FCN_custom(c_in=num_channels, c_out=num_classes)

class FCN_custom(nn.Module):
    def __init__(self, c_in, c_out, layers=[32*2, 64*2, 32*2], kss=[7, 5, 3]):
        super().__init__()
        assert len(layers) == len(kss)
        self.convblock1 = tsai.models.layers.ConvBlock(c_in, layers[0], kss[0])
        self.convblock2 = tsai.models.layers.ConvBlock(layers[0], layers[1], kss[1])
        self.convblock3 = tsai.models.layers.ConvBlock(layers[1], layers[2], kss[2])
        self.gap = tsai.models.layers.GAP1d(1)
        self.fc = nn.Linear(layers[-1], c_out)

    def forward(self, x, depth=None, pass_part=None):
        if pass_part=='first':
            if depth==0:
                return x
            x = self.convblock1(x)
            if depth==1:
                return x
            x = self.convblock2(x)
            if depth==2:
                return x
            x = self.convblock3(x)
            if depth==3:
                return x
            x = self.gap(x)
            if depth ==4:
                return x
            x = self.fc(x)
            return x
        elif pass_part=='second':
            if depth <= 0:
                x = self.convblock1(x)
            if depth <=1:
                x = self.convblock2(x)
            if depth <=2:
                x = self.convblock3(x)
            if depth <=3:
                x = self.gap(x)
            if depth <=4:
                x = self.fc(x)
                return x
        elif pass_part=='latent_space':
            x = self.convblock1(x)
            x = self.convblock2(x)
            x = self.convblock3(x)
            x = self.gap(x)
            return x
        else:
            #print(f'Input {x.shape=}')
            x = self.convblock1(x)
            #print(f'Block1 {x.shape=}')
            x = self.convblock2(x)
            #print(f'Block2 {x.shape=}')
            x = self.convblock3(x)
            #print(f'Block3 {x.shape=}')
            x = self.gap(x)
            #print(f'Gap {x.shape=}')
            x = self.fc(x)
            #print(f'Linear {x.shape=}')
            return x


### taken from: https://github.com/timeseriesAI/tsai/blob/main/tsai/models/FCN.py
def FCN_TS(num_channels=4, num_classes=2):
    return FCN(c_in=num_channels, c_out=num_classes)

class FCN(nn.Module):
    def __init__(self, c_in, c_out, layers=[128, 256, 128], kss=[7, 5, 3]):
        super().__init__()
        assert len(layers) == len(kss)
        self.convblock1 = tsai.models.layers.ConvBlock(c_in, layers[0], kss[0])
        self.convblock2 = tsai.models.layers.ConvBlock(layers[0], layers[1], kss[1])
        self.convblock3 = tsai.models.layers.ConvBlock(layers[1], layers[2], kss[2])
        self.gap = tsai.models.layers.GAP1d(1)
        self.fc = nn.Linear(layers[-1], c_out)

    def forward(self, x, depth=None, pass_part=None):
        if pass_part=='first':
            if depth==0:
                return x
            x = self.convblock1(x)
            if depth==1:
                return x
            x = self.convblock2(x)
            if depth==2:
                return x
            x = self.convblock3(x)
            if depth==3:
                return x
            x = self.gap(x)
            if depth ==4:
                return x
            x = self.fc(x)
            return x
        elif pass_part=='second':
            if depth <= 0:
                x = self.convblock1(x)
            if depth <=1:
                x = self.convblock2(x)
            if depth <=2:
                x = self.convblock3(x)
            if depth <=3:
                x = self.gap(x)
            if depth <=4:
                x = self.fc(x)
                return x
        elif pass_part=='latent_space':
            x = self.convblock1(x)
            x = self.convblock2(x)
            x = self.convblock3(x)
            x = self.gap(x)
            return x
        else:
            #print(f'Input {x.shape=}')
            x = self.convblock1(x)
            #print(f'Block1 {x.shape=}')
            x = self.convblock2(x)
            #print(f'Block2 {x.shape=}')
            x = self.convblock3(x)
            #print(f'Block3 {x.shape=}')
            x = self.gap(x)
            #print(f'Gap {x.shape=}')
            x = self.fc(x)
            #print(f'Linear {x.shape=}')
            return x

### taken from: https://github.com/timeseriesAI/tsai/blob/main/tsai/models/ResCNN.py
def ResCNN_TS(num_channels=4, num_classes=2):
    return ResCNN(c_in=num_channels, c_out=num_classes)

class _ResCNNBlock(nn.Module):
    def __init__(self, ni, nf, kss=[7, 5, 3], coord=False, separable=False, zero_norm=False):
        super().__init__()
        self.convblock1 = tsai.models.layers.ConvBlock(ni, nf, kss[0], coord=coord, separable=separable)
        self.convblock2 = tsai.models.layers.ConvBlock(nf, nf, kss[1], coord=coord, separable=separable)
        self.convblock3 = tsai.models.layers.ConvBlock(nf, nf, kss[2], act=None, coord=coord, separable=separable, zero_norm=zero_norm)

        # expand channels for the sum if necessary
        self.shortcut = tsai.models.layers.ConvBN(ni, nf, 1, coord=coord)
        self.add = tsai.models.layers.Add()
        self.act = nn.ReLU()

    def forward(self, x):
        res = x
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.add(x, self.shortcut(res))
        x = self.act(x)
        return x

class ResCNN(nn.Module):
    def __init__(self, c_in, c_out, coord=False, separable=False, zero_norm=False):
        super().__init__()
        nf = 64
        self.block1 = _ResCNNBlock(c_in, nf, kss=[7, 5, 3], coord=coord, separable=separable, zero_norm=zero_norm)
        self.block2 = tsai.models.layers.ConvBlock(nf, nf * 2, 3, coord=coord, separable=separable, act=nn.LeakyReLU, act_kwargs={'negative_slope':.2})
        self.block3 = tsai.models.layers.ConvBlock(nf * 2, nf * 4, 3, coord=coord, separable=separable, act=nn.PReLU)
        self.block4 = tsai.models.layers.ConvBlock(nf * 4, nf * 2, 3, coord=coord, separable=separable, act=nn.ELU, act_kwargs={'alpha':.3})
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.squeeze = tsai.models.layers.Squeeze(-1)
        self.lin = nn.Linear(nf * 2, c_out)

    def forward(self, x, depth=None, pass_part=None):
        if pass_part=='first':
            if depth==0:
                return x
            x = self.block1(x)
            if depth==1:
                return x
            x = self.block2(x)
            if depth==2:
                return x
            x = self.block3(x)
            if depth==3:
                return x
            x = self.block4(x)
            if depth==4:
                return x
            x = self.gap(x)
            x = self.squeeze(x)
            if depth==5:
                return x
            x = self.lin(x)
            return x
        elif pass_part=='second':
            if depth <= 0:
                x = self.block1(x)
            if depth <= 1:
                x = self.block2(x)
            if depth <= 2:
                x = self.block3(x)
            if depth <= 3:
                x = self.block4(x)
            if depth <= 4:
                x = self.gap(x)
                x = self.squeeze(x)
            if depth <= 5:
                x = self.lin(x)
                return x
        else:
            #print(f'Input {x.shape=}')
            x = self.block1(x)
            #print(f'Block1 {x.shape=}')
            x = self.block2(x)
            #print(f'Block2 {x.shape=}')
            x = self.block3(x)
            #print(f'Block3 {x.shape=}')
            x = self.block4(x)
            #print(f'Block4 {x.shape=}')
            x = self.gap(x)
            #print(f'Gap {x.shape=}')
            x = self.squeeze(x)
            #print(f'Squeeze {x.shape=}')
            x = self.lin(x)
            #print(f'Classifier {x.shape=}')
            return x

### taken from: https://github.com/timeseriesAI/tsai/blob/main/tsai/models/ResNet.py
def ResNet_TS(num_channels=4, num_classes=2):
    return ResNet(c_in=num_channels, c_out=num_classes)

class ResBlock(nn.Module):
    def __init__(self, ni, nf, kss=[7, 5, 3]):
        super().__init__()
        self.convblock1 = tsai.models.layers.ConvBlock(ni, nf, kss[0])
        self.convblock2 = tsai.models.layers.ConvBlock(nf, nf, kss[1])
        self.convblock3 = tsai.models.layers.ConvBlock(nf, nf, kss[2], act=None)

        # expand channels for the sum if necessary
        self.shortcut = tsai.models.layers.BN1d(ni) if ni == nf else tsai.models.layers.ConvBlock(ni, nf, 1, act=None)
        self.add = tsai.models.layers.Add()
        self.act = nn.ReLU()

    def forward(self, x):
        res = x
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.add(x, self.shortcut(res))
        x = self.act(x)
        return x
    
class ResNet(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        nf = 64
        kss=[7, 5, 3]
        self.resblock1 = ResBlock(c_in, nf, kss=kss)
        self.resblock2 = ResBlock(nf, nf * 2, kss=kss)
        self.resblock3 = ResBlock(nf * 2, nf * 2, kss=kss)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.squeeze = tsai.models.layers.Squeeze(-1)
        self.fc = nn.Linear(nf * 2, c_out)

    def forward(self, x):
        #print(f'Input {x.shape=}')
        x = self.resblock1(x)
        #print(f'Resblock1 {x.shape=}')
        x = self.resblock2(x)
        #print(f'Resblock2 {x.shape=}')
        x = self.resblock3(x)
        #print(f'Resblock3 {x.shape=}')
        x = self.gap(x)
        #print(f'Gap {x.shape=}')
        x = self.squeeze(x)
        #print(f'Squeeze {x.shape=}')
        x = self.fc(x)
        #print(f'Classifier {x.shape=}')
        return x
