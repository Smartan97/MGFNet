import torch
from torch import nn
from einops import rearrange
import torch.nn.functional as F
import math
class Mix(nn.Module):
    def __init__(self, m=-0.80):
        super(Mix, self).__init__()
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        self.mix_block = nn.Sigmoid()

    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)
        out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))
        return out

class Attention(nn.Module):
    def __init__(self,channel,b=1, gamma=2):
        super(Attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)#全局平均池化
        #一维卷积
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)
        self.fc = nn.Conv2d(channel, channel, 1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.mix = Mix()


    def forward(self, input):
        x = self.avg_pool(input)
        x1 = self.conv1(x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2)
        x2 = self.fc(x).squeeze(-1).transpose(-1, -2)
        
        out1 = torch.sum(torch.matmul(x1,x2),dim=1).unsqueeze(-1).unsqueeze(-1)
        out1 = self.sigmoid(out1)
        out2 = torch.sum(torch.matmul(x2.transpose(-1, -2),x1.transpose(-1, -2)),dim=1).unsqueeze(-1).unsqueeze(-1)
        out2 = self.sigmoid(out2)
        
        out = self.mix(out1,out2)
        out = self.conv1(out.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out = self.sigmoid(out)

        return input*out

class DownsamplerBlock(nn.Module):
    def __init__(self,nchan):
        super(DownsamplerBlock,self).__init__()
        self.conv = nn.Conv2d(nchan, nchan, kernel_size=(1,5), stride=3, padding=(0,1))
        self.pool = nn.MaxPool2d((1,3), stride=(1,3))
        self.conv1 = nn.Conv2d(nchan*2, nchan, kernel_size=(1,1))

    def forward(self, x):
        output = torch.cat([self.conv(x), self.pool(x)], 1)
        output = self.conv1(output)
        return output

class ClsHead(nn.Module):
    def __init__(self, linear_in, cls):
        super(ClsHead,self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(linear_in, cls, kernel_size=(1,1)),
                                    nn.BatchNorm2d(cls),
                                    nn.ReLU())
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cls = nn.Linear(cls, cls)
    def forward(self, x):
        x = self.conv1(x)
        x = self.avg_pool(x).squeeze()
        out = self.cls(x)
        
        return out

class Feature_Extract(nn.Module):
    def __init__(self, n_chans):
        super(Feature_Extract, self).__init__()

        self.temp_conv1 = DownsamplerBlock(n_chans)
        self.temp_conv2 = DownsamplerBlock(n_chans)
        self.temp_conv3 = DownsamplerBlock(n_chans)
        self.temp_conv4 = DownsamplerBlock(n_chans)
        
        self.tp    = nn.Sequential(
            nn.Conv2d(n_chans, 32, kernel_size=(1,5), padding=(0,2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((1,3), stride=(1,3)),
            nn.Conv2d(32, 64, kernel_size=(1,5), padding=(0,2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((1,3), stride=(1,3)),
            nn.Conv2d(64, 128, kernel_size=(1,5), padding=(0,2)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            Attention(channel=128),
            nn.MaxPool2d((1,3), stride=(1,3)),
            nn.Conv2d(128, 32, kernel_size=(1,5), padding=(0,2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((1,3), stride=(1,3)),
            nn.Dropout(p=0.3)
            )
        
        self.chpool0    = nn.Sequential(
            nn.Conv2d(n_chans, 32, kernel_size=(1,5), padding=(0,2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((1,3), stride=(1,3)),
            nn.Conv2d(32, 64, kernel_size=(1,5), padding=(0,2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((1,3), stride=(1,3)),
            nn.Conv2d(64, 128, kernel_size=(1,5), padding=(0,2)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            Attention(channel=128),
            nn.MaxPool2d((1,3), stride=(1,3)),
            nn.Conv2d(128, 32, kernel_size=(1,5), padding=(0,2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(p=0.3)
            )

        self.chpool1    = nn.Sequential(
            nn.Conv2d(n_chans, 32, kernel_size=(1,5), padding=(0,2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((1,3), stride=(1,3)),
            nn.Conv2d(32, 64, kernel_size=(1,5), padding=(0,2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((1,3), stride=(1,3)),
            nn.Conv2d(64, 128, kernel_size=(1,5), padding=(0,2)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            Attention(channel=128),
            nn.Conv2d(128, 32, kernel_size=(1,5), padding=(0,2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(p=0.3)
            )

        self.chpool2    = nn.Sequential(
            nn.Conv2d(n_chans, 32, kernel_size=(1,5), padding=(0,2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((1,3), stride=(1,3)),
            nn.Conv2d(32, 64, kernel_size=(1,5), padding=(0,2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(1,5), padding=(0,2)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            Attention(channel=128),
            nn.Conv2d(128, 32, kernel_size=(1,5), padding=(0,2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(p=0.3))

        self.chpool3    = nn.Sequential(
            nn.Conv2d(n_chans, 32, kernel_size=(1,5), padding=(0,2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 128, kernel_size=(1,5), padding=(0,2)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            Attention(channel=128),
            nn.Conv2d(128, 32, kernel_size=(1,5), padding=(0,2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(p=0.3))
        self.head1 = ClsHead(32,5)
        self.head2 = ClsHead(32,5)
        self.head3 = ClsHead(32,5)
        self.head4 = ClsHead(32,5)
        self.head0 = ClsHead(32,5)
        
        self.gcn = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.bn = torch.nn.BatchNorm1d(32)
        
        self.gcn1 = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.bn1 = nn.BatchNorm1d(32)
        self.adj = nn.Parameter(torch.ones(5, 5))
        
    def forward(self, x):
        temp_w0  = self.temp_conv1(x)               
        temp_w1 = self.temp_conv2(temp_w0)         
        temp_w2 = self.temp_conv3(temp_w1)      
        temp_w3 = self.temp_conv4(temp_w2)       

        t0      = self.tp(x)
        t01     = self.head0(t0)
        t0      = t0.mean(dim=(-1))
        
        w0      = self.chpool0(temp_w0)
        w01     = self.head1(w0)
        w0      = w0.mean(dim=(-1))
        
        w1      = self.chpool1(temp_w1)
        w11     = self.head2(w1)
        w1      = w1.mean(dim=(-1))
        
        w2      = self.chpool2(temp_w2)
        w21     = self.head3(w2)
        w2      = w2.mean(dim=(-1))
        
        w3      = self.chpool3(temp_w3)
        w31     = self.head4(w3)
        w3      = w3.mean(dim=(-1))
        
        node_feature = torch.stack([t0, w0, w1, w2, w3], dim=1).squeeze(-1).permute(0,2,1)
        h = torch.matmul(node_feature, self.adj).permute(0,2,1)
        concat_vector = self.gcn(h)
        bz = h.shape[0]
        concat_vector = concat_vector.reshape((bz*5, -1))                 
        concat_vector = self.bn(concat_vector)
        concat_vector = concat_vector.reshape((bz, 5, -1))   
        

        out = concat_vector + node_feature.permute(0,2,1)
        
        return out
       
        
class MGFNet(nn.Module):
    def __init__(self, n_chans=1,n_classes=5):
        super(MGFNet, self).__init__()
        self.feature = Feature_Extract(n_chans)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*5,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.Sigmoid(),
            nn.Linear(32,n_classes))

    def forward(self, x):
        x = x.unsqueeze(-2)
        x = x[:,0:1,:]
        concat_vector = self.feature(x)
        classes       = self.classifier(concat_vector) 
        return classes


if __name__ == "__main__":
    # 测试网络
    model = MGFNet()
    test_input = torch.randn(32, 1, 3000)  # batch=2
    output = model(test_input)
    print("输出形状:", output.shape)  # (2, 2) 期望
    
    from thop import profile
    inputs = torch.randn(1, 2, 3000)
    flops, params = profile(model, (inputs,))
    print("FLOPs=", str(flops/1e9) + '{}'.format("G"))
    print("params=", str(params/1e6) + '{}'.format("M")) 
