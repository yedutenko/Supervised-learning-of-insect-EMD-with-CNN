import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import h5py as h
import scipy.io as sio
import os
import torch.nn.functional as F
# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
data_folder = r'/home/myedut/Downloads/Scripts for toy tasks/data'
os.chdir(data_folder)

####Load_Data
def load_data_rr(path):
    mat_contents = h.File(path, 'r')

    train_in = mat_contents['train_in'][:]
    train_out = mat_contents['train_out'][:]
    dev_in = mat_contents['dev_in'][:]
    dev_out = mat_contents['dev_out'][:]
    test_in = mat_contents['test_in'][:]
    test_out = mat_contents['test_out'][:]

    sample_freq = mat_contents['sampleFreq'][:]
    phase_step = mat_contents['phaseStep'][:]

    train_in = np.expand_dims(train_in, axis=3)
    dev_in = np.expand_dims(dev_in, axis=3)
    test_in = np.expand_dims(test_in, axis=3)

    train_out = np.expand_dims(train_out, axis=2)
    train_out = np.expand_dims(train_out, axis=3)
    dev_out = np.expand_dims(dev_out, axis=2)
    dev_out = np.expand_dims(dev_out, axis=3)
    test_out = np.expand_dims(test_out, axis=2)
    test_out = np.expand_dims(test_out, axis=3)

    mat_contents.close()

    return train_in, train_out, dev_in, dev_out, test_in, test_out, sample_freq, phase_step




image_type_idx=0
image_types = ['nat', 'sine']
data_set_name=['DataMano.mat']
path = data_folder + '/' + data_set_name[0]
[train_in, train_out, dev_in,dev_out, test_in, test_out,sample_freq,phase_step]=load_data_rr(path)



train_in=torch.from_numpy(train_in)
train_out=torch.from_numpy(train_out)
dev_in=torch.from_numpy(dev_in)
dev_out=torch.from_numpy(dev_out)
test_in=torch.from_numpy(test_in)
test_out=torch.from_numpy(test_out)

train_in=train_in.permute(0,3,1,2)
dev_in=dev_in.permute(0,3,1,2)
test_in=test_in.permute(0,3,1,2)
train_out=train_out.permute(0,3,1,2)
dev_out=dev_out.permute(0,3,1,2)
test_out=test_out.permute(0,3,1,2)



####Define Neural Network Model

class Synaptic_Model(torch.nn.Module):
    def __init__(self):
        super(Synaptic_Model,self).__init__()
        self.conv1=nn.Conv2d(1,2,(30,1))
        self.conv2=nn.Conv2d(1,2,(30,1))
        self.conv3=nn.Conv2d(1,2,(30,1))
        self.numerator1=nn.Parameter(torch.rand(2,1,1))
        self.numerator2=nn.Parameter(torch.rand(2,1,1))
        self.numerator3=nn.Parameter(torch.rand(2,1,1))
        self.BiasV=nn.Parameter(torch.rand(2,1,1))
        self.FinalW=nn.Parameter(torch.rand(2,1,1))

    def forward(self,x):
        ###Split input into 3 spatially displaced parts
        s1=x[:,:,:,:-2]
        s2=x[:,:,:,1:-1]
        s3=x[:,:,:,2:]
        ###Separately convolve each of the inputs with corresponding filter
        g1_1=F.relu(self.conv1(s1))
        g2=F.relu(self.conv2(s2))
        g1_3=F.relu(self.conv3(s3))
        ###Now get the same for the EMDs with opposite spatial orientation
        g2_1=F.relu(self.conv1(s3))
        g2_3=F.relu(self.conv3(s1))
        ###Now, unite g-filters together and split into (presumably) ON and OFF subunits
        Combined_g_1=g1_1*self.numerator1 + g2*self.numerator2 +g1_3*self.numerator3
        Combined_g_2=g2_1*self.numerator1 + g2*self.numerator2 +g2_3*self.numerator3
        Vm_1=F.relu(Combined_g_1/(1+g1_1+g2+g1_3) + self.BiasV)
        Vm_2=F.relu(Combined_g_2/(1+g2_1+g2+g2_3) + self.BiasV)
        V=Vm_1*(0.1*torch.randn(Vm_1.shape[0],Vm_1.shape[1],Vm_1.shape[2],Vm_1.shape[3])+1)-Vm_2*(0.1*torch.randn(Vm_1.shape[0],Vm_1.shape[1],Vm_1.shape[2],Vm_1.shape[3])+1)
        LPTC=V[:,0,:,:]*self.FinalW[0]+V[:,1,:,:]*self.FinalW[1]
        
        return LPTC,Vm_1,Vm_2
train_out=train_out.tile (1,1,1,70)
dev_out=dev_out.tile(1,1,1,70)
test_out=test_out.tile(1,1,1,70)

train_out=train_out[:,:,29:,:]
dev_out=dev_out[:,:,29:,:]
test_out=test_out[:,:,29:,:]
        
model = Synaptic_Model()
model = model.double()
model=torch.load('modelLowNoise950.pth')
model.train(False)
model.eval()

###Evaluation
test_in=test_in+0.1*torch.randn([test_in.shape[0],test_in.shape[1],test_in.shape[2],test_in.shape[3]])
[TestOutput,Vm_1,Vm_2]=model(test_in)
print(TestOutput.shape)
TestOutputGlobal=torch.clone(TestOutput)
test_outGlobal=torch.clone(test_out)

TestOutput=torch.flatten(TestOutput)
TestOutput=TestOutput.unsqueeze(0)
test_out=torch.flatten(test_out)
test_out=test_out.unsqueeze(0)
CorrAnalVar=torch.cat((test_out,TestOutput),0)
Corr=torch.corrcoef(CorrAnalVar)
Explained_Var=Corr**2
print(Explained_Var)

TestOutputGlobal=torch.mean(TestOutputGlobal,-1)
test_outGlobal=torch.mean(test_outGlobal,-1)
TestOutputGlobal=torch.flatten(TestOutputGlobal)
test_outGlobal=torch.flatten(test_outGlobal)
TestOutputGlobal=TestOutputGlobal.unsqueeze(0)
test_outGlobal=test_outGlobal.unsqueeze(0)
CorrAnalVarG=torch.cat((test_outGlobal,TestOutputGlobal),0)
CorrG=torch.corrcoef(CorrAnalVarG)
print(CorrG)

###Filters
import matplotlib.pyplot as plt
import seaborn as sns
T4F1=torch.flatten(model.conv1.weight[0,:,:,:].detach())
T4F2=torch.flatten(model.conv2.weight[0,:,:,:].detach())
T4F3=torch.flatten(model.conv3.weight[0,:,:,:].detach())

T4F1=T4F1.unsqueeze(1)
T4F2=T4F2.unsqueeze(1)
T4F3=T4F3.unsqueeze(1)

T4R=torch.cat((T4F1,T4F2,T4F3),1)
T4R=torch.flip(T4R,dims=(0,))
plt.figure()
sns.heatmap(T4R)
plt.show()


T5F1=torch.flatten(model.conv1.weight[1,:,:,:].detach())
T5F2=torch.flatten(model.conv2.weight[1,:,:,:].detach())
T5F3=torch.flatten(model.conv3.weight[1,:,:,:].detach())

T5F1=T5F1.unsqueeze(1)
T5F2=T5F2.unsqueeze(1)
T5F3=T5F3.unsqueeze(1)

T5R=torch.cat((T5F1,T5F2,T5F3),1)
T5R=torch.flip(T5R,dims=(0,))
plt.figure()
sns.heatmap(T5R)
plt.show()
del train_in, train_out, dev_in, dev_out
Sample1=np.random.randint(200000,size=1000)

TestOutput=TestOutput.squeeze()
test_out=test_out.squeeze()
TestOutputGlobal=TestOutputGlobal.squeeze()
test_outGlobal=test_outGlobal.squeeze()

TestOutput=TestOutput[Sample1]

test_out=test_out[Sample1]
TestOutputGlobal=TestOutputGlobal[Sample1]
test_outGlobal=test_outGlobal[Sample1]
plt.figure()
plt.text(-50,200,'R2=0.017',fontsize=14)
plt.scatter(TestOutput.detach(),test_out.detach())
plt.show()

plt.figure()
plt.text(-50,50,'R2=0.51',fontsize=14)
plt.scatter(TestOutputGlobal.detach(),test_outGlobal.detach(),color='orange')
plt.show()

plt.figure()
sns.heatmap(Corr.detach())

#del  train_in, train_out

RUnit1=torch.flatten(Vm_1[:,0,:,:].detach())
RUnit2=torch.flatten(Vm_1[:,1,:,:].detach())
LUnit1=torch.flatten(Vm_2[:,0,:,:].detach())
LUnit2=torch.flatten(Vm_2[:,1,:,:].detach())

RUnit1=RUnit1.unsqueeze(0)
RUnit2=RUnit2.unsqueeze(0)
LUnit1=LUnit1.unsqueeze(0)
LUnit2=LUnit2.unsqueeze(0)
DeCorrAnalVar=torch.cat((RUnit1,RUnit2,LUnit1,LUnit2),0)
Decorr=torch.corrcoef(DeCorrAnalVar)
print(Decorr)
plt.figure()
Names=['RUnit1','RUnit2','LUnit1','LUnit2']
sns.heatmap(Decorr,xticklabels=Names, yticklabels=Names)
plt.show()



###Duplicate and cut labels such that they can be compared with model output
###
#1.Model vs. RealData + r^2
# 2w.22Veryfy against global
#2.Heatmap of the filters
#3. Decorrelation
#4. High Noise
#5. High Noise Loss
#6. Sinewaves
#3.model
#import matplotlib.pyplot as plt
#plt.figure()
#plt.plot(torch.flatten(model.conv1.weight[1,:,:,:].detach()))
#plt.plot(torch.flatten(model.conv2.weight[1,:,:,:].detach()))
#plt.plot(torch.flatten(model.conv3.weight[1,:,:,:].detach()))
#plt.show()
