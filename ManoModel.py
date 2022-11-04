# Import of the necessary libraries

# Import of the necessary libraries
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
        V=Vm_1*(torch.randn(Vm_1.shape[0],Vm_1.shape[1],Vm_1.shape[2],Vm_1.shape[3])+1)-Vm_2*(torch.randn(Vm_1.shape[0],Vm_1.shape[1],Vm_1.shape[2],Vm_1.shape[3])+1)
        LPTC=V[:,0,:,:]*self.FinalW[0]+V[:,1,:,:]*self.FinalW[1]
        
        return LPTC,Vm_1,Vm_2
        
model = Synaptic_Model()
model = model.double()


###Duplicate and cut labels such that they can be compared with model output
train_out=train_out.tile (1,1,1,70)
dev_out=dev_out.tile(1,1,1,70)
test_out=test_out.tile(1,1,1,70)

train_out=train_out[:,:,29:,:]
dev_out=dev_out[:,:,29:,:]
test_out=test_out[:,:,29:,:]

###Loss Function
mse_loss = nn.MSELoss()


###Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
MS=np.arange(100,1000,100)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=MS, gamma=0.8)
###The Training Loop
def train (epochs, num_batches,batch_size,input_set,output_set,optimizer):
     #scheduler.step()
    train_loss=[]
    for i in range(epochs):
        epoch_train_loss=0
        batch_loss=0
        for j in range (num_batches):
            
            input=input_set[j*batch_size:(j+1)*batch_size,:,:,:]
            input=input+torch.randn(batch_size,input.shape[1],input.shape[2],input.shape[3])
            output=output_set[j*batch_size:(j+1)*batch_size,:,:,:]
            [Velocity,Vm_1,Vm_2]=model(input)
            Velocity=Velocity.unsqueeze(1)
            loss = mse_loss(Velocity,output)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            batch_loss += loss.item()
        epoch_train_loss= batch_loss/num_batches
        train_loss.append(np.array(epoch_train_loss))
        if i %50 == 0:
            torch.save(model, 'modelHighNoise'+str(i)+ '.pth')
        #del Velocity, input, output, Vm_1, Vm_2
        ###Validation
        #val_in=valid_in+torch.randn(valid_in.shape[0],valid_in.shape[1],valid_in.shape[2],valid_in.shape[3])
        #[val_model_out,d1,d2]=model(val_in)
        #val_model_out=val_model_out.unsqueeze(1)
        #epoch_val_loss= mse_loss(val_model_out,valid_out)
        #valid_loss.append(np.array(epoch_val_loss.detach()))
        #del val_model_out, d1, d2
    
    return train_loss#, #valid_loss



#for i in range(epochs):
#    model.train(True)
##    temploss=0
 #   tempt=train(68,133,train_in,train_out,optimizer)
 ##   train_epoch_loss.append(tempt)
  #  model.eval()
  #  [tempv,d1,d2]=model(dev_in)
  #  tempv=tempv.unsqueeze(1)
  #  temploss=mse_loss(tempv,dev_out)
  #  del tempv,d1,d2
  ##  temploss=temploss.detach()
   # valid_epoch_loss.append(temploss)5

train_loss=train(1000,68,133,train_in,train_out,optimizer)   

import matplotlib.pyplot as plt
plt.figure()
plt.plot(train_loss)
#plt.plot(valid_loss)
plt.show()
#print(tempv.shape)
#print(valid_epoch_loss)

#torch.save(model,'model2.pth')


###Evaluation
#test_in=test_in+torch.randn([test_in.shape[0],test_in.shape[1],test_in.shape[2],test_in.shape[3]])
#[TestOutput,Vm_1,Vm_2]=model(test_in)
#print(TestOutput.shape)
#TestOutput=torch.flatten(TestOutput)
#TestOutput=TestOutput.unsqueeze(0)
#test_out=torch.flatten(test_out)
#test_out=test_out.unsqueeze(0)
#CorrAnalVar=torch.cat((test_out,TestOutput),0)
#Corr=torch.corrcoef(CorrAnalVar)
#Explained_Var=Corr**2
#print(Explained_Var)
#plt.figure()
##plt.plot(TestOutput[0,:1000].detach(),test_out[0,:1000])
#plt.show()
#print(Vm_1.shape)

###Calculate explained variance and correlation coefficient
##from scipy.stats import pearsonr
#import matplotlib.pyplot as plt
##corr= torch.corrcoef(TestOutput)
#print('Pearsons correlation: %.3f' % corr)
#Explained_Var=corr**2
#print(epoch_loss)
#plt.figure()
#plt.plot(epoch_loss)
#plt.show()

        

       




    

###Evaluation
test_in=test_in+torch.randn([test_in.shape[0],test_in.shape[1],test_in.shape[2],test_in.shape[3]])
[TestOutput,Vm_1,Vm_2]=model(test_in)
print(TestOutput.shape)
TestOutput=torch.flatten(TestOutput)
TestOutput=TestOutput.unsqueeze(0)
test_out=torch.flatten(test_out)
test_out=test_out.unsqueeze(0)
CorrAnalVar=torch.cat((test_out,TestOutput),0)
Corr=torch.corrcoef(CorrAnalVar)
Explained_Var=Corr**2
print(Explained_Var)
#plt.figure()
##plt.plot(TestOutput[0,:1000].detach(),test_out[0,:1000])
#plt.show()
#print(Vm_1.shape)

###Calculate explained variance and correlation coefficient
##from scipy.stats import pearsonr
#import matplotlib.pyplot as plt
##corr= torch.corrcoef(TestOutput)
#print('Pearsons correlation: %.3f' % corr)
#Explained_Var=corr**2
#print(epoch_loss)
#plt.figure()
#plt.plot(epoch_loss)
#plt.show()

        

       




    