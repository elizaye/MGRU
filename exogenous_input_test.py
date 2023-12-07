# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 23:57:19 2023

@author: eliza
"""

# imports

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.optim as optim
from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler # use this for data scaling (or other scaler?)
import math
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error
#from utils import create_dataset, to_torch
#import time_series_prediction.models as models
import os
os.chdir("/u/lexuanye/STAT556/project/mRNN-mLSTM")
import sys
sys.path.append("/u/lexuanye/STAT556/project/mRNN-mLSTM/time_series_prediction/")
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.autograd import Variable
from utils import *
import models

import statsmodels as sm


# In[8]:


# set model device
#device = torch.device("cuda")
#print(device)


# In[9]:


# the LSTM and MGRU needs revision

class MGRUFixDCell(nn.Module): # test if this version works
    
    """memory augmented GRUCell with fixed D"""
    def __init__(self,input_size,hidden_size,output_size,k):
        super(MGRUFixDCell,self).__init__()
        self.hidden_size = hidden_size; self.k = k; self.output_size = output_size
        self.r_gate = nn.Linear(input_size+hidden_size,hidden_size) # reset gate
        self.z_gate = nn.Linear(input_size+hidden_size,hidden_size) # update gate
        self.h_tilde_gate = nn.Linear(input_size+hidden_size,hidden_size) # candidate hidden state
        self.output = nn.Linear(hidden_size,output_size) # output layer
        # activation functions
        self.sigmoid = nn.Sigmoid(); self.tanh = nn.Tanh()

    def forward(self,sample,hidden,hidden_tensor,weights): # no cell state for GRU
        
        # all cells here doesn't consider bias term?
        batch_size = sample.size(0)
        if hidden is None:
            hidden = torch.zeros(batch_size,self.hidden_size,dtype=sample.dtype,device=sample.device)
        if hidden_tensor is None: # initiate for storing k-step ahead hidden states
            hidden_tensor = torch.zeros(self.k,batch_size,self.hidden_size,dtype=sample.dtype,device=sample.device)
        combined = torch.cat((sample,hidden),1) # model input & hidden state
        first = torch.einsum('ijk,ik->ijk',[-hidden_tensor,weights]).sum(dim=0) # calculate w(j)c(t)
        r_gate = self.r_gate(combined); r_gate = self.sigmoid(r_gate) # reset gate
        z_gate = self.z_gate(combined); z_gate = self.sigmoid(z_gate) # update gate
        combined2 = torch.cat((sample,torch.mul(r_gate,hidden)),1)
        h_tilde = self.h_tilde_gate(combined2); h_tilde = self.tanh(h_tilde) # candidate hidden state
        
        second = torch.mul(h_tilde,(1-z_gate)) # check if this is correct or not
        hidden = torch.add(first,second) # H = Z*H + (1-Z)*H_tilde 
        h_c = torch.cat([hidden_tensor,hidden.view([-1,hidden.size(0),hidden.size(1)])],0) # not sure if the size here is correct or not
        h_c_1 = h_c[1:,:] # remove the k+1 step ahenad hidden state
        output = self.output(hidden) # linear layer
        return output,hidden,h_c_1

    def init_hidden(self):
        return Variable(torch.zeros(1,self.hidden_size))
    
class MGRUCell(nn.Module):
    """memory augmented GRUcell with dynamic D"""
    def __init__(self,input_size,hidden_size,output_size,k):
        super(MGRUCell,self).__init__()
        self.hidden_size = hidden_size; self.k = k; self.output_size = output_size
        self.d_gate = nn.Linear(input_size+hidden_size*2,hidden_size) # for calculating dynamic d
        self.r_gate = nn.Linear(input_size+hidden_size,hidden_size) # reset gate
        self.z_gate = nn.Linear(input_size+hidden_size,hidden_size) # update gate
        self.h_tilde_gate = nn.Linear(input_size+hidden_size,hidden_size) # candidate hidden state
        self.output = nn.Linear(hidden_size,output_size) # output layer
        # activation functions
        self.sigmoid = nn.Sigmoid(); self.tanh = nn.Tanh()

    def get_ws(self,d_values):
        weights = [1] * (self.k+1)
        for i in range(0,self.k):
            weights[self.k-i-1] = weights[self.k-i]*(i-d_values)/(i+1)
        return torch.cat(weights[0:self.k])

    def filter_d(self,hidden_tensor,d_values):
        weights = torch.ones(self.k,d_values.size(0),d_values.size(1),dtype=d_values.dtype,device=d_values.device)
        hidden_size = weights.shape[2]; batch_size = weights.shape[1]
        for batch in range(batch_size):
            for hidden in range(hidden_size):
                weights[:,batch,hidden] = self.get_ws(d_values[batch,hidden].view([1]))
        outputs = hidden_tensor.mul(weights).sum(dim=0)
        return outputs

    def forward(self,sample,hidden,hidden_tensor,d_0):
        batch_size = sample.size(0)
        if hidden is None:
            hidden = torch.zeros(batch_size,self.hidden_size,dtype=sample.dtype,device=sample.device)
        if hidden_tensor is None: # initiate for storing k-step ahead hidden states
            hidden_tensor = torch.zeros(self.k,batch_size,self.hidden_size,dtype=sample.dtype,device=sample.device)
        if d_0 is None: # dynamic d; check where this should be put
            d_0 = torch.zeros(batch_size,self.hidden_size,dtype=sample.dtype,device=sample.device)
        
        # replace the previous reset gate with the d calculation gate
        combined = torch.cat((sample,hidden),1) # model input & hidden state
        combined_d = torch.cat((sample,hidden,d_0),1)
        d_values = self.d_gate(combined_d); d_values = self.sigmoid(d_values)*0.5 # calculate dynamic d
        first = -self.filter_d(hidden_tensor,d_values) # check if the shape here is correct or not
        r_gate = self.r_gate(combined); r_gate = self.sigmoid(r_gate) # reset gate
        z_gate = self.z_gate(combined); z_gate = self.sigmoid(z_gate) # update gate
        combined2 = torch.cat((sample,torch.mul(r_gate,hidden)),1) # check if r_gate is still needed here
        h_tilde = self.h_tilde_gate(combined2); h_tilde = self.tanh(h_tilde) # candidate hidden state

        second = torch.mul(h_tilde,(1-z_gate)) # check if this is correct or not
        hidden = torch.add(first,second) # H = Z*H + (1-Z)*H_tilde 
        h_c = torch.cat([hidden_tensor,hidden.view([-1,hidden.size(0),hidden.size(1)])],0) # not sure if the size here is correct or not
        h_c_1 = h_c[1:,:] # remove the k+1 step ahenad hidden state
        output = self.output(hidden) # linear layer
        return output,hidden,h_c_1,d_values
    
    def init_hidden(self):
        return Variable(torch.zeros(1,self.hidden_size))


# In[10]:

class LSTM(nn.Module):
    """LSTM model for time series prediction"""
    def __init__(self,input_size,hidden_size,output_size):
        super(LSTM,self).__init__()
        self.in_size = input_size
        self.h_size = hidden_size
        self.out_size = output_size
        self.lstm_cell = nn.LSTMCell(input_size,hidden_size)
        self.output = nn.Linear(hidden_size,output_size)

    def forward(self,inputs,hidden_state=None):
        #print(inputs.device)
        time_steps = inputs.shape[0]; batch_size = inputs.shape[1]
        outputs = torch.Tensor(time_steps,batch_size,self.out_size)
        if hidden_state is None:
            h_0 = torch.zeros(batch_size,self.h_size,device=inputs.device)
            c_0 = torch.zeros(batch_size,self.h_size,device=inputs.device)
            #print(h_0.device,c_0.device)
            hidden_state = (h_0, c_0)
        else:
            h_0 = hidden_state[0]
            c_0 = hidden_state[1]
            #print(h_0.device,c_0.device)
        for times in range(time_steps):
            #print(h_0.device,c_0.device)
            h_0, c_0 = self.lstm_cell(inputs[times,:],(h_0, c_0))
            outputs[times,:] = self.output(h_0)
        return outputs,(h_0,c_0)
    
class GRU(nn.Module):
    """GRU model for time series prediction"""
    def __init__(self,input_size,hidden_size,output_size):
        super(GRU,self).__init__()
        self.input_size = input_size; self.hidden_size = hidden_size; self.output_size = output_size
        self.gru = nn.GRU(input_size=self.input_size,hidden_size=self.hidden_size,num_layers=1,bias=False)# no using bias here
        self.hidden2output = nn.Linear(self.hidden_size,output_size)

    def forward(self,input_y,hidden_state=None):
        samples = input_y
        gru_out,last_gru_hidden = self.gru(samples,hidden_state) # check when hidden_stat is missing will this work or not
        output = self.hidden2output(gru_out.view(-1,self.hidden_size))
        return output.view(samples.shape[0],samples.shape[1],self.output_size),last_gru_hidden


class MGRUFixD(nn.Module):
    
    """mGRU with fixed d for time series prediction"""
    def __init__(self,input_size,hidden_size,k,output_size):
        super(MGRUFixD,self).__init__()
        self.input_size = input_size; self.hidden_size = hidden_size
        self.k = k; self.d_values = Parameter(torch.Tensor(torch.zeros(1,hidden_size)),requires_grad=True)
        self.output_size = output_size
        self.mgru_cell = MGRUFixDCell(self.input_size,self.hidden_size,self.output_size,self.k)
        self.sigmoid = nn.Sigmoid()

    def get_w(self,d_values):
        k = self.k
        weights = [1.]*(k+1)
        for i in range(k):
            weights[k-i-1] = weights[k-i]*(i-d_values)/(i+1)
        return torch.cat(weights[0:k])

    def get_wd(self,d_value):
        weights = torch.ones(self.k,1,d_value.size(1),dtype=d_value.dtype,device=d_value.device)
        batch_size = weights.shape[1]; hidden_size = weights.shape[2]
        for sample in range(batch_size):
            for hidden in range(hidden_size):
                weights[:,sample,hidden] = self.get_w(d_value[0,hidden].view([1]))
        return weights.squeeze(1)

    def forward(self,inputs,hidden_states=None):
        if hidden_states is None:
            hidden = None; h_c = None
        else:
            hidden = hidden_states[0]; h_c = hidden_states[1]
        time_steps = inputs.shape[0]; batch_size = inputs.shape[1]
        outputs = torch.zeros(time_steps,batch_size,self.output_size,dtype=inputs.dtype,device=inputs.device)
        self.d_values_sigmoid = 0.5*F.sigmoid(self.d_values); weights_d = self.get_wd(self.d_values_sigmoid)
        for i in range(time_steps):
            outputs[i,:],hidden,h_c = self.mgru_cell(inputs[i,:],hidden,h_c,weights_d)
        return outputs,(hidden,h_c)
    
class MGRU(nn.Module):
    """mGRU with dynamic d for time series prediction"""
    def __init__(self,input_size,hidden_size,k,output_size):
        super(MGRU,self).__init__()
        self.input_size = input_size; self.hidden_size = hidden_size; self.k = k
        self.output_size = output_size; self.mgru_cell = MGRUCell(self.input_size,self.hidden_size,self.output_size,self.k)

    def forward(self,inputs,hidden_state=None):
        if hidden_state is None:
            hidden = None; h_c = None; d_values = None
        else:
            hidden = hidden_state[0]; h_c = hidden_state[1]; d_values = hidden_state[2]
        time_steps = inputs.shape[0]; batch_size = inputs.shape[1]
        outputs = torch.zeros(time_steps,batch_size,self.output_size,dtype=inputs.dtype,device=inputs.device)
        for i in range(time_steps):
            outputs[i,:],hidden,h_c,d_values = self.mgru_cell(inputs[i,:],hidden,h_c,d_values)
        return outputs,(hidden,h_c,d_values)


# ### 1. exogenous input test

# In[11]:


def create_dataset_multivar(x,y):
    """creat multivariable train dataset for time series prediction"""
    dataYp,dataYc = [],[]
    for i in range(len(y)-1):
        dataYp.append(x[i,:])
        dataYc.append(y[i+1,:])
    return np.array(dataYp),np.array(dataYc)


# In[13]:


# use the traffic dataset instead

df_data = pd.read_csv(os.path.join("data","time_series_prediction",'traffic.csv'))
df_data

for var in ["temp"]:
    
    df_data[var] = list(df_data[var])[1:]+[np.nan]
    
df_data = df_data.dropna()

# In[15]:


# split train/val/test dataset

#train_size = 1300; validate_size = 365 # currently use this splitting
train_size = 1200; validate_size = 200 # traffic
batch_size = 1; epochs = 1000 # currently use this value
patience = 100
rep_times = 10; K = 25; lr = 0.01; hidden_size = 1
input_size = 2; output_size = 1

dataset = "traffic"

for i in range(rep_times): # each time the training results needs to be stored
    
    seed = i; print('seed ----------------------------------', seed)
    #x = np.array(df_data[["pm2.5","DEWP","TEMP","PRES","Iws","Is","Ir"]]); y = np.array(df_data[["pm2.5"]])
    x = np.array(df_data[["x","temp"]]); y = np.array(df_data[["x"]])
    #x = x.reshape(-1,1); y = y.reshape(-1,7) # for time series the input size & output size are both 1
    # normalize the data
    scaler = MinMaxScaler(feature_range=(0,1))
    x = scaler.fit_transform(x); y = scaler.fit_transform(y)
    # use this function to prepare the data for modeling
    data_x,data_y = create_dataset_multivar(x,y)
    # continue
    #print(data_x.shape,data_y.shape)
    
    # train & test set splitting
    train_x,train_y = data_x[0:train_size],data_y[0:train_size]
    validate_x,validate_y = data_x[train_size:train_size+validate_size],data_y[train_size:train_size+validate_size]
    test_x,test_y = data_x[train_size+validate_size:len(data_y)],data_y[train_size+validate_size:len(data_y)]

    # reshape input to be [time steps,samples,features]
    # what is batchsize here? currently use 1
    train_x = np.reshape(train_x,(train_x.shape[0],batch_size,input_size))
    validate_x = np.reshape(validate_x,(validate_x.shape[0],batch_size,input_size))
    test_x = np.reshape(test_x,(test_x.shape[0],batch_size,input_size))
    train_y = np.reshape(train_y,(train_y.shape[0],batch_size,output_size))
    validate_y = np.reshape(validate_y,(validate_y.shape[0],batch_size,output_size))
    test_y = np.reshape(test_y,(test_y.shape[0],batch_size,output_size))
    torch.manual_seed(seed)
    
    # initialize model
    for algorithm in ['RNN','mRNN_fixD','mRNN',\
                      'LSTM','mLSTM_fixD','mLSTM',\
                      'GRU','mGRU_fixD''mGRU']: # test different models
    
        if algorithm == 'RNN':
            #model = models.RNN(input_size=7,hidden_size=64,output_size=1)
            model = models.RNN(input_size=input_size,hidden_size=hidden_size,output_size=output_size)
        elif algorithm == 'LSTM':
            #model = models.LSTM(input_size=7,hidden_size=64,output_size=1)
            model = LSTM(input_size=input_size,hidden_size=hidden_size,output_size=output_size)
        elif algorithm == 'GRU':
            model = GRU(input_size=input_size,hidden_size=hidden_size,output_size=output_size)
        elif algorithm == 'mRNN_fixD':
            #model = models.MRNNFixD(input_size=7,hidden_size=64,output_size=1,k=25) # currently use fixed K
            model = models.MRNNFixD(input_size=input_size,hidden_size=hidden_size,output_size=output_size,k=K) # currently use fixed K
        elif algorithm == 'mRNN':
            #model = models.MRNN(input_size=7,hidden_size=64,output_size=1,k=25)
            model = models.MRNN(input_size=input_size,hidden_size=hidden_size,output_size=output_size,k=K)
        elif algorithm == 'mLSTM_fixD':
            #model = models.MLSTMFixD(input_size=7,hidden_size=64,output_size=1,k=25)
            model = models.MLSTMFixD(input_size=input_size,hidden_size=hidden_size,output_size=output_size,k=K)
        elif algorithm == 'mLSTM':
            #model = models.MLSTM(input_size=7,hidden_size=64,output_size=1,k=25)
            model = models.MLSTM(input_size=input_size,hidden_size=hidden_size,output_size=output_size,k=K)
        elif algorithm == 'mGRU_fixD':
            model = MGRUFixD(input_size=input_size,hidden_size=hidden_size,output_size=output_size,k=K)
        elif algorithm == 'mGRU':
            model = MGRU(input_size=input_size,hidden_size=hidden_size,output_size=output_size,k=K)
            
        #model = model
            
        # evaluation metrics
        rmse_list = []; mae_list = []
            
        print("model tested: %s"%algorithm)
            
        criterion = nn.MSELoss() # MSE loss is used for training
        optimizer = optim.Adam(model.parameters(),lr=lr) # should set a learning rate decay here
        scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=100,gamma=0.5)
        best_loss = np.infty; best_train_loss = np.infty
        stop_criterion = 1e-5
        rec = np.zeros((epochs,3)) # storing training records
        epoch = 0
        val_loss = -1; train_loss = -1; cnt = 0 # what is this for?

        # model training
        def train():
            model.train()
            optimizer.zero_grad()
            target = torch.from_numpy(train_y).float()
            output,hidden_state = model(torch.from_numpy(train_x).float())
            #print(output.device,hidden_state[0].device)
            with torch.no_grad():
                val_y, _ = model(torch.from_numpy(validate_x).float(),hidden_state)
                target_val = torch.from_numpy(validate_y).float()
                #print(val_y.device,target_val.device)
                val_loss = criterion(val_y,target_val)

            loss = criterion(output,target)
            loss.backward()
            optimizer.step()
            return loss,val_loss
        
        # model evaluation
        # output from the model is not on the device
        def compute_test(best_model):
            model = best_model
            train_predict, hidden_state = model(to_torch(train_x))
            train_predict = train_predict.cpu().detach().numpy()
            val_predict, hidden_state = model(to_torch(validate_x),hidden_state)
            test_predict, _ = model(to_torch(test_x),hidden_state)
            test_predict = test_predict.cpu().detach().numpy()
            # invert predictions
            test_predict_r = scaler.inverse_transform(test_predict[:,0,:])
            test_y_r = scaler.inverse_transform(test_y[:,0,:])
            # calculate error
            test_rmse = math.sqrt(mean_squared_error(test_y_r[:,0],test_predict_r[:,0]))
            test_mape = (abs((test_predict_r[:,0]-test_y_r[:,0])/test_y_r[:,0])).mean()
            test_mae = mean_absolute_error(test_predict_r[:,0],test_y_r[:,0])
            return test_rmse,test_mape,test_mae
        
        # main function for model training
        while epoch < epochs:
            _time = time.time()
            loss,val_loss = train()
            if val_loss < best_loss:
                best_loss = val_loss; best_epoch = epoch; best_model = deepcopy(model)
            # stop_criteria = abs(criterion(val_Y,target_val)-val_loss)
            if (best_train_loss-loss) > stop_criterion:
                best_train_loss = loss; cnt = 0
            else:
                cnt += 1
            if cnt == patience: # what does patience here mean?
                break
            # save training records
            time_elapsed = time.time()-_time
            rec[epoch,:] = np.array([loss.cpu().detach().numpy(),val_loss.cpu().detach().numpy(),time_elapsed])
            print("epoch: {:2.0f} train_loss: {:2.5f} val_loss: {:2.5f} time: {:2.1f}s".format(epoch,loss.item(),val_loss.item(),time_elapsed))
            epoch = epoch + 1

        # test model performance
        test_rmse,test_mape,test_mae = compute_test(best_model)
        
        # store the results
        torch.save({"train_val_rec":rec,"rmse_list":rmse_list,"mae_list":mae_list},\
                    os.path.join("results","exo_%s_%s_%s.sav"%(dataset,algorithm,seed)))

        rmse_list.append(test_rmse);mae_list.append(test_mae)
        print('RMSE:{}'.format(rmse_list)); print('MAE:{}'.format(mae_list))
