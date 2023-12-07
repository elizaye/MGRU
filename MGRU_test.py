# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 23:58:06 2023

@author: eliza
"""

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
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import torch

def create_dataset(x, y):
    """creat train dataset for time series prediction"""
    dataYp, dataYc = [], []
    for i in range(len(y) - 1):
        dataYp.append(x[i, :])
        dataYc.append(y[i + 1, :])
    return np.array(dataYp), np.array(dataYc)

def padding(data, length, input_size):
    term = [0] * input_size
    data_pad = []
    for text in data:
        if len(text) >= length:
            text = np.array(text[0:length])
        else:
            pad_list = np.array([term] * (length - len(text)))
            text = np.vstack([np.array(text), pad_list])
        data_pad.append(text)
    data_pad = np.array(data_pad)
    return data_pad

def accuracy(output, labels):
    """compute the accuracy for review classification"""
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

# transform data to tensor in torch
def to_torch(state):
    state = torch.from_numpy(state).float()
    return state


# In[2]:


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


# In[3]:


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

class GRUCell(nn.Module): # test if this version works
    
    # check how to better write this version
    
    
    pass


# In[4]:


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
    
class GRU(nn.Module):
    
    pass
    
    # check how to write this part


# In[8]:


os.makedirs("results",exist_ok=True)


# In[12]:


# use the original long-memory datasets for testing

batch_size = 1; rep_times = 10; K = 25; lr = 0.01; epochs = 1000; hidden_size = 1 # try to use the default setting to see if there is improvement
input_size = 1; output_size = 1; patience = 100

for dataset in ['tree7','DJI','traffic','arfima']:
    
    df_data = pd.read_csv(os.path.join(os.path.join("data","time_series_prediction",'%s.csv'%dataset)))
    # split train/val/test
    if dataset == 'tree7':
        train_size = 2500; validate_size = 1000
    if dataset == 'DJI':
        train_size = 2500; validate_size = 1500
    if dataset == 'traffic':
        train_size = 1200; validate_size = 200
    if dataset == 'arfima':
        train_size = 2000; validate_size = 1200
    
    for i in range(rep_times):
        
        seed = i; print('seed ----------------------------------', seed)
        x = np.array(df_data ['x']); y = np.array(df_data ['x'])
        x = x.reshape(-1,input_size); y = y.reshape(-1, output_size)
        scaler = MinMaxScaler(feature_range=(0,1))
        x = scaler.fit_transform(x); y = scaler.fit_transform(y)
        data_x,data_y = create_dataset(x, y)

        # train & test sets splitting
        train_x,train_y = data_x[0:train_size],data_y[0:train_size]
        validate_x,validate_y = data_x[train_size:train_size+validate_size],data_y[train_size:train_size+validate_size]
        test_x,test_y = data_x[train_size + validate_size:len(data_y)],data_y[train_size + validate_size:len(data_y)]

        # reshape input to be [time steps,samples,features]
        train_x = np.reshape(train_x,(train_x.shape[0],batch_size,input_size))
        validate_x = np.reshape(validate_x, (validate_x.shape[0],batch_size,input_size))

        test_x = np.reshape(test_x,(test_x.shape[0],batch_size,input_size))
        train_y = np.reshape(train_y,(train_y.shape[0],batch_size,output_size))
        validate_y = np.reshape(validate_y,(validate_y.shape[0],batch_size,output_size))
        test_y = np.reshape(test_y,(test_y.shape[0],batch_size,output_size))

        torch.manual_seed(seed)
        
        for algorithm in ['GRU','mGRU_fixD','mGRU']:

            rmse_list = []; mae_list = []
            # initialize model
            if algorithm == 'mGRU_fixD':
                model = MGRUFixD(input_size=input_size,hidden_size=hidden_size,output_size=output_size,k=K)
            elif algorithm == 'mGRU':
                model = MGRU(input_size=input_size,hidden_size=hidden_size,output_size=output_size,k=K)
                
            model.to(device)
                
            print("model tested %s"%algorithm)
        
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(),lr=lr)
            best_loss = np.infty; best_train_loss = np.infty
            stop_criterion = 1e-5
            rec = np.zeros((epochs,3))
            epoch = 0
            val_loss = -1; train_loss = -1
            cnt = 0

            def train():
                model.train()
                optimizer.zero_grad()
                target = torch.from_numpy(train_y).float().to(device)
                output,hidden_state = model(torch.from_numpy(train_x).float().to(device))
                with torch.no_grad():
                    val_y, _ = model(torch.from_numpy(validate_x).float().to(device),hidden_state)
                    target_val = torch.from_numpy(validate_y).float().to(device)
                    val_loss = criterion(val_y,target_val)

                loss = criterion(output,target)
                loss.backward()
                optimizer.step()
                return loss, val_loss

            def compute_test(best_model):
                model = best_model
                train_predict, hidden_state = model(to_torch(train_x).to(device))
                train_predict = train_predict.cpu().detach().numpy()
                val_predict, hidden_state = model(to_torch(validate_x).to(device),hidden_state)
                test_predict, _ = model(to_torch(test_x).to(device),hidden_state)
                test_predict = test_predict.cpu().detach().numpy()
                # invert predictions
                test_predict_r = scaler.inverse_transform(test_predict[:,0,:])
                test_y_r = scaler.inverse_transform(test_y[:,0,:])
                # calculate error
                test_rmse = math.sqrt(mean_squared_error(test_y_r[:,0],test_predict_r[:,0]))
                test_mape = (abs((test_predict_r[:,0]-test_y_r[:,0])/test_y_r[:,0])).mean()
                test_mae = mean_absolute_error(test_predict_r[:,0],test_y_r[:,0])
                return test_rmse, test_mape, test_mae

            while epoch < epochs:
            #while epoch < 10:
                _time = time.time()
                loss, val_loss = train()
                if val_loss < best_loss:
                    best_loss = val_loss; best_epoch = epoch; best_model = deepcopy(model)
                # stop_criteria = abs(criterion(val_Y, target_val) - val_loss)
                if (best_train_loss - loss) > stop_criterion:
                    best_train_loss = loss
                    cnt = 0
                else:
                    cnt += 1
                if cnt == patience:
                    break
                # save training records
                time_elapsed = time.time()-_time
                rec[epoch,:] = np.array([loss.cpu().detach().numpy(),val_loss.cpu().detach().numpy(),time_elapsed])
                print("epoch: {:2.0f} train_loss: {:2.5f} val_loss: {:2.5f} time: {:2.1f}s".format(epoch,loss.item(),val_loss.item(),time_elapsed))
                epoch = epoch + 1

            # make predictions
            test_rmse,test_mape,test_mae = compute_test(best_model)

            rmse_list.append(test_rmse)
            mae_list.append(test_mae)
            print('RMSE:{}'.format(rmse_list));print('MAE:{}'.format(mae_list))
            
            # store the results
            torch.save({"train_val_rec":rec,"rmse_list":rmse_list,"mae_list":mae_list},\
                       os.path.join("results","%s_%s_%s.sav"%(dataset,algorithm,seed)))
            