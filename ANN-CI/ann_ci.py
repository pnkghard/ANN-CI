#!/usr/bin/python

from numpy import vstack
import math
from pandas import read_csv
#from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,mean_squared_error
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import Tensor
import torch
from torch.nn import Sigmoid,Softmax,ReLU,Linear,Tanh
from torch.nn import Module
from torch.optim import SGD,Adam
from torch.nn import BCELoss,NLLLoss,CrossEntropyLoss,MSELoss
from torch.nn.init import kaiming_uniform_,xavier_uniform_
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
from setup import readInput


n_inputs, frac_val, frac_test, H, l_rate, batch, epochs, test_datapath,path = readInput()

#n_inputs = 18
#H = 20     # Number of nodes in a hidden layer (please check if it is 1 hidden layer)
f = open("output_Amplitude_train.out","w")


# dataset loading
class CSVDataset(Dataset):
    def __init__(self, path,lst,det,flag):
        df  = read_csv(path,usecols=lst, header=None)
        df_det = read_csv(path,usecols=[int(det)], header=None)

        if flag:
            self.X = df.values[:, :-1]
            self.y = df.values[:, -1]
            self.det = df_det.values[:,-1]

            self.X = self.X.astype('float32')
            self.y = self.y.astype('float32')
            self.det = self.det.astype('float32')

            self.y = self.y.reshape((len(self.y), 1))
            self.det = self.det.reshape((len(self.det), 1))

        else:
            self.X = df.values
            self.det = df_det.values

            self.X = self.X.astype('float32')
            self.det = self.det.astype('float32')

            self.det = self.det.reshape((len(self.det), 1))


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if flag:
            return [self.X[idx], self.y[idx], self.det[idx]]
        else:
            return [self.det[idx],self.X[idx]]
            

    def get_splits(self, n_test):                              # spliting of dataset 
        test_size = int(round(n_test * len(self.X)))
        train_size = int(len(self.X) - test_size)
        return random_split(self, [train_size, test_size])


#------------------------------PREPARE THE DATA--------------------------------------#
def prepare_data(path,lst,det,frac,flag):
    dataset = CSVDataset(path,lst,det,flag)
    train, test = dataset.get_splits(frac)
    train_dl = DataLoader(train, batch_size=batch, shuffle=False)
    test_dl = DataLoader(test, batch_size=batch, shuffle=False)
    return train_dl

def prepare_testdata(path,lst,det,frac,flag):
    dataset = CSVDataset(path,lst,det,flag)
    train, test = dataset.get_splits(frac)
    test_dl = DataLoader(test, batch_size=batch, shuffle=False)
    return test_dl

#====================================================================================#

# prepare the data
#path = 'input_train_18sites.csv'
train_list= [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,21]
det=19
#frac_val=0.0
flag=True
train_dl = prepare_data(path,train_list,det,frac_val,flag)


#---------------------------------- MODEL STRUCTURE ----------------------------------#

class Network(Module):
    def __init__(self,n_inputs):
        super(Network,self).__init__()
        
        #input descriptor
        self.hidden1 = Linear(n_inputs, H)                           # input to 1st hidden layer
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.relu = ReLU()
        self.output = Linear(H, 1)                                  # 2nd hidden layer to output
        xavier_uniform_(self.output.weight)
        self.relu = ReLU()
    def forward(self, X):
        X = self.hidden1(X)            
        X = self.relu(X)
        X = self.output(X)
        X = self.relu(X)


        return X

#====================================================================================#



model = Network(n_inputs)


#----------------------------------TRAINING------------------------------------------#

def train_model(train_dl, model):
    criterion = MSELoss()                                               #loss function
    optimizer = Adam(model.parameters(), lr=l_rate, betas=(0.9, 0.999))  #optimizer should be used
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, targets, dets) in enumerate(train_dl):
            optimizer.zero_grad()
            yhat = model(inputs)
            loss = criterion(yhat, targets)
            loss.backward()
            optimizer.step()
            running_loss +=loss.item()
        epoch_loss = running_loss/len(train_dl)
        #print(str(epoch)+"  "+str(epoch_loss))
        plt.scatter(epoch, epoch_loss,c='k')
        plt.pause(1e-17)
        time.sleep(0.00001)
        #print(epcoh,'\t', epoch_loss)
    print('\n\nYep!! The training is done !!')

#==================================================================================#

train_model(train_dl, model)
torch.save(model.state_dict(), "model.pth")



#test_datapath = "18fciDet_cutoff.csv"
test_list= [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
det=0
#frac_test=1.0
flag=False
test_dl = prepare_testdata(test_datapath,test_list,det,frac_test,flag)
#-----------------------------------EVALUATION-------------------------------------#

def evaluate_model(test_dl, model):                                          #send the test dataset through the network
    for i, (dets,inputs) in enumerate(test_dl):
        yhat = model(inputs)
        yhat = yhat.detach().numpy()
        dets = dets.numpy()
        for j in range(len(yhat)):
            f.write(str(int(dets[j][0]))+"   "+str(10**(-1.0*yhat[j][0]))+"\n")
            #print (int(dets[j][0]),yhat[j][0])
    return True

#----------------------------------------------------------------------------------#

#==================================================================================#

evaluate_model(test_dl, model)
f.close()

