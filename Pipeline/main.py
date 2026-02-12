#required libraries
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#preprocessing data
class ProcessData:
    def __init__(self,path,target,steps=12):
        self.path=path
        self.target=target
        self.steps=steps
        self.scaler=StandardScaler()

    def loadData(self):
        df=pd.read_csv(self.path)
        df["Timestamp"]=pd.to_datetime(df["Timestamp"])
        df=df.sort_values("Timestamp")
        df.reset_index(drop=True,inplace=True)
        return df
    
    def createLag(self,df):
        for lag in range(1,self.steps+1):
            df[f"{self.target}_lag_{lag}"]=df[self.target].shift(lag)
        df=df.dropna()
        return df
    
    def preprocessing(self):
        df=self.loadData()
        df=self.createLag(df)
        x=df.drop(columns=["Timestamp",self.target])
        y=df[self.target]
        xScaled=self.scaler.fit_transform(x)
        joblib.dump(self.scaler,"scaler.pkl")
        return xScaled,y.values
    
#time series model
class TimeSeriesModel(nn.Module):
    def __init__(self,input):
        super(TimeSeriesModel,self).__init__()
        self.model=nn.Sequential(nn.Linear(input,128),nn.ReLU(),nn.Linear(128,128),nn.ReLu(),nn.Linear(128,1))

    def forward(self,x):
        return self.model(x)