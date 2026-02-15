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
        self.model=nn.Sequential(nn.Linear(input,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,1))

    def forward(self,x):
        return self.model(x)

class Fitting:
    def __init__(self,input,lr=0.001,epochs=50):
        self.model=TimeSeriesModel(input)
        self.criterion=nn.MSELoss()
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=lr)
        self.epochs=epochs

    def train(self,x,y):
        xTensor=torch.tensor(x,dtype=torch.float32)
        yTensor=torch.tensor(y,dtype=torch.float32).view(-1,1)
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            outputs=self.model(xTensor)
            loss=self.criterion(outputs,yTensor)
            loss.backward()
            self.optimizer.step()

            if epoch%10==0:
                print(f"Epoch [{epoch}/{self.epochs}] Loss: {loss.item():.4f}")
            
        torch.save(self.model.state_dict(),"model.pth")
        print("Model saved")

class Prediction:
    def __init__(self,input):
        self.model=TimeSeriesModel(input)
        self.model.load_state_dict(torch.load("model.pth"))
        self.model.eval()
        self.scaler=joblib.load("scaler.pkl")

    def predict(self,x):
        xScaled=self.scaler.transform(x)
        xTensor=torch.tensor(xScaled,dtype=torch.float32)
        prediction=self.model(xTensor)
        return prediction.detach().numpy()
    
    def recommend(self,predict_val,nutrient):
        recommendations=[]
        optimal_range={
            "N_ppm": (120, 200),
            "P_ppm": (35, 50),
            "K_ppm": (180, 250),
            "pH": (5.5, 6.5),
            "EC_mS_cm": (1.2, 1.8)
        }
        low,high=optimal_range[nutrient]

        if predict_val<low:
            recommendations.append(f"Increase {nutrient}")
        elif predict_val>high:
            recommendations.append(f"Decrease {nutrient}")
        else:
            recommendations.append(f"{nutrient} is optimal")

        return recommendations
    
if __name__=="__main__":
    target="N_ppm" #just a case
    processor=ProcessData(path="Pipeline1/lettuce_hydroponic_data.csv",target=target,steps=12)
    x,y=processor.preprocessing()
    modelfit=Fitting(input=x.shape[1])
    modelfit.train(x,y)