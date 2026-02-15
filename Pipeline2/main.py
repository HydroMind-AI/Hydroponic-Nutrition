import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import math
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error

class PositionalEncoding(nn.Module):
    def __init__(self,d_model,max_len=5000):
        super(PositionalEncoding,self).__init__()
        pe=torch.zeros(max_len,d_model)
        position=torch.arange(0,max_len,dtype=torch.float).unsqueeze(1)
        div_term=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
        pe[:,0::2]=torch.sin(position*div_term)
        pe[:,1::2]=torch.cos(position*div_term)
        pe=pe.unsqueeze(0)
        self.register_buffer('pe',pe)

    def forward(self,x):
        return x+self.pe[:,:x.size(1),:]

class ProcessData:
    def __init__(self,path,target,seq_length=12):
        self.path=path
        self.target=target
        self.seq_length=seq_length
        self.scaler=StandardScaler()

    def loadData(self):
        df=pd.read_csv(self.path)
        df["Timestamp"]=pd.to_datetime(df["Timestamp"])
        df=df.sort_values("Timestamp")
        df.reset_index(drop=True,inplace=True)
        return df

    def createSequences(self,df):
        data=df[self.target].values
        X,y=[],[]
        for i in range(len(data)-self.seq_length):
            X.append(data[i:i+self.seq_length])
            y.append(data[i+self.seq_length])
        return np.array(X),np.array(y)

    def preprocessing(self):
        df=self.loadData()
        X,y=self.createSequences(df)
        X=X.reshape(-1,1)
        X_scaled=self.scaler.fit_transform(X)
        X_scaled=X_scaled.reshape(-1,self.seq_length)
        joblib.dump(self.scaler,"scaler_transformer.pkl")
        return X_scaled,y

class TransformerTimeSeries(nn.Module):
    def __init__(self,input_dim=1,d_model=64,nhead=4,num_layers=2,dropout=0.1):
        super(TransformerTimeSeries,self).__init__()
        self.input_projection=nn.Linear(input_dim,d_model)
        self.pos_encoder=PositionalEncoding(d_model)
        encoder_layer=nn.TransformerEncoderLayer(d_model=d_model,nhead=nhead,dim_feedforward=256,dropout=dropout,batch_first=True)
        self.transformer_encoder=nn.TransformerEncoder(encoder_layer,num_layers=num_layers)
        self.output_layer=nn.Linear(d_model,1)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x):
        x=x.unsqueeze(-1)
        x=self.input_projection(x)
        x=self.pos_encoder(x)
        x=self.transformer_encoder(x)
        x=x[:,-1,:]
        x=self.dropout(x)
        x=self.output_layer(x)
        return x

class Fitting:
    def __init__(self,d_model=64,nhead=4,num_layers=2,lr=0.001,epochs=100,batch_size=32):
        self.model=TransformerTimeSeries(input_dim=1,d_model=d_model,nhead=nhead,num_layers=num_layers)
        self.criterion=nn.MSELoss()
        self.optimizer=torch.optim.AdamW(self.model.parameters(),lr=lr,weight_decay=0.01)
        self.scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,mode='min',factor=0.5,patience=5)
        self.epochs=epochs
        self.batch_size=batch_size

    def createDataLoader(self,X,y,shuffle=True):
        dataset=torch.utils.data.TensorDataset(torch.tensor(X,dtype=torch.float32),torch.tensor(y,dtype=torch.float32).view(-1,1))
        return torch.utils.data.DataLoader(dataset,batch_size=self.batch_size,shuffle=shuffle)

    def train(self,X_train,y_train,X_val,y_val):
        train_loader=self.createDataLoader(X_train,y_train,shuffle=True)
        val_loader=self.createDataLoader(X_val,y_val,shuffle=False)
        best_val_loss=float('inf')
        patience_counter=0
        
        for epoch in range(self.epochs):
            self.model.train()
            train_loss=0
            for batch_X,batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs=self.model(batch_X)
                loss=self.criterion(outputs,batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),max_norm=1.0)
                self.optimizer.step()
                train_loss+=loss.item()
            
            train_loss/=len(train_loader)
            
            self.model.eval()
            val_loss=0
            with torch.no_grad():
                for batch_X,batch_y in val_loader:
                    outputs=self.model(batch_X)
                    loss=self.criterion(outputs,batch_y)
                    val_loss+=loss.item()
            
            val_loss/=len(val_loader)
            self.scheduler.step(val_loss)
            
            if val_loss<best_val_loss:
                best_val_loss=val_loss
                patience_counter=0
                torch.save(self.model.state_dict(),"model_transformer.pth")
            else:
                patience_counter+=1
            
            if epoch%10==0:
                print(f"Epoch [{epoch}/{self.epochs}] Train Loss: {train_loss:.6f} Val Loss: {val_loss:.6f}")
            
            if patience_counter>=15:
                print(f"Early stopping at epoch {epoch}")
                break
        
        self.model.load_state_dict(torch.load("model_transformer.pth"))
        print("Best model loaded and saved")

class Prediction:
    def __init__(self,d_model=64,nhead=4,num_layers=2):
        self.model=TransformerTimeSeries(input_dim=1,d_model=d_model,nhead=nhead,num_layers=num_layers)
        self.model.load_state_dict(torch.load("model_transformer.pth"))
        self.model.eval()
        self.scaler=joblib.load("scaler_transformer.pkl")

    def predict(self,x):
        x_scaled=self.scaler.transform(x.reshape(-1,1)).reshape(x.shape)
        x_tensor=torch.tensor(x_scaled,dtype=torch.float32)
        with torch.no_grad():
            prediction=self.model(x_tensor)
        return prediction.numpy()

    def recommend(self,predict_val,nutrient):
        recommendations=[]
        optimal_range={
            "N_ppm":(120,200),
            "P_ppm":(35,50),
            "K_ppm":(180,250),
            "pH":(5.5,6.5),
            "EC_mS_cm":(1.2,1.8)
        }
        low,high=optimal_range[nutrient]

        if predict_val<low:
            recommendations.append(f"Increase {nutrient}")
        elif predict_val>high:
            recommendations.append(f"Decrease {nutrient}")
        else:
            recommendations.append(f"{nutrient} is optimal")

        return recommendations

class ModelEvaluator:
    def __init__(self,model,scaler):
        self.model=model
        self.scaler=scaler

    def evaluate(self,X_test,y_test):
        self.model.eval()
        X_test_tensor=torch.tensor(X_test,dtype=torch.float32)
        with torch.no_grad():
            predictions=self.model(X_test_tensor).numpy()
        
        predictions_original=self.scaler.inverse_transform(predictions)
        rmse=np.sqrt(mean_squared_error(y_test,predictions_original))
        mae=mean_absolute_error(y_test,predictions_original)
        
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE:  {mae:.4f}")
        print("="*50)
        
        return rmse,mae,predictions_original

if __name__=="__main__":
    print("="*60)
    print("MODERN TRANSFORMER-BASED HYDROPONIC NUTRITION PREDICTOR")
    print("Architecture: Multi-Head Self-Attention Transformer (2024)")
    print("="*60)
    
    target="N_ppm"
    processor=ProcessData(path="Pipeline2/lettuce_hydroponic_data.csv",target=target,seq_length=12)
    X,y=processor.preprocessing()
    
    split_idx=int(0.8*len(X))
    X_train,y_train=X[:split_idx],y[:split_idx]
    X_temp,y_temp=X[split_idx:],y[split_idx:]
    
    val_split=int(0.5*len(X_temp))
    X_val,y_val=X_temp[:val_split],y_temp[:val_split]
    X_test,y_test=X_temp[val_split:],y_temp[val_split:]
    
    modelfit=Fitting(d_model=64,nhead=4,num_layers=2,lr=0.001,epochs=100,batch_size=32)
    
    modelfit.train(X_train,y_train,X_val,y_val)
    
    scaler=joblib.load("scaler_transformer.pkl")
    evaluator=ModelEvaluator(modelfit.model,scaler)
    rmse,mae,predictions=evaluator.evaluate(X_test,y_test)
    
    predictor=Prediction(d_model=64,nhead=4,num_layers=2)
    
    for i in range(min(5,len(X_test))):
        sample=X_test[i:i+1]
        pred=predictor.predict(sample)[0][0]
        pred_original=scaler.inverse_transform([[pred]])[0][0]
        actual=y_test[i]
        recommendations=predictor.recommend(pred_original,target)
        
        print(f"\nSample {i+1}:")
        print(f"  Actual {target}: {actual:.2f}")
        print(f"  Predicted {target}: {pred_original:.2f}")
        print(f"  Error: {abs(actual-pred_original):.2f}")
        print(f"  Recommendation: {recommendations[0]}")
    
    print("\n" + "="*60)
    print("="*60)