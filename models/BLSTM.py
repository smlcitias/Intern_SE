import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.normalization import LayerNorm
import pdb
import torch
        
class Blstm(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers=1,dropout=0):
        super().__init__()
        self.blstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, num_layers=num_layers, dropout=dropout, bidirectional=True)

    def forward(self,x):
        out,_=self.blstm(x)
        out = out[:,:,:int(out.size(-1)/2)]+out[:,:,int(out.size(-1)/2):] 
        return out
        
class BLSTM_01(nn.Module):
    
    def __init__(self,):
        super().__init__()
        
        self.lstm_enc = nn.Sequential(           
            Blstm(input_size=257, hidden_size=300, num_layers=2),
            nn.Linear(300, 257, bias=True),
        )
        
    def forward(self,x):
        se_out = self.lstm_enc(x)      
    
        return se_out
    
class BLSTM_02(nn.Module):
    
    def __init__(self,):
        super().__init__()
        
        self.lstm_enc = nn.Sequential(           
            Blstm(input_size=257, hidden_size=300, num_layers=2),
            nn.Linear(300, 257, bias=True),
            nn.ReLU(),
        )
        
    def forward(self,x):
        se_out = self.lstm_enc(x)      
    
        return se_out