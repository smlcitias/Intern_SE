import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.normalization import LayerNorm
import pdb
import torch
        
class lstm(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers=1,dropout=0):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, num_layers=num_layers, dropout=dropout, bidirectional=False)

    def forward(self,x):
        out,_=self.lstm(x)
        return out
    
class reBlstm(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers=1,dropout=0):
        super().__init__()
        self.blstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, num_layers=num_layers, dropout=dropout, bidirectional=True)

    def forward(self,x):
        out,_=self.blstm(x)
        out = out[:,:,:int(out.size(-1)/2)]+out[:,:,int(out.size(-1)/2):] 
        return out
    
    
class LSTM_2O2L_idlps(nn.Module):
    
    def __init__(self,):
        super().__init__()
        
        self.lstm_enc = nn.Sequential(
            nn.Linear(257, 256, bias=True),
            lstm(input_size=256, hidden_size=512, num_layers=2),          
        )
        self.fc_irm  = nn.Sequential(
            nn.Linear(512, 256, bias=True),
            nn.Linear(256, 257, bias=True),
            # nn.ReLU(),
        )
        self.fc_xi  = nn.Sequential(
            nn.Linear(512,256, bias=True),
            nn.Linear(256, 257, bias=True),
            # nn.ReLU(),
        )

    def forward(self,x):

        deco = self.lstm_enc(x)      
        irm_out = self.fc_irm(deco)
        irm_out = irm_out
        
        xi_out = self.fc_xi(deco)
        xi_out = xi_out

        return irm_out, xi_out
    
class LSTM_2O5L_idlps(nn.Module):
    
    def __init__(self,):
        super().__init__()
        
        self.lstm_enc = nn.Sequential(
            nn.Linear(257, 256, bias=True),
            lstm(input_size=256, hidden_size=512, num_layers=5),          
        )
        self.fc_irm  = nn.Sequential(
            nn.Linear(512, 256, bias=True),
            nn.Linear(256, 257, bias=True),
            # nn.ReLU(),
        )
        self.fc_xi  = nn.Sequential(
            nn.Linear(512,256, bias=True),
            nn.Linear(256, 257, bias=True),
            # nn.ReLU(),
        )

    def forward(self,x):

        deco = self.lstm_enc(x)      
        irm_out = self.fc_irm(deco)
        irm_out = irm_out
        
        xi_out = self.fc_xi(deco)
        xi_out = xi_out

        return irm_out, xi_out
    
