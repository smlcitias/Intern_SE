import torch, os
import torch.nn as nn
from torch.optim import Adam,SGD
from util import *
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data.dataset import Dataset
import pdb
from tqdm import tqdm 
from joblib  import parallel_backend, Parallel, delayed
from collections import OrderedDict
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchaudio

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

def weights_init(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        for name, param in m.named_parameters():
            if param.requires_grad==True:
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_normal_(param)
        
        
def load_checkoutpoint(model,optimizer,checkpoint_path):

    if os.path.isfile(checkpoint_path):
        model.eval()
        print("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        try:
            model.load_state_dict(checkpoint['model'])
        except:
            model.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['model'].items()})
#             model.load_state_dict({k.replace('module.',''):v for k,v in torch.load('checkpoint.pt').items()})

        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        print(f"=> loaded checkpoint '{checkpoint_path}' (epoch {epoch})")
        
        return model,epoch,best_loss,optimizer
    else:
        raise NameError(f"=> no checkpoint found at '{checkpoint_path}'")



def Load_model(args,model,checkpoint_path,model_path):
    
    criterion = {
        'mse'     : nn.MSELoss(),
        'l1'      : nn.L1Loss(),
        'l1smooth': nn.SmoothL1Loss(),
        'cosine'  : nn.CosineEmbeddingLoss()
    }

    device    = torch.device(f'cuda:{args.gpu}')
    if args.loss=='stoi':
        criterion = criterion['mse'].to(device)
    else:
        criterion = criterion[args.loss].to(device)
    
    optimizers = {
        'adam'    : Adam(model.parameters(),lr=args.lr,weight_decay=0),
        'SGD'     : SGD(model.parameters(),lr=args.lr,weight_decay=0)
    }
    
    optimizer = optimizers[args.optim]
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=44, verbose=True, threshold=1e-8, threshold_mode='rel', cooldown=22, min_lr=0, eps=1e-8)
    
    if args.resume:
        model,epoch,best_loss,optimizer = load_checkoutpoint(model,optimizer,checkpoint_path)
    elif args.retrain:
        model,epoch,best_loss,optimizer = load_checkoutpoint(model,optimizer,model_path)
               
    else:
        epoch = 0
        best_loss = 500
        model.apply(weights_init)
        
    para = count_parameters(model)
    print(f'Num of model parameter : {para}')
        
    return model,epoch,best_loss,optimizer,scheduler,criterion,device


def Load_data(args, Train_path):
    
    file_paths = get_filepaths(Train_path['noisy'],'.wav')
    clean_path = Train_path['clean']
    # pdb.set_trace()
    train_paths,val_paths = train_test_split(file_paths,test_size=0.2,random_state=999)
    
    if args.task =='VCTK':
        train_dataset, val_dataset = CustomDataset_VCTK(train_paths,clean_path,args.feature), CustomDataset_VCTK(val_paths,clean_path,args.feature)
    
    elif args.task =='TMHINTQI_V2':
        train_dataset, val_dataset = CustomDataset_TMHINTQI_V2(train_paths,clean_path,args.feature), CustomDataset_TMHINTQI_V2(val_paths,clean_path,args.feature)

    loader = { 
        'train':DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=False),
        'val'  :DataLoader(val_dataset, batch_size=args.batch_size,
                            num_workers=4, pin_memory=False)
    }

    return loader

def load_torch(path):
    return torch.load(path)

class CustomDataset_VCTK(Dataset):

    def __init__(self, paths, clean_path,feature):   # initial logic happens like transform
        
        self.feature = feature
        self.n_paths = paths
        self.c_paths = [os.path.join(clean_path, noisy_path.split('/')[-1]) for noisy_path in paths]        
    def __getitem__(self, index):
        
        noisy, sr = torchaudio.load(self.n_paths[index])
        # y, y_phase, y_len = make_spectrum_torch(wave=noisy,feature_type = self.feature)
        
        clean, sr = torchaudio.load(self.c_paths[index])
        # c, c_phase, c_len = make_spectrum_torch(wave=clean,feature_type = self.feature)
        
        return noisy, clean
        # return y, y_phase, c, c_phase, y_len

    def __len__(self):  # return count of sample we have
        
        return len(self.n_paths)        
        
class CustomDataset_TMHINTQI_V2(Dataset):

    def __init__(self, paths, clean_path,feature):   # initial logic happens like transform
        
        self.feature = feature
        self.n_paths = paths
        self.c_paths = [os.path.join(clean_path, noisy_path.split('/')[-1].split('_')[-4]+'_'+noisy_path.split('/')[-1].split('_')[-3]+'_'+noisy_path.split('/')[-1].split('_')[-2]+'_'+noisy_path.split('/')[-1].split('_')[-1]) for noisy_path in paths]
    def __getitem__(self, index):
        
        noisy, sr = torchaudio.load(self.n_paths[index])
        # y, y_phase, y_len = make_spectrum_torch(wave=noisy,feature_type = self.feature)
        
        clean, sr = torchaudio.load(self.c_paths[index])
        # c, c_phase, c_len = make_spectrum_torch(wave=clean,feature_type = self.feature)
        
        return noisy, clean

    def __len__(self):  # return count of sample we have
        
        return len(self.n_paths)       
