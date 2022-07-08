import torch.nn as nn
import torch
import pandas as pd
import os, sys
from tqdm import tqdm
import librosa, scipy
import pdb
import numpy as np
from scipy.io.wavfile import write as audiowrite
from util import *
# import pyworld as pw
from sklearn import preprocessing
# import torchaudio

maxv = np.iinfo(np.int16).max
epsilon = np.finfo(float).eps

class Trainer:
    def __init__(self, model, version, epochs, epoch, best_loss, optimizer,scheduler, 
                 criterion, device, loader, Test_path, writer, model_path, score_path, args, Output_path, save_results, target):
        self.epoch = epoch  
        self.epochs = epochs
        self.best_loss = best_loss
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.version = version
        self.device = device
        self.loader = loader
        self.criterion = criterion
        self.save_results = save_results
        self.target = target
        
        self.Test_path = Test_path
        self.Output_path = Output_path

        self.train_loss = 0
        self.val_loss = 0
        self.writer = writer
        self.model_path = model_path
        self.score_path = score_path
        self.args = args

    def save_checkpoint(self,):
        state_dict = {
            'epoch': self.epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_loss': self.best_loss
            }
        check_folder(self.model_path)
        torch.save(state_dict, self.model_path)
         
    def _train_step(self, in_y, in_c, target):
        
        device = self.device
        # spec_y, spec_c = in_y.to(device), in_c.to(device)  
        spec_y, spec_c = in_y.transpose(1,2).to(device), in_c.transpose(1,2).to(device)           
        log1p_y = torch.log1p(spec_y)
        log1p_c = torch.log1p(spec_c)
        
        if target == 'MAP':
            pred = self.model(log1p_y)       
            loss = self.criterion(pred, log1p_c)
            
        elif target == 'IRM':      
            pred_irm = self.model(log1p_y)        
            loss = self.criterion(log1p_y*pred_irm, log1p_c)
        
        self.train_loss += loss.item()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # self.scheduler.step(loss)


    def _train_epoch(self):
        self.train_loss = 0

        progress = tqdm(total=len(self.loader['train']), desc=f'Epoch {self.epoch} / Epoch {self.epochs} | train', unit='step')
        self.model.train()
         
        for spec_y, spec_c in self.loader['train']:
            self._train_step(spec_y, spec_c, self.target)
            progress.update(1)
            
        progress.close()
        self.train_loss /= len(self.loader['train'])

        print(f'train_loss:{self.train_loss}')

    def _val_step(self, in_y, in_c, target):
        
        device = self.device
        # spec_y, spec_c = in_y.to(device), in_c.to(device) 
        spec_y, spec_c = in_y.transpose(1,2).to(device), in_c.transpose(1,2).to(device)           
        log1p_y = torch.log1p(spec_y)
        log1p_c = torch.log1p(spec_c)
        
        if target == 'MAP':
            pred = self.model(log1p_y)       
            loss = self.criterion(pred, log1p_c)
            
        elif target == 'IRM':      
            pred_irm = self.model(log1p_y)        
            loss = self.criterion(log1p_y*pred_irm, log1p_c)
        
        self.val_loss += loss.item()   
       
        # self.scheduler.step(loss)

    def _val_epoch(self):
        self.val_loss = 0
     
        progress = tqdm(total=len(self.loader['val']), desc=f'Epoch {self.epoch} / Epoch {self.epochs} | valid', unit='step')
        self.model.eval()

        for spec_y, spec_c in self.loader['val']:
            self._val_step(spec_y, spec_c, self.target)
            progress.update(1)

        progress.close()

        self.val_loss /= len(self.loader['val'])
        
        print(f'val_loss:{self.val_loss}')
       
        if self.best_loss > self.val_loss:
            
            print(f"Save model to '{self.model_path}'")
            self.save_checkpoint()
            self.best_loss = self.val_loss

            
    def write_score(self, test_file, clean_path, audio_path, target):
        
        self.model.eval()
        noisy, sr = librosa.load(test_file,sr=16000)
        clean, sr = librosa.load(os.path.join(clean_path, test_file.split('/')[-1]),sr=16000)
        log1p_y, y_phase, y_len = make_spectrum(y=noisy,feature_type ='log1p')
        
        log1p_y_1 = torch.from_numpy(log1p_y).cuda().detach().transpose(0,1)
        pred = self.model(log1p_y_1.unsqueeze(0))
        pred = pred.cpu().detach().numpy().squeeze(0).T
                        
        if target == 'MAP':
            pred_clean = recons_spec_phase(pred, y_phase, y_len, feature_type='log1p')
        elif target == 'IRM':      
            pred_clean = recons_spec_phase(pred*log1p_y, y_phase, y_len, feature_type='log1p')

        if self.save_results == 'True':
            out_a_path = os.path.join(audio_path,  f"{test_file.split('/')[-1].split('.')[0]+'.wav'}")
            check_folder(out_a_path)
            audiowrite(out_a_path,16000,(pred_clean* maxv).astype(np.int16))

        
        clean = clean/abs(clean).max()
        noisy = noisy/abs(noisy).max()
        pred_clean_wav = pred_clean/abs(pred_clean).max()
        
        n_pesq, n_stoi = cal_score(clean,noisy)
        s_pesq, s_stoi = cal_score(clean,pred_clean_wav)
        
        wave_name = test_file.split('/')[-1].split('.')[0]
        with open(self.score_path['PESQ'], 'a') as f:
            f.write(f'{wave_name},{n_pesq},{s_pesq}\n')
        with open(self.score_path['STOI'], 'a') as f:
            f.write(f'{wave_name},{n_stoi},{s_stoi}\n')

    def train(self):
        while self.epoch < self.epochs:
            self._train_epoch()
            self._val_epoch()
            
            self.sc_name = f'{self.args.task}/{self.model.__class__.__name__}_{self.args.optim}' \
            f'_{self.args.loss}'
            
            self.writer.add_scalars(self.sc_name, {'train': self.train_loss},self.epoch)
            self.writer.add_scalars(self.sc_name, {'val': self.val_loss},self.epoch)
                                
            self.epoch += 1
            
    def test(self):
        # load model
        self.model.eval()
        checkpoint = torch.load(self.model_path)
        self.model.load_state_dict(checkpoint['model'])        
        
        test_folders = get_filepaths(self.Test_path['noisy'],'.wav')
        test_folders = test_folders
        clean_path = self.Test_path['clean']
        
        audio_path = self.Output_path['audio']        
        print(self.score_path)
        
        check_folder(self.score_path['PESQ'])
        if os.path.exists(self.score_path['PESQ']):
            os.remove(self.score_path['PESQ'])
        with open(self.score_path['PESQ'], 'a') as f:
            f.write('Filename,Noisy_PESQ,Pred_PESQ\n')
            
        check_folder(self.score_path['STOI'])
        if os.path.exists(self.score_path['STOI']):
            os.remove(self.score_path['STOI'])
        with open(self.score_path['STOI'], 'a') as f:
            f.write('Filename,Noisy_STOI,Pred_STOI\n')   
            
        for test_file in tqdm(test_folders):
            
            self.write_score(test_file, clean_path, audio_path, self.target)
        
        data = pd.read_csv(self.score_path['PESQ'])
        n_pesq_mean = data['Noisy_PESQ'].to_numpy().astype('float').mean()
        s_pesq_mean = data['Pred_PESQ'].to_numpy().astype('float').mean()

        with open(self.score_path['PESQ'], 'a') as f:
            f.write(','.join(('Average',str(n_pesq_mean),str(s_pesq_mean)))+'\n')


        data = pd.read_csv(self.score_path['STOI'])
        n_stoi_mean = data['Noisy_STOI'].to_numpy().astype('float').mean()
        s_stoi_mean = data['Pred_STOI'].to_numpy().astype('float').mean()
        
        with open(self.score_path['STOI'], 'a') as f:
            f.write(','.join(('Average',str(n_stoi_mean),str(s_stoi_mean)))+'\n')
    
    
