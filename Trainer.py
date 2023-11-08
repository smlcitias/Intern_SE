import torch.nn as nn
import torch, torchaudio
import pandas as pd
import os, sys
from tqdm import tqdm
import scipy
import pdb
import numpy as np
from scipy.io.wavfile import write as audiowrite
from util import *
from sklearn import preprocessing
from speechbrain.nnet.loss.stoi_loss import stoi_loss

maxv = torch.iinfo(int).max
epsilon = torch.finfo(float).eps

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
        wav_y, wav_c = in_y.transpose(1,2).squeeze(2).to(device), in_c.transpose(1,2).squeeze(2).to(device)   # (B, L)

        if self.args.feature== 'wave':
            y = padpad(wave=wav_y, stride=48).unsqueeze(0)   # (B, C, L)
            c = padpad(wave=wav_c, stride=48).unsqueeze(0)   # (B, C, L)
        else:
            y, y_phase, y_len = make_spectrum_torch(wave=wav_y,feature_type = self.args.feature, device=device)   # (B, C, L)
            c, c_phase, c_len = make_spectrum_torch(wave=wav_c,feature_type = self.args.feature, device=device)   # (B, C, L)
            y = y.transpose(1,2)
            c = c.transpose(1,2)

        # pdb.set_trace()
        if target == 'MAP':
            pred = self.model(y)       
            wave = pred
            
        elif target == 'MASK':      
            pred_irm = self.model(y)
            wave = y*pred_irm
           
        if self.args.loss == 'stoi':
            if self.args.feature== 'wave':
                rec_wav = wave.squeeze(0)[:,:wav_c.shape[1]]
                loss = stoi_loss(rec_wav, wav_c, wav_c.shape, reduction='mean')
                
            else:
                rec_wav = recons_spec_phase_torch(wave, phase=y_phase, length_wav=y_len, feature_type=self.args.feature,device=device)
                loss = stoi_loss(rec_wav, wav_c, wav_c.shape, reduction='mean')
                
        elif self.args.loss == 'l1_stoi':
            if self.args.feature== 'wave':
                rec_wav = wave.squeeze(0)[:,:wav_c.shape[1]]
                loss = 100*self.criterion(wave+epsilon, c+epsilon)+stoi_loss(rec_wav, wav_c, wav_c.shape, reduction='mean')
            else:
                rec_wav = recons_spec_phase_torch(wave, phase=y_phase, length_wav=y_len, feature_type=self.args.feature,device=device)
                loss = 100*self.criterion(wave+epsilon, c+epsilon)+stoi_loss(rec_wav, wav_c, wav_c.shape, reduction='mean')
        else:
            loss = self.criterion(wave+epsilon, c+epsilon)
        # pdb.set_trace() 
        self.train_loss += loss.item()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # self.scheduler.step(loss)


    def _train_epoch(self):
        self.train_loss = 0

        progress = tqdm(total=len(self.loader['train']), desc=f'Epoch {self.epoch} / Epoch {self.epochs} | train', unit='step')
        self.model.train()
         
        for y, c in self.loader['train']:
            self._train_step(y, c, self.target)
            progress.update(1)
            
        progress.close()
        self.train_loss /= len(self.loader['train'])
        
        
        print(f'{self.args.model}_train_loss:{self.train_loss}')

    def _val_step(self, in_y, in_c, target):
        
        device = self.device 
        wav_y, wav_c = in_y.transpose(1,2).squeeze(2).to(device), in_c.transpose(1,2).squeeze(2).to(device)   # (B, L)

        if self.args.feature== 'wave':
            y = padpad(wave=wav_y, stride=48).unsqueeze(0)   # (B, C, L)
            c = padpad(wave=wav_c, stride=48).unsqueeze(0)   # (B, C, L)
        else:
            y, y_phase, y_len = make_spectrum_torch(wave=wav_y,feature_type = self.args.feature, device=device)   # (B, C, L)
            c, c_phase, c_len = make_spectrum_torch(wave=wav_c,feature_type = self.args.feature, device=device)   # (B, C, L)
            y = y.transpose(1,2)
            c = c.transpose(1,2)

        if target == 'MAP':
            pred = self.model(y)       
            wave = pred
            
        elif target == 'MASK':      
            pred_irm = self.model(y)
            wave = y*pred_irm
           
        if self.args.loss == 'stoi':
            if self.args.feature== 'wave':
                rec_wav = wave.squeeze(0)[:,:wav_c.shape[1]]
                loss = stoi_loss(rec_wav, wav_c, wav_c.shape, reduction='mean')
            else:
                rec_wav = recons_spec_phase_torch(wave, phase=y_phase, length_wav=y_len, feature_type=self.args.feature,device=device)
                loss = stoi_loss(rec_wav, wav_c, wav_c.shape, reduction='mean')
                
        elif self.args.loss == 'l1_stoi':
            if self.args.feature== 'wave':
                rec_wav = wave.squeeze(0)[:,:wav_c.shape[1]]
                loss = 100*self.criterion(wave+epsilon, c+epsilon)+stoi_loss(rec_wav, wav_c, wav_c.shape, reduction='mean')
            else:
                rec_wav = recons_spec_phase_torch(wave, phase=y_phase, length_wav=y_len, feature_type=self.args.feature,device=device)
                loss = 100*self.criterion(wave+epsilon, c+epsilon)+stoi_loss(rec_wav, wav_c, wav_c.shape, reduction='mean')
        
        else:
            loss = self.criterion(wave+epsilon, c+epsilon)
        
        self.val_loss += loss.item()   
       
        # self.scheduler.step(loss)

    def _val_epoch(self):
        self.val_loss = 0
     
        progress = tqdm(total=len(self.loader['val']), desc=f'Epoch {self.epoch} / Epoch {self.epochs} | valid', unit='step')
        self.model.eval()

        for y, c in self.loader['val']:
            self._val_step(y, c, self.target)
            progress.update(1)

        progress.close()

        self.val_loss /= len(self.loader['val'])
        
        print(f'{self.args.model}_val_loss:{self.val_loss}')
       
        if self.best_loss > self.val_loss:
            
            print(f"Save model to '{self.model_path}'")
            self.save_checkpoint()
            self.best_loss = self.val_loss

            
    def write_score(self, test_file, clean_path, audio_path, target, args):
        
        self.model.eval()
        if args.task=='VCTK':           
            noisy, sr = torchaudio.load(test_file)
            clean, sr = torchaudio.load(os.path.join(clean_path, test_file.split('/')[-1]))
        elif args.task=='TMHINT':
            noisy, sr = torchaudio.load(test_file)
            rename = test_file.split('/')[-1].split('_')[-4]+'_'+test_file.split('/')[-1].split('_')[-3]+'_'+test_file.split('/')[-1].split('_')[-2]+'_'+test_file.split('/')[-1].split('_')[-1]
            clean, sr = torchaudio.load(os.path.join(clean_path, rename))  # (C, L)
        
        if args.feature== 'wave':
            y = padpad(wave=noisy, stride=48).unsqueeze(0).cuda()   # (B, C, L)
            c = padpad(wave=clean, stride=48).unsqueeze(0).cuda()   # (B, C, L)
        else:
            y, y_phase, y_len = make_spectrum_torch(wave=noisy.to(self.device),feature_type =args.feature, device=self.device)   # (B, C, L)
            c, c_phase, c_len = make_spectrum_torch(wave=clean.to(self.device),feature_type =args.feature, device=self.device)   # (B, C, L)
            y = y.transpose(1,2)
       
        if target == 'MAP':
            pred = self.model(y)       
            wave = pred

        elif target == 'MASK':
            pred_irm = self.model(y)
            wave = y*pred_irm

        wave = wave.transpose(1,2)
        
        if args.feature== 'wave':
            pred_clean = wave.squeeze(0)[:,:clean.shape[1]]
        else:   
            pred_clean = recons_spec_phase_torch(wave, phase=y_phase, length_wav=y_len, feature_type=self.args.feature,device=self.device)
        # pdb.set_trace()
        if self.save_results == 'True':
            out_a_path = os.path.join(audio_path,  f"{test_file.split('/')[-1].split('.')[0]+'.wav'}")
            check_folder(out_a_path)
            torchaudio.save(out_a_path, pred_clean.cpu(), sr, format='wav', encoding='PCM_S', bits_per_sample=16)
        
        n_pesq, n_stoi = cal_score(clean,noisy)
        s_pesq, s_stoi = cal_score(clean,pred_clean)
        
        wave_name = test_file.split('/')[-1].split('.')[0]
        with open(self.score_path, 'a') as f:
            f.write(f'{wave_name},{n_pesq},{s_pesq},{n_stoi},{s_stoi}\n')

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
        
        check_folder(self.score_path)
        if os.path.exists(self.score_path):
            os.remove(self.score_path)
        with open(self.score_path, 'a') as f:
            f.write('Filename,Noisy_PESQ,Pred_PESQ,Noisy_STOI,Pred_STOI\n')
                         
        for test_file in tqdm(test_folders):
            
            self.write_score(test_file, clean_path, audio_path, self.target, self.args)
        
        data = pd.read_csv(self.score_path)
        n_pesq_mean = data['Noisy_PESQ'].to_numpy().astype('float').mean()
        s_pesq_mean = data['Pred_PESQ'].to_numpy().astype('float').mean()
        n_stoi_mean = data['Noisy_STOI'].to_numpy().astype('float').mean()
        s_stoi_mean = data['Pred_STOI'].to_numpy().astype('float').mean()

        with open(self.score_path, 'a') as f:
            f.write(','.join(('Average',str(n_pesq_mean),str(s_pesq_mean),str(n_stoi_mean),str(s_stoi_mean)))+'\n')