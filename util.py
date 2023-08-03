import numpy as np
import scipy
import pdb
import torch, os
from torchmetrics.functional.audio.stoi import short_time_objective_intelligibility
from torchmetrics.functional.audio.pesq import perceptual_evaluation_speech_quality
epsilon = np.finfo(float).eps



def check_path(path):
    if not os.path.isdir(path): 
        os.makedirs(path)
        
def check_folder(path):
    path_n = '/'.join(path.split('/')[:-1])
    check_path(path_n)

def cal_score(clean,enhanced):
    clean = clean/abs(clean).max()
    enhanced = enhanced/abs(enhanced).max()
    s_stoi = short_time_objective_intelligibility(enhanced, clean, 16000).float()
    s_pesq = perceptual_evaluation_speech_quality(enhanced, clean, 16000, 'wb').float()
    
    return round(s_pesq.numpy()[0],5), round(s_stoi.numpy()[0],5)


def get_filepaths(directory,ftype='.wav'):
    file_paths = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            if filename.endswith(ftype):
                filepath = os.path.join(root, filename)
                file_paths.append(filepath)  # Add it to the list.

    return sorted(file_paths)

def make_spectrum(filename=None, y=None, is_slice=False, feature_type='log1p', mode=None, FRAMELENGTH=None,
                 SHIFT=None, _max=None, _min=None):
    if y is not None:
        y = y
    else:
        y, sr = librosa.load(filename, sr=16000)
        if sr != 16000:
            raise ValueError('Sampling rate is expected to be 16kHz!')
        if y.dtype == 'int16':
            y = np.float32(y/32767.)
        elif y.dtype !='float32':
            y = np.float32(y)

    ### Normalize waveform
    # y = y / np.max(abs(y)) / 2.

    D = librosa.stft(y,center=False, n_fft=512, hop_length=256,win_length=512,window=scipy.signal.hamming)
    utt_len = D.shape[-1]
    phase = np.exp(1j * np.angle(D))
    D = np.abs(D)

    ### Feature type
    if feature_type == 'log1p':
        Sxx = np.log1p(D)
    elif feature_type == 'lps':
        Sxx = np.log10(D**2)
    elif feature_type == 'lps+':
        Sxx = np.log10((D+1e-12)**2)
    else:
        Sxx = D

    if mode == 'mean_std':
        mean = np.mean(Sxx, axis=1).reshape(((hp.n_fft//2)+1, 1))
        std = np.std(Sxx, axis=1).reshape(((hp.n_fft//2)+1, 1))+1e-12
        Sxx = (Sxx-mean)/std  
    elif mode == 'minmax':
        Sxx = 2 * (Sxx - _min)/(_max - _min) - 1

    return Sxx, phase, len(y)

def recons_spec_phase(Sxx_r, phase, length_wav, feature_type='log1p'):
    if feature_type == 'log1p':
        Sxx_r = np.expm1(Sxx_r)
        if np.min(Sxx_r) < 0:
            print("Expm1 < 0 !!")
        # Sxx_r = np.clip(Sxx_r, a_min=0., a_max=None)
    elif feature_type == 'lps':
        Sxx_r = np.sqrt(10**(Sxx_r))

    R = np.multiply(Sxx_r , phase)
    
#     pdb.set_trace()
    result = librosa.istft(R,
                     center=False,
                     hop_length=256,
                     win_length=512,
                     window=scipy.signal.hamming,
                     length=length_wav)
    return result

def make_spectrum_torch(wave=None, feature_type='log1p', device=None):
    
    # pdb.set_trace()
    win = torch.hamming_window(window_length=512, device=device)
    com_stft = torch.stft(input=wave,n_fft=512, hop_length=256, win_length=512, window=win, center=False, return_complex=True)

    utt_len = com_stft.shape[-1]
    phase = torch.exp(1j * torch.angle(com_stft))
    D = torch.abs(com_stft)

    ### Feature type
    if feature_type == 'log1p':
        Sxx = torch.log1p(D)
    elif feature_type == 'lps':
        Sxx = torch.log10(D**2)
    elif feature_type == 'lps+':
        Sxx = torch.log10((D+1e-12)**2)
    else:
        Sxx = D

    return Sxx, phase, wave.shape[-1]

def recons_spec_phase_torch(Sxx_r, phase, length_wav, feature_type='log1p', device=None):
      
    if feature_type == 'log1p':
        Sxx_r = torch.expm1(Sxx_r)
        # if torch.min(Sxx_r) < 0:
            # print("Expm1 < 0 !!")
    elif feature_type == 'lps' or 'lps+':
        Sxx_r = torch.sqrt(10**(Sxx_r))

    R = torch.multiply(Sxx_r , phase)
    win = torch.hamming_window(window_length=512, device=device)
    result = torch.istft(R, n_fft=512, hop_length=256, win_length=512, window=win, center=False, length=length_wav, return_complex=False) 

    return result