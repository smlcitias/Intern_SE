import os, argparse, torch, random, sys
from Trainer import Trainer
from Load_model import Load_model, Load_data
from util import check_folder
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
import pandas as pd
import pdb

# fix random
SEED = 999
random.seed(SEED)
torch.manual_seed(SEED)
cudnn.deterministic = True

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, default='V1')
    parser.add_argument('--mode', type=str, default='train') #transformerencoder
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--batch_size', type=int, default=8)  
    parser.add_argument('--lr', type=float, default=0.00005)
    parser.add_argument('--loss', type=str, default='l1')
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--model', type=str, default='BLSTM') 
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--target', type=str, default='MAP') #'MAP' or 'MASK'
    parser.add_argument('--feature', type=str, default='log1p') 
    parser.add_argument('--task', type=str, default='VCTK') 
    parser.add_argument('--resume' , action='store_true')
    parser.add_argument('--retrain', action='store_true')
    parser.add_argument('--save_results', type=str, default='False')
    parser.add_argument('--re_epochs', type=int, default=300)
    parser.add_argument('--checkpoint', type=str, default=None)

    args = parser.parse_args()
    return args

def get_path(args):
    
    checkpoint_path = f'./checkpoint/'\
    f'{args.model}_{args.version}_{args.task}_{args.target}_epochs{args.epochs}_{args.optim}' \
    f'_{args.loss}_{args.feature}_batch{args.batch_size}_lr{args.lr}.pth.tar'
    
    model_path = f'./save_model/'\
    f'{args.model}_{args.version}_{args.task}_{args.target}_epochs{args.epochs}_{args.optim}' \
    f'_{args.loss}_{args.feature}_batch{args.batch_size}_lr{args.lr}.pth.tar'
    
    score_path = f'./scores/'\
    f'{args.model}_{args.version}_{args.task}_{args.target}_epochs{args.epochs}_{args.optim}' \
    f'_{args.loss}_{args.feature}_batch{args.batch_size}_lr{args.lr}.csv'    
    
    
    return checkpoint_path,model_path,score_path

if __name__ == '__main__':
    # get current path
    cwd = os.path.dirname(os.path.abspath(__file__))
    print(cwd)
    print(SEED)
        
    # get parameter
    args = get_args()
    
    print('model name =', args.model)
    print('target mode =', args.target)
    print('version =', args.version)
    print('Lr = ', args.lr)
    
    # data path
    if args.task=='VCTK':    
        Train_path = {
        'noisy':'/mnt/Intern_SE/Data/noisy_trainset_wav',
        'clean':'/mnt/Intern_SE/Data/clean_trainset_wav'
        } 
        Test_path = {
        'noisy':'/mnt/Intern_SE/Data/noisy_testset_wav',
        'clean':'/mnt/Intern_SE/Data/clean_testset_wav'
        }
    elif args.task=='TMHINTQI_V2':    
        Train_path = {
        'noisy':'/mnt/TMHINT_QI_V2re/training/noisy',
        'clean':'/mnt/TMHINT_QI_V2re/training/clean'
        } 
        Test_path = {
        'noisy':'/mnt/TMHINT_QI_V2re/testing/noisy',
        'clean':'/mnt/TMHINT_QI_V2re/testing/clean'
        }
        
    Output_path = {
    'audio':f'./result/'\
        f'{args.model}_{args.version}_{args.task}_{args.target}_epochs{args.epochs}_{args.optim}' \
        f'_{args.loss}_{args.feature}_batch{args.batch_size}_lr{args.lr}'
    }
    
    # declair path
    checkpoint_path,model_path,score_path = get_path(args)

    # tensorboard
    writer = SummaryWriter(f'./logs/'\
                           f'{args.model}_{args.version}_{args.task}_{args.target}_epochs{args.epochs}_{args.optim}' \
                           f'_{args.loss}_{args.feature}_batch{args.batch_size}_lr{args.lr}')
    

    exec (f"from models.{args.model.split('_')[0]} import {args.model} as model")
    model     = model()
    model, epoch, best_loss, optimizer, scheduler, criterion, device = Load_model(args,model,checkpoint_path, model_path)
    
    loader = Load_data(args, Train_path)
    if args.retrain:
        args.epochs = args.re_epochs 
        checkpoint_path, model_path, score_path = get_path(args)
    
    
    Trainer = Trainer(model, args.version, args.epochs, epoch, best_loss, optimizer,scheduler, 
                      criterion, device, loader, Test_path, writer, model_path, score_path, args, Output_path, args.save_results, args.target)
    try:
        if args.mode == 'train':
            Trainer.train()
        Trainer.test()
        
    except KeyboardInterrupt:
        state_dict = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_loss': best_loss
            }
        check_folder(checkpoint_path)
        torch.save(state_dict, checkpoint_path)
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)