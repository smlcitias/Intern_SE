export CUDA_VISIBLE_DEVICES='0'

python main.py --model BLSTM_02 --target MASK --batch_size 1 --epochs 50 --feature log1p \
               --loss l1 --version jin --lr 5e-4 --task VCTK --mode train --save_results False 
               
python main.py --model transformerencoder --target MASK --batch_size 1 --epochs 50 --feature log1p  \
               --loss l1 --version jin --lr 5e-5 --task VCTK --mode train --save_results False

python main.py --model BLSTM_02 --target MASK --batch_size 1 --epochs 50 --feature log1p \
               --loss stoi --version jin --lr 5e-4 --task VCTK --mode train --save_results False 
               
python main.py --model transformerencoder --target MASK --batch_size 1 --epochs 50 --feature log1p  \
               --loss stoi --version jin --lr 5e-5 --task VCTK --mode train --save_results False
               