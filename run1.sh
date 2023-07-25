export CUDA_VISIBLE_DEVICES='0'

python main.py --model BLSTM_01 --target MAP --batch_size 1 --epochs 50 \
               --loss mse --version None --lr 5e-4 --task VCTK --mode train --save_results False
               
python main.py --model transformerencoder --target MAP --batch_size 1 --epochs 100 \
               --loss l1 --version None --lr 1e-5 --task VCTK --mode train --save_results False
               
# python main.py --model BLSTM_02 --target IRM --batch_size 1 --epochs 50 \
#                --loss mse --version jin_0608 --lr 5e-4 --task VCTK

