export CUDA_VISIBLE_DEVICES='3'

python main.py --model BLSTM_01 --target MAP --batch_size 1 --epochs 50 \
               --loss mse --version jin_0608 --lr 5e-4 --task VCTK
               
# python main.py --model BLSTM_02 --target IRM --batch_size 1 --epochs 50 \
#                --loss mse --version jin_0608 --lr 5e-4 --task VCTK

