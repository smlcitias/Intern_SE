# Intern_SE

We are present the two task SE model with BLSTM, one is the Mapping task and other is the IRM task.
The Dataset is VCTK.

you can used run1.sh to training and testing the SE model.
Or you can used the main.py to get results

python main.py --model BLSTM_01 --target MAP --batch_size 1 --epochs 50 \
               --loss mse --version jin_0608 --lr 5e-4 --task VCTK
               
--model              which model you used \n
--target             defined the training task MAP or IRM \n
--batch_size         training batch \n
--epochs             training epochs \n
--loss               training loss with mse, l1, l1smooth, cosine \n
--version            defined your version \n
--lr                 training learn rate \n
--task               defined your datasets \n
