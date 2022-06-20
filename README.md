# Intern_SE

We are present the two task SE model with BLSTM, one is the Mapping task and other is the IRM task.
The Dataset is VCTK.

you can used run1.sh to training and testing the SE model.  
Or you can used the main.py to get results  

python main.py --model BLSTM_01 --target MAP --batch_size 1 --epochs 50 \  
               --loss mse --version jin_0608 --lr 5e-4 --task VCTK  
               
--model　　　　　　　　which model you used  
--target　　　　　　　　defined the training task MAP or IRM  
--batch_size　　　　　　training batch   
--epochs　　　　　　　　training epochs  
--loss　　　　　　　　　training loss with mse, l1, l1smooth, cosine  
--version　　　　　　　defined your version  
--lr　　　　　　　　　　training learn rate  
--task　　　　　　　　　defined your datasets  

Environment Setup  
python-----------3.6.13  
torch------------1.10.0  
librosa----------0.9.1  
numpy------------1.19.5  
scipy------------1.5.4  
tensorboardx-----2.5.1  
tqdm-------------4.64.0  
pandas-----------1.1.5  
pystoi-----------0.3.3  
pesq-------------0.0.4  
