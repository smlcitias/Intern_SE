# Intern_SE_torch

We show two task SE models with BLSTM, one for the Mapping task and the other for the Masking task. The dataset is VCTK.  
The Dataset is [VCTK_sp28](https://drive.google.com/file/d/1sePGXayGyJkqSFaCPwzI8GshZ5OFL-kJ/view?usp=share_link)

you can used run1.sh to training and testing the SE model.  
Or you can used the main.py to get results  

python main.py --model BLSTM_01 --target MAP --batch_size 1 --epochs 50 \  
               --loss mse --version name --lr 5e-4 --task VCTK  
               
--model　　　　　　　　  which model you used  
--target　　　　　　　　defined the training task MAP or MASK  
--batch_size　　　　　　training batch   
--epochs　　　　　　　　training epochs  
--loss　　　　　　　　　training loss with mse, l1, l1smooth, cosine ,CE, stoi   
--version　　　　　　　 defined your version  
--lr　　　　　　　　　　training learn rate  
--task　　　　　　　　　defined your datasets  

## Environment Setup  
python-----------3.11.4  
torch------------2.0.1  
torchaudio-------2.0.2  
torchmetrics-----1.0.1  
speechbrain------0.5.15  
scikit-learn-----1.3.0  
numpy------------1.25.2  
scipy------------1.11.1  
tensorboardx-----2.6.2  
tqdm-------------4.65.0  
pandas-----------2.0.3  
pystoi-----------0.3.3  
pesq-------------0.0.4  
