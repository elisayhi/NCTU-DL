import os

models = ['EEG', 'deep']
acti_func = ['ReLU', 'LeakyReLU', 'ELU'] 
lrs = [1e-3]#, 1e-4]
bsizes = [64, 128, 256, 512, 2056]

acti = 'ELU'
model = 'EEG'
for lr in lrs:
    for bs in bsizes:
        os.system(f'python hw2.py -m {model} -a {acti} -l {lr} -e 3000 -b {bs} -s 0 -gpu 3')
