# Reproduce the result
```
# Reproduce results in OOD
nohup python main.py --dataset ogbg_molsider --shot 5 > molsider_shot5.log --ood_flag 1 2>&1 &
nohup python main.py --dataset ogbg_molsider --shot 25 > molsider_shot25.log  --ood_flag 1 2>&1 &
nohup python main.py --dataset ogbg_moltox21 --shot 5 > moltox21_shot5.log  --ood_flag 1 2>&1 &
nohup python main.py --dataset ogbg_moltox21 --shot 25 > moltox21_shot25.log  --ood_flag 1 2>&1 &

# Reproduce results in ID 
nohup python main.py --dataset ogbg_molsider --shot 5 > molsider_shot5.log --ood_flag 0 2>&1 &
nohup python main.py --dataset ogbg_molsider --shot 25 > molsider_shot25.log  --ood_flag 0 2>&1 &
nohup python main.py --dataset ogbg_moltox21 --shot 5 > moltox21_shot5.log  --ood_flag 0 2>&1 &
nohup python main.py --dataset ogbg_moltox21 --shot 25 > moltox21_shot25.log  --ood_flag 0 2>&1 &
```