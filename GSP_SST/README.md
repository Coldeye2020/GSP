
### Download dataset and unzip
You can download `Graph-SST2` and `Graph-SST5` in this [link](https://drive.google.com/drive/folders/1dt0aGMBvCEUYzaG00TYu1D03GPO7305z).
Then, put them into `.data/` and unzip. 


### Reproduce the result
```
# Reproduce Graph-SST results with distribution shift (OOD)
nohup python main.py --dataset Graph-SST --shot 5 > SST_ood_shot5.log 2>&1 &
nohup python main.py --dataset Graph-SST --shot 25 > SST_ood_shot5.log 2>&1 &

# Reproduce Graph-SST results with no distribution shift (I.I.D)
nohup python main.py --dataset Graph-SST --shot 5 > SST_iid_shot5.log 2>&1 &
nohup python main.py --dataset Graph-SST --shot 25 > SST_iid_shot25.log 2>&1 &
```