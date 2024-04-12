# Reproduce the result
```
# Reproduce tox21 results with extreme distribution shift (topology & size)
nohup python main.py --dataset ogbg_moltox21 --shot 25 > moltox21_shot25.log 2>&1 &

# Reproduce sider results with extreme distribution shift (topology & size)
nohup python main.py --dataset ogbg_molsider --shot 25 > molsider_shot25.log 2>&1 &
```