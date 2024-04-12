import os
import pickle
import torch
import numpy as np
from torch_geometric.data import Batch, Data
from torch_geometric.utils import degree
from torch_geometric.loader import DataLoader
from ogb.graphproppred import PygGraphPropPredDataset
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_data_loaders(data_dir, dataset_name, data_config, splits, random_state, mutag_x=False, ood_flag=1):
    pretrain_task_idx = data_config["pretrain_task_idx"]
    finetune_task_idx = data_config["finetune_task_idx"]
    batch_size = data_config["batch_size"]
    fs_batch_size = data_config["fs_batch_size"]
    num_fs_samples = data_config["num_fs_samples"]
    multi_label = False

    assert dataset_name in ['ogbg_moltox21', 'ogbg_molsider',
                            'ogbg_molmuv', 'ogbg_moltoxcast']

    if 'ogbg' in dataset_name:
        dataset = PygGraphPropPredDataset(root=data_dir, name='-'.join(dataset_name.split('_')))
        split_idx = dataset.get_idx_split()
        print('[INFO] Using default splits!')
        split_idx = get_extrem_ood_split_idx(dataset, split_idx, dataset_name)
        loaders = get_loaders_and_test_set(batch_size, dataset=dataset, split_idx=split_idx, pretrain_task_idx=pretrain_task_idx, finetune_task_idx=finetune_task_idx)
        train_set = dataset[split_idx['train']]
        valid_set = dataset[split_idx['valid']]
        test_set = dataset[split_idx["test"]]
        set_seed(3)
        try:
            fs_set = torch.load(data_dir / f'{dataset_name}/few_shot/{num_fs_samples}.pt')
        except:
            print("Generating new fewshot set!")
            fs_set = get_fs_set(sampled_set=train_set, finetune_task_idx=finetune_task_idx, batch_size=fs_batch_size, num_fs_samples=num_fs_samples)
            if not os.path.exists(data_dir / f'{dataset_name}/few_shot/'):
                os.makedirs(data_dir / f'{dataset_name}/few_shot/')
            torch.save(fs_set, data_dir / f'{dataset_name}/few_shot/{num_fs_samples}.pt')  
        fs_loader = DataLoader(fs_set, batch_size=fs_batch_size, shuffle=True)
        loaders["fs"] = fs_loader
        
        if ood_flag == 0:
            randpermed_train_idx = torch.randperm(len(train_set))
            # Construct iid valid set
            iid_valid_set = randpermed_train_idx[:int(len(train_set) * 0.2)]
            # Construct iid test set
            iid_test_set = randpermed_train_idx[int(len(train_set) * 0.2):]
            add_iid_loaders(loaders, 1024, dataset=train_set, split_idx={"iid_valid": iid_valid_set, "iid_test": iid_test_set}, pretrain_task_idx=pretrain_task_idx, finetune_task_idx=finetune_task_idx)


    x_dim = test_set[0].x.shape[1]
    edge_attr_dim = 0 if test_set[0].edge_attr is None else test_set[0].edge_attr.shape[1]
    if isinstance(test_set, list):
        num_class = Batch.from_data_list(test_set).y.unique().shape[0]
    elif test_set.y.shape[-1] == 1 or len(test_set.y.shape) == 1:
        num_class = test_set.y.unique().shape[0]
    else:
        num_class = test_set.y.shape[-1]
        multi_label = True
    try: 
        deg = torch.load(data_dir / f'{dataset_name}/deg.pt')
    except:
        print('[INFO] Calculating degree...')
        batched_train_set = Batch.from_data_list(train_set)
        d = degree(batched_train_set.edge_index[1], num_nodes=batched_train_set.num_nodes, dtype=torch.long)
        deg = torch.bincount(d, minlength=10)
        torch.save(deg, data_dir / f'{dataset_name}/deg.pt')

    aux_info = {'deg': deg, 'multi_label': multi_label}
    return loaders, train_set, fs_set, valid_set, test_set, x_dim, edge_attr_dim, num_class, aux_info

def get_extrem_ood_split_idx(dataset, split_idx, dataset_name):
    if dataset_name == "ogbg_moltox21":
        split_size_h = 26
        split_size_l = 14
    elif dataset_name == "ogbg_molsider":
        split_size_h = 32
        split_size_l = 15
    extrem_split_idx = {"train": [], "valid": [], "test": []}
    for i in split_idx["train"]:
        if(dataset[i].num_nodes <= split_size_h):
            extrem_split_idx["train"].append(i)
    for i in split_idx["valid"]:
        if(dataset[i].num_nodes >= split_size_l):
            extrem_split_idx["valid"].append(i)
    for i in split_idx["test"]:
        if(dataset[i].num_nodes >= split_size_l):
            extrem_split_idx["test"].append(i)
    return extrem_split_idx

def get_random_split_idx(dataset, splits, random_state=None, mutag_x=False):
    if random_state is not None:
        np.random.seed(random_state)

    print('[INFO] Randomly split dataset!')
    idx = np.arange(len(dataset))
    np.random.shuffle(idx)

    if not mutag_x:
        n_train, n_valid = int(splits['train'] * len(idx)), int(splits['valid'] * len(idx))
        train_idx = idx[:n_train]
        valid_idx = idx[n_train:n_train+n_valid]
        test_idx = idx[n_train+n_valid:]
    else:
        print('[INFO] mutag_x is True!')
        n_train = int(splits['train'] * len(idx))
        train_idx, valid_idx = idx[:n_train], idx[n_train:]
        test_idx = [i for i in range(len(dataset)) if (dataset[i].y.squeeze() == 0 and dataset[i].edge_label.sum() > 0)]
    return {'train': train_idx, 'valid': valid_idx, 'test': test_idx}


def get_loaders_and_test_set(batch_size, dataset=None, split_idx=None, pretrain_task_idx=None, finetune_task_idx=None):
    num_tasks = dataset.y.shape[-1]
    # assert len(pretrain_task_idx) + len(finetune_task_idx) == num_tasks
    processed_data_list = []
    pretrain_task_idx = torch.LongTensor(pretrain_task_idx)
    finetune_task_idx = torch.LongTensor(finetune_task_idx)
    for i in dataset:
        i.py = i.y[:, pretrain_task_idx] # pretrain_task_labels
        i.task_mask = torch.ones(1, len(finetune_task_idx), dtype=torch.bool) # Can be removed
        i.fy = i.y[:, finetune_task_idx] # finetune_task_labels
        processed_data_list.append(i)

    train_loader = DataLoader([processed_data_list[i] for i in split_idx["train"]], batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader([processed_data_list[i] for i in split_idx["valid"]], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader([processed_data_list[i] for i in split_idx["test"]], batch_size=batch_size, shuffle=False)
    return {'train': train_loader, 'valid': valid_loader, 'test': test_loader}

def add_iid_loaders(loaders, batch_size, dataset=None, split_idx=None, pretrain_task_idx=None, finetune_task_idx=None):
    processed_data_list = []
    for i in dataset:
        i.fy = i.y[:, finetune_task_idx] # finetune_task_labels
        processed_data_list.append(i)
    loaders["iid_valid"] = DataLoader([processed_data_list[i] for i in split_idx["iid_valid"]], batch_size=batch_size, shuffle=False)
    loaders["iid_test"] = DataLoader([processed_data_list[i] for i in split_idx["iid_test"]], batch_size=batch_size, shuffle=False)
    return loaders


def get_fs_set(sampled_set, finetune_task_idx, batch_size, num_fs_samples):
    num_tasks = sampled_set.y.shape[1]
    # assert num_pretrain_tasks + num_finetune_tasks == num_tasks
    train_pos_num = torch.zeros(num_tasks, dtype=torch.int)
    train_neg_num = torch.zeros(num_tasks, dtype=torch.int)
    # To check if num_fs_samples is valid
    for i in range(num_tasks):
        if train_pos_num[i] == 0:
            train_pos_num[i] = torch.sum(sampled_set.y[:, i] == 1)
            train_neg_num[i] = torch.sum(sampled_set.y[:, i] == 0)
    # assert num_fs_samples <= min(torch.min(train_pos_num), torch.min(train_neg_num))
    # Sample fs_samples with equal number of positive and negative samples in each task
    sampled_dict = {i: [] for i in finetune_task_idx}
    for i in finetune_task_idx:
        pos_mask = (sampled_set.y[:, i] == 1)
        neg_mask = (sampled_set.y[:, i] == 0)
        pos_idx = torch.nonzero(pos_mask).squeeze()
        neg_idx = torch.nonzero(neg_mask).squeeze()
        pos_idx = pos_idx[torch.randperm(pos_idx.shape[0])]
        neg_idx = neg_idx[torch.randperm(neg_idx.shape[0])]
        if num_fs_samples < min(train_pos_num[i], train_neg_num[i]):
            pos_idx = pos_idx[:num_fs_samples]
            neg_idx = neg_idx[:num_fs_samples]
        else:
            max_sample = min(train_pos_num[i], train_neg_num[i])
            pos_idx = pos_idx[:max_sample]
            neg_idx = neg_idx[:max_sample]
        sampled_dict[i].append(pos_idx)
        sampled_dict[i].append(neg_idx)
    sampled_dict = {i:torch.cat(sampled_dict[i], dim=0) for i in sampled_dict.keys()}
    # Construct fs set
    few_shot_list = []
    num_finetune_tasks = len(finetune_task_idx)
    one_hot = torch.eye(num_finetune_tasks, dtype=torch.bool)
    for i, task_idx in enumerate(finetune_task_idx):
        for j in sampled_dict[task_idx]:
            data = sampled_set[j]
            data.task_mask = one_hot[i:i+1]
            data.fy = data.y[:, finetune_task_idx]
            few_shot_list.append(data)
    return few_shot_list