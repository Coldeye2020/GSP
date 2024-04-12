import os
import pickle
import glob
import torch
import numpy as np
import json
from torch_geometric.data import Batch, Data
from torch_geometric.utils import degree
from torch_geometric.loader import DataLoader
from ogb.graphproppred import PygGraphPropPredDataset
import random
import os.path as osp
from torch_geometric.data import Data, InMemoryDataset


def undirected_graph(data):
    data.edge_index = torch.cat([torch.stack([data.edge_index[1], data.edge_index[0]], dim=0),
                                 data.edge_index], dim=1)
    return data

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
                            'ogbg_molmuv', 'ogbg_moltoxcast', 'Graph-SST']

    # if 'ogbg' in dataset_name:
    #     dataset = PygGraphPropPredDataset(root=data_dir, name='-'.join(dataset_name.split('_')))
    #     split_idx = dataset.get_idx_split()
    #     print('[INFO] Using default splits!')
    #     loaders = get_loaders_and_test_set(batch_size, dataset=dataset, split_idx=split_idx, pretrain_task_idx=pretrain_task_idx, finetune_task_idx=finetune_task_idx)
    #     train_set = dataset[split_idx['train']]
    #     valid_set = dataset[split_idx['valid']]
    #     test_set = dataset[split_idx["test"]]
    #     set_seed(random_state)
    #     try:
    #         fs_set = torch.load(data_dir / f'{dataset_name}/few_shot/{random_state}/{num_fs_samples}.pt')
    #     except:
    #         print("Generating new fewshot set!")
    #         fs_set = get_fs_set(sampled_set=train_set, finetune_task_idx=finetune_task_idx, batch_size=fs_batch_size, num_fs_samples=num_fs_samples)
    #         if not os.path.exists(data_dir / f'{dataset_name}/few_shot/{random_state}'):
    #             os.makedirs(data_dir / f'{dataset_name}/few_shot/{random_state}')
    #         torch.save(fs_set, data_dir / f'{dataset_name}/few_shot/{random_state}/{num_fs_samples}.pt')  
    #     fs_loader = DataLoader(fs_set, batch_size=fs_batch_size, shuffle=True)
    #     loaders["fs"] = fs_loader
        
    #     if ood_flag == 0:
    #         randpermed_train_idx = torch.randperm(len(train_set))
    #         # Construct iid valid set
    #         iid_valid_set = randpermed_train_idx[:int(len(train_set) * 0.2)]
    #         # Construct iid test set
    #         iid_test_set = randpermed_train_idx[int(len(train_set) * 0.2):]
    #         add_iid_loaders(loaders, 1024, dataset=train_set, split_idx={"iid_valid": iid_valid_set, "iid_test": iid_test_set}, pretrain_task_idx=pretrain_task_idx, finetune_task_idx=finetune_task_idx)

    if dataset_name == 'Graph-SST':
        dataset_SST2 = SentiGraphDataset(root=data_dir, name=f'{dataset_name}2', transform=None)  # Target task
        dataset_SST5 = SentiGraphDataset(root=data_dir, name=f'{dataset_name}5', transform=None)  # Source task
        degree_bias_flag = ood_flag == 1
        loaders_sst2, loaders_sst5, train_sst2 = get_SST_loaders(
            dataset_SST2, dataset_SST5, 
            batch_size=batch_size, 
            degree_bias=degree_bias_flag, 
            pretrain_task_idx=pretrain_task_idx, 
            finetune_task_idx=finetune_task_idx
        )
    
        few_shot_type = 'ood' if ood_flag else 'iid'
        fs_dir = data_dir / f'{dataset_name}/few_shot/{few_shot_type}'
        fs_file = fs_dir / f'{num_fs_samples}.pt'
        fs_dir.mkdir(parents=True, exist_ok=True)
        try:
            fs_set = torch.load(fs_file)
        except FileNotFoundError:
            print("Generating new few-shot set!")
            fs_set = get_fs_set(
                sampled_set=train_sst2, 
                finetune_task_idx=finetune_task_idx, 
                batch_size=fs_batch_size, 
                num_fs_samples=num_fs_samples
            )
            torch.save(fs_set, fs_file)
    
        fs_loader = DataLoader(fs_set, batch_size=fs_batch_size, shuffle=True)
            
        loaders_sst2["fs"] = fs_loader
    
    x_dim = train_sst2[0].x.shape[1]
    edge_attr_dim = 0 if train_sst2[0].edge_attr is None else train_sst2[0].edge_attr.shape[1]
    num_class = train_sst2.y.shape[-1]
    multi_label = True
    # try: 
    #     deg = torch.load(data_dir / f'{dataset_name}/deg.pt')
    # except:
    #     print('[INFO] Calculating degree...')
    #     batched_train_set = Batch.from_data_list(train_set)
    #     d = degree(batched_train_set.edge_index[1], num_nodes=batched_train_set.num_nodes, dtype=torch.long)
    #     deg = torch.bincount(d, minlength=10)
    #     torch.save(deg, data_dir / f'{dataset_name}/deg.pt')

    aux_info = {'deg': 0, 'multi_label': multi_label}
    return loaders_sst2, loaders_sst5, None, x_dim, edge_attr_dim, num_class, aux_info


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

def get_loaders_and_test_set(batch_size, dataset=None, split_idx=None, pretrain_task_idx=None, finetune_task_idx=None):
    num_tasks = dataset.y.shape[-1]
    # assert len(pretrain_task_idx) + len(finetune_task_idx) == num_tasks
    processed_data_list = []
    pretrain_task_idx = torch.LongTensor(pretrain_task_idx)
    finetune_task_idx = torch.LongTensor(finetune_task_idx)
    for i in dataset:
        i.py = i.y[:, pretrain_task_idx] # pretrain_task_labels
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


class SentiGraphDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=undirected_graph):
        self.name = name
        super(SentiGraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices, self.supplement = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return ['node_features', 'node_indicator', 'sentence_tokens', 'edge_index',
                'graph_labels', 'split_indices']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        # Read data into huge `Data` list.
        self.data, self.slices, self.supplement \
              = read_sentigraph_data(self.raw_dir, self.name, num_classes=5 if self.name == 'Graph-SST5' else 2)

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)
        torch.save((self.data, self.slices, self.supplement), self.processed_paths[0])
        

def read_sentigraph_data(folder: str, prefix: str, num_classes):
    txt_files = glob.glob(os.path.join(folder, "{}_*.txt".format(prefix)))
    json_files = glob.glob(os.path.join(folder, "{}_*.json".format(prefix)))
    txt_names = [f.split(os.sep)[-1][len(prefix) + 1:-4] for f in txt_files]
    json_names = [f.split(os.sep)[-1][len(prefix) + 1:-5] for f in json_files]
    names = txt_names + json_names

    with open(os.path.join(folder, prefix+"_node_features.pkl"), 'rb') as f:
        x: np.array = pickle.load(f)
    x: torch.FloatTensor = torch.from_numpy(x)
    edge_index: np.array = read_file(folder, prefix, 'edge_index')
    edge_index: torch.tensor = torch.tensor(edge_index, dtype=torch.long).T
    batch: np.array = read_file(folder, prefix, 'node_indicator') - 1     # from zero
    y: np.array = read_file(folder, prefix, 'graph_labels')
    y: torch.tensor = torch.tensor(y, dtype=torch.long)
    # TODO: convert 5 classfication label into 7 binary classification labels
    if num_classes == 5:
        y: torch.tensor = torch.nn.functional.one_hot(y, num_classes=num_classes)
    else:
        y: torch.tensor = y.unsqueeze(-1)
    if num_classes == 5:
        y = torch.cat([y, torch.zeros((y.size(0), 1))], dim=1)
    else:
        y = torch.cat([torch.zeros((y.size(0), 5)), y], dim=1)
        
    edge_attr = torch.ones((edge_index.size(1), 1)).float()
    name = torch.tensor(range(y.size(0)))
    supplement = dict()
    if 'split_indices' in names:
        split_indices: np.array = read_file(folder, prefix, 'split_indices')
        split_indices = torch.tensor(split_indices, dtype=torch.long)
        supplement['split_indices'] = split_indices
    if 'sentence_tokens' in names:
        with open(os.path.join(folder, prefix + '_sentence_tokens.json')) as f:
            sentence_tokens: dict = json.load(f)
        supplement['sentence_tokens'] = sentence_tokens
    data = Data(name=name, x=x, edge_index=edge_index, y=y, sentence_tokens=list(sentence_tokens.values()))
    data, slices = split(data, batch)

    return data, slices, supplement


    # i-th contains elements from slice[i] to slice[i+1]
    node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
    node_slice = torch.cat([torch.tensor([0]), node_slice])
    row, _ = data.edge_index
    edge_slice = torch.cumsum(torch.from_numpy(np.bincount(batch[row])), 0)
    edge_slice = torch.cat([torch.tensor([0]), edge_slice])

    # Edge indices should start at zero for every graph.
    data.edge_index -= node_slice[batch[row]].unsqueeze(0)
    data.__num_nodes__ = np.bincount(batch).tolist()

    slices = dict()
    slices['x'] = node_slice
    slices['edge_index'] = edge_slice
    slices['y'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)
    slices['sentence_tokens'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)
    return data, slices

def read_file(folder, prefix, name):
    file_path = osp.join(folder, prefix + f'_{name}.txt')
    return np.genfromtxt(file_path, dtype=np.int64)


def get_SST_loaders(dataset_SST2, dataset_SST5, batch_size, degree_bias=True, data_split_ratio=[0.8, 0.1, 0.1], pretrain_task_idx=None, finetune_task_idx=None):
    """
    Args:
        dataset:
        batch_size: int
        random_split_flag: bool
        data_split_ratio: list, training, validation and testing ratio
        seed: random seed to split the dataset randomly
    Returns:
        a dictionary of training, validation, and testing dataLoader
    """

    if degree_bias:
        train_sst2, test_sst2 = [], []
        train_sst2_list, test_sst2_list = [], []
        train_sst5, test_sst5 = [], []
        for idx, g in enumerate(dataset_SST2):
            if g.num_edges <= 2: continue
            degree = float(g.num_edges) / g.num_nodes
            if degree >= 1.83:
                g.fy = g.y[:, finetune_task_idx]
                train_sst2.append(idx)
                train_sst2_list.append(g)
            elif degree < 1.83:
                g.fy = g.y[:, finetune_task_idx]
                test_sst2_list.append(g)
        valid_sst2 = train_sst2[:int(len(train_sst2) * 0.1)]
        valid_sst2 = dataset_SST2[valid_sst2]
        valid_sst2_list = train_sst2_list[:int(len(train_sst2) * 0.1)]
        train_sst2 = train_sst2[int(len(train_sst2) * 0.1):]
        train_sst2 = dataset_SST2[train_sst2]
        train_sst2_list = train_sst2_list[int(len(train_sst2) * 0.1):]
        for g in dataset_SST5:
            if g.num_edges <= 2: continue
            degree = float(g.num_edges) / g.num_nodes
            if degree >= 1.83:
                g.py = g.y[:, pretrain_task_idx]
                train_sst5.append(g)
            elif degree < 1.83:
                g.py = g.y[:, pretrain_task_idx]
                test_sst5.append(g)
        valid_sst5 = train_sst5[int(len(train_sst5) * 0.1):]
        train_sst5 = train_sst5[:int(len(train_sst5) * 0.1)]

        loader_sst2 = dict()
        loader_sst2['train'] = DataLoader(train_sst2_list, batch_size=batch_size, shuffle=True)
        loader_sst2['valid'] = DataLoader(valid_sst2_list, batch_size=batch_size, shuffle=False)
        loader_sst2['test'] = DataLoader(test_sst2_list, batch_size=batch_size, shuffle=False)

        loader_sst5 = dict()
        loader_sst5['train'] = DataLoader(train_sst5, batch_size=batch_size, shuffle=True)
        loader_sst5['valid'] = DataLoader(valid_sst5, batch_size=batch_size, shuffle=False)
        loader_sst5['test'] = DataLoader(test_sst5, batch_size=batch_size, shuffle=False)
    
        return loader_sst2, loader_sst5, train_sst2

    # random split
    else:
        num_sst2 = len(dataset_SST2)
        num_sst5 = len(dataset_SST5)
        idx_sst2 = list(range(num_sst2))
        idx_sst5 = list(range(num_sst5))
        random.shuffle(idx_sst2)
        random.shuffle(idx_sst5)
        train_sst2 = idx_sst2[:int(num_sst2 * data_split_ratio[0])]
        valid_sst2 = idx_sst2[int(num_sst2 * data_split_ratio[0]):int(num_sst2 * (data_split_ratio[0] + data_split_ratio[1]))]
        test_sst2 = idx_sst2[int(num_sst2 * (data_split_ratio[0] + data_split_ratio[1])):]
        dataset_SST2_list = []
        for data in dataset_SST2:
            data.fy = data.y[:, finetune_task_idx]
            dataset_SST2_list.append(data)
        
        train_sst5 = idx_sst5[:int(num_sst5 * data_split_ratio[0])]
        valid_sst5 = idx_sst5[int(num_sst5 * data_split_ratio[0]):int(num_sst5 * (data_split_ratio[0] + data_split_ratio[1]))]
        test_sst5 = idx_sst5[int(num_sst5 * data_split_ratio[0]):]
        dataset_SST5_list = []
        for data in dataset_SST5:
            data.py = data.y[:, pretrain_task_idx]
            dataset_SST5_list.append(data)
        
        loader_sst2 = dict()
        loader_sst2["train"] = DataLoader([dataset_SST2_list[i] for i in train_sst2], batch_size=batch_size, shuffle=True)
        loader_sst2["iid_valid"] = DataLoader([dataset_SST2_list[i] for i in valid_sst2], batch_size=batch_size, shuffle=False)
        loader_sst2["iid_test"] = DataLoader([dataset_SST2_list[i] for i in test_sst2], batch_size=batch_size, shuffle=False)
        
        loader_sst5 = dict()
        loader_sst5["train"] = DataLoader([dataset_SST5_list[i] for i in train_sst5], batch_size=batch_size, shuffle=True)
        loader_sst5["valid"] = DataLoader([dataset_SST5_list[i] for i in valid_sst5], batch_size=batch_size, shuffle=False)
        loader_sst5["test"] = DataLoader([dataset_SST5_list[i] for i in test_sst5], batch_size=batch_size, shuffle=False)
        
        return loader_sst2, loader_sst5, dataset_SST2[train_sst2]
        
        
        
       
       
        
    