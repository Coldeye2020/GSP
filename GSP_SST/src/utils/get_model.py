import torch
import torch.nn as nn
import torch.nn.functional as F
from models import GIN, PNA
from torch_geometric.nn import InstanceNorm


def get_model(x_dim, edge_attr_dim, num_class, multi_label, model_config, device, node_encoder=False):
    if model_config['model_name'] == 'GIN':
        model = GIN(x_dim, edge_attr_dim, num_class, multi_label, model_config, node_encoder)
    elif model_config['model_name'] == 'PNA':
        model = PNA(x_dim, edge_attr_dim, num_class, multi_label, model_config, node_encoder)
    else:
        raise ValueError('[ERROR] Unknown model name!')
    return model.to(device)

def create_prompts(num_tasks, hidden_dim, device):
    prompts = nn.ModuleDict()
    for task_idx in range(num_tasks):
        prompts[str(task_idx)] = graph_prompt(hidden_dim)
    return prompts.to(device)

class graph_prompt(nn.Module):
    def __init__(self, hidden_dim):
        super(graph_prompt, self).__init__()
        self.prompt = nn.Parameter(torch.zeros(1, hidden_dim))
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.prompt)
    def reassign_prompt(self, prompt):
        self.prompt.data = prompt.data
    def forward(self, emb):
        return emb * self.prompt.sigmoid()

def get_predictors(encoder_name, hidden_size, num_tasks, device):
    predictors = nn.ModuleDict()
    for task_idx in range(num_tasks):
        if encoder_name == 'GIN':
            predictors[str(task_idx)] = nn.Sequential(nn.Linear(hidden_size, 1))
        elif encoder_name == 'PNA':
            predictors[str(task_idx)] = nn.Sequential(nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(),
                                    nn.Linear(hidden_size // 2, hidden_size // 4), nn.ReLU(),
                                    nn.Linear(hidden_size // 4, 1))
    return predictors.to(device)


class Criterion(nn.Module):
    def __init__(self, num_class, multi_label):
        super(Criterion, self).__init__()
        self.num_class = num_class
        self.multi_label = multi_label
        # print(f'[INFO] Using multi_label: {self.multi_label}, num_labels: {self.num_class}')

    def forward(self, logits, targets, task_mask=None):
        # TODO: check whether Hadamard product is OK
        is_labeled = targets == targets  # mask for labeled data
        if task_mask is not None:
            is_labeled = is_labeled * task_mask
        loss = F.binary_cross_entropy_with_logits(logits[is_labeled], targets[is_labeled].float())
        return loss


def get_preds(logits, multi_label):
    if multi_label:
        preds = (logits.sigmoid() > 0.5).float()
    elif logits.shape[1] > 1:  # multi-class
        preds = logits.argmax(dim=1).float()
    else:  # binary
        preds = (logits.sigmoid() > 0.5).float()
    return preds


class BatchSequential(nn.Sequential):
    def forward(self, inputs, batch):
        for module in self._modules.values():
            if isinstance(module, (InstanceNorm)):
                inputs = module(inputs, batch)
            else:
                inputs = module(inputs)
        return inputs


class MLP(BatchSequential):
    def __init__(self, channels, dropout, bias=True):
        m = []
        for i in range(1, len(channels)):
            m.append(nn.Linear(channels[i - 1], channels[i], bias))

            if i < len(channels) - 1:
                # TODO: modify
                # m.append(InstanceNorm(channels[i]))
                m.append(nn.ReLU())
                m.append(nn.Dropout(dropout))

        super(MLP, self).__init__(*m)
