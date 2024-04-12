import yaml
import scipy
import numpy as np
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from datetime import datetime
import time
import os
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_sparse import transpose
from torch_geometric.loader import DataLoader
from torch_geometric.utils import subgraph, is_undirected
from ogb.graphproppred import Evaluator
from sklearn.metrics import roc_auc_score
from rdkit import Chem

from utils import Criterion, MLP, visualize_a_graph, save_checkpoint, load_checkpoint, get_preds, get_lr, set_seed, process_data
from utils import get_local_config_name, get_model, get_predictors, get_data_loaders, get_fs_set, write_stat_from_metric_dicts, reorder_like, init_metric_dict, create_prompts, graph_prompt
import warnings
# warnings.filterwarnings("ignore")

class GSP(nn.Module):

    def __init__(self, node_encoder, graph_encoder, predictors, extractor, prompts, global_prompt, optimizer, scheduler, device, pretrain_model_dir, dataset_name, multi_label, random_state,
                 method_config, shared_config, model_config, data_config, args):
        super().__init__()
        self.node_encoder = node_encoder
        self.graph_encoder = graph_encoder
        self.extractor = extractor
        self.predictors = predictors
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.device = device
        self.dataset_name = dataset_name
        self.random_state = random_state
        
        self.method_name = method_config['method_name']

        self.learn_edge_att = shared_config['learn_edge_att']

        self.epochs = method_config['epochs']
        self.pred_loss_coef = method_config['pred_loss_coef']
        self.info_loss_coef = args.lambda_2

        self.fix_r = method_config.get('fix_r', None)
        self.decay_interval = method_config.get('decay_interval', None)
        self.decay_r = method_config.get('decay_r', None)
        self.final_r = method_config.get('final_r', 0.1)
        self.init_r = method_config.get('init_r', 0.9)

        self.multi_label = multi_label
        self.pretrain_task_idx = data_config['pretrain_task_idx']
        self.finetune_task_idx = data_config['finetune_task_idx']
        self.num_pretrain_tasks = len(data_config['pretrain_task_idx'])
        self.num_finetune_tasks = len(data_config['finetune_task_idx'])
        self.criterion = Criterion(self.num_pretrain_tasks, multi_label)
        
        # Encoder setting:
        self.encoder_name = model_config['model_name']
        self.hidden_size = model_config['hidden_size']
        

        # configuration:
        self.shared_config = shared_config
        
        # save dir:
        self.pretrain_model_dir = pretrain_model_dir
        # self.finetune_model_dir = None
        
        # Prompt
        self.prompts = prompts
        self.prompts_weight = None
        self.pretrain_global_prompt_weight = args.pretrain_global_prompt_weight
        self.global_prompt = global_prompt
        self.prompted_graph_mixup = args.prompted_graph_mixup


    def __loss__(self, att, clf_logits, clf_labels, epoch, task_mask=None):
        pred_loss = self.criterion(clf_logits, clf_labels, task_mask)

        r = self.fix_r if self.fix_r else self.get_r(self.decay_interval, self.decay_r, epoch, final_r=self.final_r, init_r=self.init_r)
        info_loss = (att * torch.log(att/r + 1e-6) + (1-att) * torch.log((1-att)/(1-r+1e-6) + 1e-6)).mean()
        
        pred_loss = pred_loss * self.pred_loss_coef
        info_loss = info_loss * self.info_loss_coef
        loss = pred_loss + info_loss
        loss_dict = {'loss': loss.item(), 'pred': pred_loss.item(), 'info': info_loss.item()}
        return loss, loss_dict

    def forward_pass(self, data, epoch, training, fine_tuning=False):
        emb = self.node_encoder.get_emb(data.x, data.edge_index, batch=data.batch, edge_attr=data.edge_attr)
        if not fine_tuning:
            prompted_embs_dict = dict()
            clf_logits_dict = {}
            _atts_dict = {} # for possible loss calculation
            
            if self.global_prompt is not None:
                global_prompted_emb = self.global_prompt(emb)
            else:
                global_prompted_emb = emb
            global_att_log_logits = self.extractor(global_prompted_emb, data.edge_index, data.batch)
            global_att = self.sampling(global_att_log_logits, epoch, training)
            global_edge_att = self.lift_node_att_to_edge_att(global_att, data.edge_index)
            global_embedding = self.graph_encoder(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr, edge_atten=global_edge_att)
            for i in self.pretrain_task_idx:
                prompted_embs_dict[str(i)] = self.prompts[str(i)](emb)
            if self.prompted_graph_mixup == "n_emb":
                for i in self.pretrain_task_idx:
                    # mix global and task-specific in node embedding
                    prompted_emb = (1 / (1 + self.pretrain_global_prompt_weight))  * prompted_embs_dict[str(i)] + (self.pretrain_global_prompt_weight / (1 + self.pretrain_global_prompt_weight)) * global_prompted_emb
                    att_log_logits = self.extractor(prompted_emb, data.edge_index, data.batch)
                    att = self.sampling(att_log_logits, epoch, training)
                    edge_att = self.lift_node_att_to_edge_att(att, data.edge_index)
                    _atts_dict[str(i)] = att
                    clf_embedding = self.graph_encoder(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr, edge_atten=edge_att)
                    clf_logits = self.predictors[str(i)](clf_embedding)
                    clf_logits_dict[str(i)] = clf_logits
            elif self.prompted_graph_mixup == "att":
                for i in self.pretrain_task_idx:
                    att_log_logits = self.extractor(prompted_embs_dict[str(i)], data.edge_index, data.batch)
                    att = self.sampling(att_log_logits, epoch, training)
                    # mix global and task-specific in node att
                    att = (1 / (1 + self.pretrain_global_prompt_weight)) * att + (self.pretrain_global_prompt_weight / (1 + self.pretrain_global_prompt_weight)) * global_att
                    edge_att = self.lift_node_att_to_edge_att(att, data.edge_index)
                    _atts_dict[str(i)] = att
                    clf_embedding = self.graph_encoder(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr, edge_atten=edge_att)
                    clf_logits = self.predictors[str(i)](clf_embedding)
                    clf_logits_dict[str(i)] = clf_logits
            elif self.prompted_graph_mixup == "edge_att":
                for i in self.pretrain_task_idx:
                    att_log_logits = self.extractor(prompted_embs_dict[str(i)], data.edge_index, data.batch)
                    att = self.sampling(att_log_logits, epoch, training)
                    att = (1 / (1 + self.pretrain_global_prompt_weight)) * att + (self.pretrain_global_prompt_weight / (1 + self.pretrain_global_prompt_weight)) * global_att
                    edge_att = self.lift_node_att_to_edge_att(att, data.edge_index)
                    # mix global and task-specific in edge att
                    edge_att = (1 / (1 + self.pretrain_global_prompt_weight)) * edge_att + (self.pretrain_global_prompt_weight / (1 + self.pretrain_global_prompt_weight)) * global_edge_att
                    _atts_dict[str(i)] = att
                    clf_embedding = self.graph_encoder(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr, edge_atten=edge_att)
                    clf_logits = self.predictors[str(i)](clf_embedding)
                    clf_logits_dict[str(i)] = clf_logits
            elif self.prompted_graph_mixup == "g_emb":
                for i in self.pretrain_task_idx:
                    att_log_logits = self.extractor(prompted_embs_dict[str(i)], data.edge_index, data.batch)
                    att = self.sampling(att_log_logits, epoch, training)
                    att = (1 / (1 + self.pretrain_global_prompt_weight)) * att + (self.pretrain_global_prompt_weight / (1 + self.pretrain_global_prompt_weight)) * global_att
                    edge_att = self.lift_node_att_to_edge_att(att, data.edge_index)
                    _atts_dict[str(i)] = att
                    clf_embedding = self.graph_encoder(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr, edge_atten=edge_att)
                    # mix global and task-specific in graph emb
                    clf_embedding = (1 / (1 + self.pretrain_global_prompt_weight)) * clf_embedding + (self.pretrain_global_prompt_weight / (1 + self.pretrain_global_prompt_weight)) * global_embedding
                    clf_logits = self.predictors[str(i)](clf_embedding)
                    clf_logits_dict[str(i)] = clf_logits
            clf_logits = torch.stack([clf_logits_dict[str(i)] for i in self.pretrain_task_idx]).permute(1, 0, 2).squeeze(-1)
            att = torch.stack([_atts_dict[str(i)] for i in self.pretrain_task_idx]).permute(1, 0, 2).squeeze(-1)
        else:
            prompted_embs_dict = {}
            _atts_dict = {} # for possible loss calculation
            atts_dict ={}
            edge_atts_dict = {}
            g_embs_dict = {}
            clf_logits_dict = {}
            if self.global_prompt is not None:
                prompted_embs_dict["global"] = self.global_prompt(emb)
            else:
                prompted_embs_dict["global"] = emb
            
            for i in self.finetune_task_idx:
                prompted_embs_dict[str(i)] = self.prompts[str(i)](emb)


            if self.prompted_graph_mixup == "n_emb":
                for i in self.finetune_task_idx:
                    prompted_emb = (1 / (1 + self.pretrain_global_prompt_weight))  * prompted_embs_dict[str(i)] + (self.pretrain_global_prompt_weight / (1 + self.pretrain_global_prompt_weight)) * prompted_embs_dict["global"]
                    att_log_logits = self.extractor(prompted_emb, data.edge_index, data.batch)
                    att = self.sampling(att_log_logits, epoch, training)
                    edge_att = self.lift_node_att_to_edge_att(att, data.edge_index)
                    _atts_dict[str(i)] = att
                    clf_embedding = self.graph_encoder(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr, edge_atten=edge_att)
                    clf_logits = self.predictors[str(i)](clf_embedding)
                    clf_logits_dict[str(i)] = clf_logits
            elif self.prompted_graph_mixup == "att":
                for n in prompted_embs_dict.keys():
                    att_log_logits = self.extractor(prompted_embs_dict[n], data.edge_index, data.batch)
                    atts_dict[n] = self.sampling(att_log_logits, epoch, training)
                for i in self.finetune_task_idx:
                    att = (1 / (1 + self.pretrain_global_prompt_weight)) * atts_dict[str(i)] + (self.pretrain_global_prompt_weight / (1 + self.pretrain_global_prompt_weight)) * atts_dict["global"]
                    edge_att = self.lift_node_att_to_edge_att(att, data.edge_index)
                    _atts_dict[str(i)] = att
                    clf_embedding = self.graph_encoder(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr, edge_atten=edge_att)
                    clf_logits = self.predictors[str(i)](clf_embedding)
                    clf_logits_dict[str(i)] = clf_logits
            elif self.prompted_graph_mixup == "edge_att":
                for n in prompted_embs_dict.keys():
                    att_log_logits = self.extractor(prompted_embs_dict[n], data.edge_index, data.batch)
                    att = self.sampling(att_log_logits, epoch, training)
                    edge_atts_dict[n] = self.lift_node_att_to_edge_att(att, data.edge_index)
                for i in self.finetune_task_idx:
                    edge_att = (1 / (1 + self.pretrain_global_prompt_weight)) * edge_atts_dict[str(i)] + (self.pretrain_global_prompt_weight / (1 + self.pretrain_global_prompt_weight)) * edge_atts_dict["global"]
                    _atts_dict[str(i)] = edge_att
                    clf_embedding = self.graph_encoder(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr, edge_atten=edge_att)
                    clf_logits = self.predictors[str(i)](clf_embedding)
                    clf_logits_dict[str(i)] = clf_logits
            elif self.prompted_graph_mixup == "g_emb":
                for n in prompted_embs_dict.keys():
                    _atts_dict[n] = att
                    att_log_logits = self.extractor(prompted_embs_dict[n], data.edge_index, data.batch)
                    att = self.sampling(att_log_logits, epoch, training)
                    edge_atts_dict[n] = self.lift_node_att_to_edge_att(att, data.edge_index)
                    g_embs_dict[n] = self.graph_encoder(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr, edge_atten=edge_att)
                for i in self.finetune_task_idx:
                    clf_embedding = (1 / (1 + self.pretrain_global_prompt_weight)) * g_embs_dict[str(i)] + (self.pretrain_global_prompt_weight / (1 + self.pretrain_global_prompt_weight)) * g_embs_dict["global"]
                    clf_logits = self.predictors[str(i)](clf_embedding)
                    clf_logits_dict[str(i)] = clf_logits
                
            clf_logits = torch.stack([clf_logits_dict[str(i)] for i in self.finetune_task_idx]).permute(1, 0, 2).squeeze(-1)
            att = torch.stack([_atts_dict[n] for n in _atts_dict.keys()]).permute(1, 0, 2).squeeze(-1)
            
        if fine_tuning:
            if training:
                loss, loss_dict = self.__loss__(att, clf_logits, data.fy, epoch)
            else:
                loss, loss_dict = self.__loss__(att, clf_logits, data.fy, epoch)
        else:
            loss, loss_dict = self.__loss__(att, clf_logits, data.py, epoch)
        return edge_att, loss, loss_dict, clf_logits

    @torch.no_grad()
    def eval_one_batch(self, data, epoch, fine_tuning=False):
        self.extractor.eval()
        self.node_encoder.eval()
        self.graph_encoder.eval()
        self.predictors.eval()
        self.prompts.eval()
        if fine_tuning:
            for param in self.prompts_weight.values():
                param.requires_grad = False
                
        att, loss, loss_dict, clf_logits = self.forward_pass(data, epoch, training=False, fine_tuning=fine_tuning)
        return att.cpu().reshape(-1), loss_dict, clf_logits.cpu()

    def train_one_batch(self, data, epoch, fine_tuning=False):
        self.extractor.train()
        self.node_encoder.train()
        self.graph_encoder.train()
        self.predictors.train()
        self.prompts.train()
        if fine_tuning:
            for param in self.prompts_weight.values():
                param.requires_grad = True

        att, loss, loss_dict, clf_logits = self.forward_pass(data, epoch, training=True, fine_tuning=fine_tuning)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return att.cpu().reshape(-1), loss_dict, clf_logits.cpu()

    def run_one_epoch(self, data_loader, epoch, phase, use_edge_attr, fine_tuning=False):
        loader_len = len(data_loader)
        run_one_batch = self.train_one_batch if phase == 'train' else self.eval_one_batch
        phase = 'test ' if phase == 'test' else phase  # align tqdm desc bar
        training = True if phase == 'train' else False

        all_loss_dict = {}
        all_att, all_clf_labels, all_clf_logits = ([] for i in range(3))
        for idx, data in enumerate(data_loader):
            data = process_data(data, use_edge_attr)
            att, loss_dict, clf_logits = run_one_batch(data.to(self.device), epoch, fine_tuning)
            if fine_tuning:
                labels = data.fy.cpu()
            else:
                labels = data.py.cpu()

            for k, v in loss_dict.items():
                all_loss_dict[k] = all_loss_dict.get(k, 0) + v

            all_att.append(att)
                
            all_clf_labels.append(labels)
            all_clf_logits.append(clf_logits)

            if idx == loader_len - 1:
                all_att = torch.cat(all_att),
                all_clf_labels, all_clf_logits = torch.cat(all_clf_labels), torch.cat(all_clf_logits)

                for k, v in all_loss_dict.items():
                    all_loss_dict[k] = v / loader_len
                desc, clf_acc, clf_roc, avg_loss = self.log_epoch(epoch, phase, all_loss_dict, all_att, all_clf_labels, all_clf_logits, batch=False, fine_tuning=fine_tuning)
                print(desc)
        return clf_acc, clf_roc, avg_loss

    def train(self, loaders, test_set, metric_dict, use_edge_attr):
        start = time.time()
        for epoch in range(self.epochs):
            train_res = self.run_one_epoch(loaders['train'], epoch, 'train', use_edge_attr)
            valid_res = self.run_one_epoch(loaders['valid'], epoch, 'valid', use_edge_attr)
            test_res = self.run_one_epoch(loaders['test'], epoch, 'test', use_edge_attr)

            assert len(train_res) == 3
            main_metric_idx = 1 if 'ogb' in self.dataset_name else 0  # clf_roc or clf_acc
            if self.scheduler is not None:
                self.scheduler.step(valid_res[main_metric_idx])

            r = self.fix_r if self.fix_r else self.get_r(self.decay_interval, self.decay_r, epoch, final_r=self.final_r, init_r=self.init_r)
            if (r == self.final_r or self.fix_r) and epoch > 10 and ((valid_res[main_metric_idx] > metric_dict['metric/best_clf_valid'])
                                                                     or (valid_res[main_metric_idx] == metric_dict['metric/best_clf_valid']
                                                                         and valid_res[2] < metric_dict['metric/best_clf_valid_loss'])):

                metric_dict = {'metric/best_clf_epoch': epoch, 'metric/best_clf_valid_loss': valid_res[2],
                               'metric/best_clf_train': train_res[main_metric_idx], 'metric/best_clf_valid': valid_res[main_metric_idx], 'metric/best_clf_test': test_res[main_metric_idx]}
                torch.save(self.extractor.state_dict(), self.pretrain_model_dir / 'best_extractor.pt')
                torch.save(self.prompts.state_dict(), self.pretrain_model_dir / 'best_prompts.pt')
                torch.save(self.node_encoder.state_dict(), self.pretrain_model_dir / 'best_node_encoder.pt')
                torch.save(self.graph_encoder.state_dict(), self.pretrain_model_dir / 'best_graph_encoder.pt')
                torch.save(self.predictors.state_dict(), self.pretrain_model_dir / 'best_predictors.pt')
                torch.save(self.global_prompt.state_dict(), self.pretrain_model_dir / 'best_global_prompt.pt')


            if epoch == self.epochs - 1:
                print("Loading best epoch...")
                self.extractor.load_state_dict(torch.load(self.pretrain_model_dir / 'best_extractor.pt'))
                self.node_encoder.load_state_dict(torch.load(self.pretrain_model_dir / 'best_node_encoder.pt'))
                self.graph_encoder.load_state_dict(torch.load(self.pretrain_model_dir / 'best_graph_encoder.pt'))
                self.prompts.load_state_dict(torch.load(self.pretrain_model_dir / 'best_prompts.pt'))
                self.predictors.load_state_dict(torch.load(self.pretrain_model_dir / 'best_predictors.pt'))
                self.global_prompt.load_state_dict(torch.load(self.pretrain_model_dir / 'best_global_prompt.pt'))

            print(f'[Seed {self.random_state}, Pretrain Epoch: {epoch}]: Best Epoch: {metric_dict["metric/best_clf_epoch"]}, '
                  f'Best Val Pred ACC/ROC: {metric_dict["metric/best_clf_valid"]:.3f}, Best Test Pred ACC/ROC: {metric_dict["metric/best_clf_test"]:.3f}')
            
            if (epoch + 1) % 10 == 0:
                time_consumption = time.time() - start
                print("Time Consumption: {:.4f}s".format(time_consumption))
                start = time.time()
            print('====================================')
            print('====================================')

        return metric_dict

    def log_epoch(self, epoch, phase, loss_dict, att, clf_labels, clf_logits, batch, fine_tuning=False):
        desc = f'[Seed {self.random_state}, {"F" if fine_tuning else "P"} Epoch: {epoch}]: gsp_{phase}........., ' if batch else f'[Seed {self.random_state}, Epoch: {epoch}]: gsp_{phase} finished, '

        eval_desc, clf_acc, clf_roc = self.get_eval_score(epoch, phase, att, clf_labels, clf_logits, batch)
        desc += eval_desc
        return desc, clf_acc, clf_roc, loss_dict['pred']

    def get_eval_score(self, epoch, phase, att, clf_labels, clf_logits, batch):
        clf_preds = get_preds(clf_logits, self.multi_label)
        clf_acc = 0 if self.multi_label else (clf_preds == clf_labels).sum().item() / clf_labels.shape[0]
        if batch:
            return f'clf_acc: {clf_acc:.3f}', None, None, None, None
        clf_roc = 0
        if 'ogb' in self.dataset_name:
            evaluator = Evaluator(name="ogbg-moltox21") # The name of dataset is not important
            evaluator.num_tasks = clf_logits.shape[-1]
            clf_roc = evaluator.eval({'y_pred': clf_logits, 'y_true': clf_labels})['rocauc']

        desc = f'clf_acc: {clf_acc:.3f}, clf_roc: {clf_roc:.3f}'
        return desc, clf_acc, clf_roc
    
    def get_r(self, decay_interval, decay_r, current_epoch, init_r=0.9, final_r=0.5):
        r = init_r - current_epoch // decay_interval * decay_r
        if r < final_r:
            r = final_r
        return r

    def sampling(self, att_log_logits, epoch, training):
        att = self.concrete_sample(att_log_logits, temp=1, training=training)
        return att

    @staticmethod
    def lift_node_att_to_edge_att(node_att, edge_index):
        src_lifted_att = node_att[edge_index[0]]
        dst_lifted_att = node_att[edge_index[1]]
        edge_att = src_lifted_att * dst_lifted_att
        return edge_att

    @staticmethod
    def concrete_sample(att_log_logit, temp, training):
        if training:
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            att_bern = ((att_log_logit + random_noise) / temp).sigmoid()
        else:
            att_bern = (att_log_logit).sigmoid()
        return att_bern
   
    def few_shot_tuning(self, data_config, finetune_config, loaders, test_set, metric_dict, use_edge_attr, ood_flag):
        # read finetuning configuration 
        self.epochs = finetune_config['epochs']

        # Create Prompts weight
        self.prompts_weight = nn.ParameterDict()
        for i in self.finetune_task_idx:
            self.prompts_weight[str(i)] = nn.Parameter(torch.zeros(1, self.num_pretrain_tasks + 1))
        self.prompts_weight.to(self.device)
        # New criterion
        self.criterion = Criterion(self.num_finetune_tasks, self.multi_label)
        
        # Finetuning model save dir
        # self.finetune_model_dir = finetune_model_dir

        # Optimizer
        lr = finetune_config['lr']
        wd = finetune_config['wd']
        # TODO: Decide which parameters to be optimized in finetuning stage
        # self.optimizer = torch.optim.Adam(list(self.predictor.parameters()) + list(self.prompts_weight.parameters()) + list(self.extractor.parameters()) + list(self.node_encoder.parameters()) + list(self.graph_encoder.parameters()), lr=lr, weight_decay=wd)
        # self.optimizer = torch.optim.Adam(list(self.prompts_weight.parameters()) + list(self.predictors.parameters()), lr=lr, weight_decay=wd)
        self.optimizer = torch.optim.Adam(list(self.predictors.parameters()) + list(self.prompts.parameters()), lr=lr, weight_decay=wd)   
        # self.optimizer = torch.optim.Adam(list(self.predictors.parameters()), lr=lr, weight_decay=wd)
        # self.optimizer = torch.optim.Adam(list(self.prompts_weight.parameters()), lr=lr, weight_decay=wd)
        start = time.time()
        for epoch in range(self.epochs):
            if ood_flag == 1:
                train_res = self.run_one_epoch(loaders['fs'], epoch, 'train', use_edge_attr, fine_tuning=True)
                valid_res = self.run_one_epoch(loaders['valid'], epoch, 'valid', use_edge_attr, fine_tuning=True)
                test_res = self.run_one_epoch(loaders['test'], epoch, 'test', use_edge_attr, fine_tuning=True)
            elif ood_flag == 0:
                train_res = self.run_one_epoch(loaders['fs'], epoch, 'train', use_edge_attr, fine_tuning=True)
                valid_res = self.run_one_epoch(loaders['iid_valid'], epoch, 'valid', use_edge_attr, fine_tuning=True)

            main_metric_idx = 1 if 'ogb' in self.dataset_name else 0  # clf_roc or clf_acc
            if ((valid_res[main_metric_idx] > metric_dict['metric/best_clf_valid']) or (valid_res[main_metric_idx] == metric_dict['metric/best_clf_valid'] and valid_res[2] < metric_dict['metric/best_clf_valid_loss'])):
                if ood_flag == 0:
                    test_res = self.run_one_epoch(loaders['iid_test'], epoch, 'test', use_edge_attr, fine_tuning=True)
                metric_dict = {'metric/best_clf_epoch': epoch, 'metric/best_clf_valid_loss': valid_res[2],
                               'metric/best_clf_train': train_res[main_metric_idx], 'metric/best_clf_valid': valid_res[main_metric_idx], 'metric/best_clf_test': test_res[main_metric_idx]}
                               
            print(f'[Seed {self.random_state}, Finetune Epoch: {epoch}]: Best Epoch: {metric_dict["metric/best_clf_epoch"]}, '
                  f'Best Val Pred ACC/ROC: {metric_dict["metric/best_clf_valid"]:.3f}, Best Test Pred ACC/ROC: {metric_dict["metric/best_clf_test"]:.3f}')
            if (epoch + 1) % 10 == 0:
                time_consumption = time.time() - start
                print("Time Consumption: {:.4f}s".format(time_consumption))
                start = time.time()
            print('====================================')
        return metric_dict


class ExtractorMLP(nn.Module):
    def __init__(self, hidden_size, shared_config):
        super().__init__()
        self.learn_edge_att = shared_config['learn_edge_att']
        dropout_p = shared_config['extractor_dropout_p']

        if self.learn_edge_att:
            self.feature_extractor = MLP([hidden_size * 2, hidden_size * 4, hidden_size, 1], dropout=dropout_p)
        else:
            self.feature_extractor = MLP([hidden_size * 1, hidden_size * 2, hidden_size, 1], dropout=dropout_p)

    def forward(self, emb, edge_index, batch):
        if self.learn_edge_att:
            col, row = edge_index
            f1, f2 = emb[col], emb[row]
            f12 = torch.cat([f1, f2], dim=-1)
            att_log_logits = self.feature_extractor(f12, batch[col])
        else:
            att_log_logits = self.feature_extractor(emb, batch)
        return att_log_logits

    

def pretrain_gsp_one_seed(gsp, loaders_info, local_config, data_dir, model_name, dataset_name, method_name, device, random_state):
    print('====================================')
    print('====================================')
    print(f'[INFO] Using device: {device}')
    print(f'[INFO] Using random_state: {random_state}')
    print(f'[INFO] Using dataset: {dataset_name}')
    print(f'[INFO] Using model: {model_name}')

    set_seed(random_state)
    
    model_config = local_config['model_config']
    data_config = local_config['data_config']
    method_config = local_config[f'{method_name}_config']
    shared_config = local_config['shared_config']
    assert model_config['model_name'] == model_name
    assert method_config['method_name'] == method_name

    loaders, train_set, fs_set, valid_set, test_set, x_dim, edge_attr_dim, num_tasks, aux_info = loaders_info
   
    print('====================================')
    print('[INFO] Pretraining GSP...')
    metric_dict = deepcopy(init_metric_dict)
    metric_dict = gsp.train(loaders, test_set, metric_dict, model_config.get('use_edge_attr', True))
    
    
    finetune_init_dict = {"gsp_graph_encoder": gsp.graph_encoder.state_dict(), 
                          "gsp_node_encoder": gsp.node_encoder.state_dict(),
                          "gsp_extractor": gsp.extractor.state_dict(),
                          "gsp_predictors": gsp.predictors.state_dict(),
                          "gsp_prompts": gsp.prompts.state_dict(),
                         "loaders": loaders,
                         "test_set": test_set,
                        }
    return None, metric_dict, finetune_init_dict

def finetune_gsp_one_seed(gsp, finetune_init_dict, local_config, data_dir, model_name, dataset_name, method_name, device, random_state, ood_flag):
    set_seed(random_state)
    loaders = finetune_init_dict["loaders"]
    test_set = finetune_init_dict["test_set"]
    finetune_config = local_config['finetune_config']
    model_config = local_config['model_config']
    data_config = local_config['data_config']
    print('====================================')
    print('[INFO] Finetuning GSP...{} time'.format(random_state))
    metric_dict = deepcopy(init_metric_dict) # TODO: finetuneing metric dict
    metric_dict = gsp.few_shot_tuning(data_config, finetune_config, loaders, test_set, metric_dict, model_config.get('use_edge_attr', True), ood_flag)
    return metric_dict



def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train GSP')
    parser.add_argument('--dataset', type=str, help='dataset used', default="ogbg_moltox21")
    parser.add_argument('--backbone', type=str, help='backbone model used', default="GIN")
    parser.add_argument('--cuda', type=int, help='cuda device id, -1 for cpu', default=1)
    parser.add_argument('--global_prompt', type=bool, help="whether to use global prompt", default=True)
    parser.add_argument('--pretrain_global_prompt_weight', type=float, help="weight of global prompt in pretraining", default=1.0)
    parser.add_argument('--prompted_graph_mixup', type=str, help="whether to use prompted graph mixup", default="att", choices=["n_emb", "att", "edge_att", "g_emb"])
    parser.add_argument('--shot', type=int, help="number of samples per task", default=1)
    parser.add_argument('--ood_flag', type=int, help="whether to use iid(0) or ood(1) test set", default=1)
    parser.add_argument('--pid', type=int, help="pretrain id", default=0)
    parser.add_argument('--seed', type=int, help="seed", default=0)
    parser.add_argument('--lambda_1', type=float, default=1.0)
    parser.add_argument('--lambda_2', type=float, default=1.0)
    args = parser.parse_args()
    args.pretrain_global_prompt_weight = args.lambda_1
    dataset_name = args.dataset
    model_name = args.backbone
    cuda_id = args.cuda

    torch.set_num_threads(5)
    config_dir = Path('./configs')
    method_name = 'GSP'

    print('====================================')
    print('====================================')
    print(f'[INFO] Running {method_name} on {dataset_name} with {model_name}')
    print(f"OOD flag: {args.ood_flag}")
    print('====================================')

    global_config = yaml.safe_load((config_dir / 'global_config.yml').open('r'))
    local_config_name = get_local_config_name(model_name, dataset_name)
    local_config = yaml.safe_load((config_dir / local_config_name).open('r'))

    data_dir = Path(global_config['data_dir'])
    pretrain_dir = Path(global_config['pretrain_dir'])
    finetune_dir = Path(global_config["finetune_dir"])
    num_seeds = 5

    dtime = datetime.now().strftime("%m_%d_%Y-%H_%M_%S")
    device = torch.device(f'cuda:{cuda_id}' if cuda_id >= 0 else 'cpu')

    # Ensure which tasks are used for pretrain and finetune, respectively
    # The following task splits are based on the former works: PAR[https://arxiv.org/pdf/2107.07994.pdf] and Meta-MGNN[https://arxiv.org/pdf/2102.07916.pdf]
    if dataset_name == 'ogbg_moltox21':
        local_config["data_config"]["pretrain_task_idx"] = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        local_config["data_config"]["finetune_task_idx"] = [9, 10, 11]
        local_config["data_config"]["batch_size"] = 128
        local_config["data_config"]["num_fs_samples"] = args.shot
        local_config["data_config"]["fs_batch_size"] = args.shot
        local_config['finetune_config']["epochs"] = 40
    elif dataset_name == 'ogbg_molsider':
        local_config["data_config"]["pretrain_task_idx"] = [i for i in range(21)]
        local_config["data_config"]["finetune_task_idx"] = [21, 22, 23, 24, 25, 26] 
        local_config["data_config"]["batch_size"] = 128
        local_config["data_config"]["num_fs_samples"] = args.shot
        local_config["data_config"]["fs_batch_size"] = args.shot
        local_config['finetune_config']["epochs"] = 40
    elif dataset_name == 'ogbg_molmuv':
        local_config["data_config"]["pretrain_task_idx"] = [i for i in range(12)]
        local_config["data_config"]["finetune_task_idx"] = [12, 13, 14, 15, 16] 
        local_config["data_config"]["batch_size"] = 50
        local_config["data_config"]["num_fs_samples"] = args.shot
        local_config["data_config"]["fs_batch_size"] = args.shot
        local_config['finetune_config']["epochs"] = 60
    elif dataset_name == 'ogbg_moltoxcast':
        toxcast_drop_tasks = [343, 348, 349, 352, 354, 355, 356, 357, 358, 360, 361, 362, 364, 367, 368, 369, 370, 371, 372,
                          373, 374, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 387, 388, 389, 390, 391, 392, 393,
                          394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 406, 408, 409, 410, 411, 412, 413, 414,
                          415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 426, 428, 429, 430, 431, 432, 433, 434, 435,
                          436, 437, 438, 439, 440, 442, 443, 444, 445, 446, 447, 449, 450, 451, 452, 453, 474, 475, 477,
                          480, 481, 482, 483]
        local_config["data_config"]["pretrain_task_idx"] = [i for i in list(range(450)) if i not in toxcast_drop_tasks]
        local_config["data_config"]["finetune_task_idx"] = [i for i in list(range(450, 617)) if i not in toxcast_drop_tasks]
        local_config["data_config"]["batch_size"] = 50
        local_config["data_config"]["num_fs_samples"] = args.shot
        local_config["data_config"]["fs_batch_size"] = args.shot
        local_config['finetune_config']["epochs"] = 60
        
    # convert configuration
    model_config = local_config['model_config']
    data_config = local_config['data_config']
    method_config = local_config[f'{method_name}_config']
    shared_config = local_config['shared_config']
    assert model_config['model_name'] == model_name
    assert method_config['method_name'] == method_name
    batch_size, splits = data_config['batch_size'], data_config.get('splits', None)
    
    # create dataloader:
    print('====================================')
    print('====================================')
    loaders_info = get_data_loaders(data_dir, dataset_name, data_config, splits, args.seed, data_config.get('mutag_x', False), args.ood_flag)
    loaders, train_set, fs_set, valid_set, test_set, x_dim, edge_attr_dim, num_tasks, aux_info = loaders_info
    model_config['deg'] = aux_info['deg']
    
    # create network
    node_encoder = get_model(x_dim, edge_attr_dim, num_tasks, aux_info['multi_label'], model_config, device, node_encoder=True)
    graph_encoder = get_model(x_dim, edge_attr_dim, num_tasks, aux_info['multi_label'], model_config, device)
    predictors = get_predictors(model_config["model_name"], model_config["hidden_size"], num_tasks, device)
    extractor = ExtractorMLP(model_config['prompt_hidden_size'], shared_config).to(device)
    prompts = create_prompts(num_tasks, model_config["prompt_hidden_size"], device) # task_specific_prompt
    if args.global_prompt:
        global_prompt = graph_prompt(model_config["prompt_hidden_size"]).to(device)
    else:
        global_prompt = None
        
    # create optimizer
    lr, wd = method_config['lr'], method_config.get('weight_decay', 0)
    optimizer = torch.optim.Adam(list(extractor.parameters()) + list(prompts.parameters()) + list(global_prompt.parameters()) +list(node_encoder.parameters()) + list(graph_encoder.parameters()) + list(predictors.parameters()), lr=lr, weight_decay=wd)
    scheduler_config = method_config.get('scheduler', {})
    scheduler = None if scheduler_config == {} else ReduceLROnPlateau(optimizer, mode='max', **scheduler_config)
    metric_dict_list = []
    
    time_consumption = 0
    for random_state in range(num_seeds):
        pretrain_model_dir = pretrain_dir / dataset_name / model_name
        if not osp.exists(pretrain_model_dir):
            os.makedirs(pretrain_model_dir)
        hparam_dict = {**model_config, **data_config}
        hparam_dict = {k: str(v) if isinstance(v, (dict, list)) else v for k, v in hparam_dict.items()}
        metric_dict = deepcopy(init_metric_dict)
        gsp = GSP(node_encoder, graph_encoder, predictors, extractor, prompts, global_prompt, optimizer, scheduler, device, pretrain_model_dir, dataset_name, aux_info['multi_label'], random_state, method_config, shared_config, model_config, data_config, args)
        try:
            finetune_init_dict = {}
            gsp.node_encoder.load_state_dict(torch.load(gsp.pretrain_model_dir / 'best_node_encoder.pt', map_location=device))
            gsp.graph_encoder.load_state_dict(torch.load(gsp.pretrain_model_dir / 'best_graph_encoder.pt', map_location=device))
            gsp.extractor.load_state_dict(torch.load(gsp.pretrain_model_dir / 'best_extractor.pt', map_location=device))
            gsp.prompts.load_state_dict(torch.load(gsp.pretrain_model_dir / 'best_prompts.pt', map_location=device))
            gsp.predictors.load_state_dict(torch.load(gsp.pretrain_model_dir / 'best_predictors.pt', map_location=device))
            try:
                gsp.global_prompt.load_state_dict(torch.load(gsp.pretrain_model_dir / 'best_global_prompt.pt', map_location=device))
            except:
                gsp.global_prompt = None
            finetune_init_dict["loaders"] = loaders
            finetune_init_dict["test_set"] = test_set
        except:
            hparam_dict, metric_dict, finetune_init_dict = pretrain_gsp_one_seed(gsp, loaders_info, local_config, data_dir, model_name, dataset_name, method_name, device, random_state)
        start = time.time()
        metric_dict = finetune_gsp_one_seed(gsp, finetune_init_dict, local_config, data_dir, model_name, dataset_name, method_name, device, random_state, args.ood_flag)
        time_consumption = time_consumption + time.time() - start
        metric_dict_list.append(metric_dict)
    fi_res = []
    for i in metric_dict_list:
        print(i)
    for i in metric_dict_list:
        fi_res.append(i["metric/best_clf_test"])
    print(np.mean(np.array(fi_res)), np.std(np.array(fi_res)))

if __name__ == '__main__':
    main()
