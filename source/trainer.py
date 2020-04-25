import torch
import torch.nn as nn
from torch.optim import Adam
import time
import os
from tqdm import tqdm
import numpy as np
import torch_geometric
from torch_geometric.data import Data,Batch
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import label_binarize

class EventTrainer_new:
    def __init__(self, option, model, train_dataloader=None, valid_dataloader=None, test_dataloader=None):
        # Superparam
        self.option = option
        lr = option.lr
        betas = (option.adam_beta1, option.adam_beta2)
        weight_decay = option.adam_weight_decay
        with_cuda = option.with_cuda
        cuda_devices = option.gpu
        self.log_freq = option.log_freq
        self.save_path = option.this_expsdir
        self.epochs = option.epochs
        self.clip_norm = option.clip_norm
        self.patience = option.patience

        self.best_valid_acc = 0
        self.best_valid_auc = 0
        self.best_valid_loss = np.inf

        self.start = time.time()
        self.msg_with_time = lambda msg: \
                "%s Time elapsed %0.2f hrs (%0.1f mins)" \
                % (msg, (time.time() - self.start) / 3600.,
                        (time.time() - self.start) / 60.)
        # Setup cuda device for BERT training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:{}".format(cuda_devices[0]) if cuda_condition else "cpu")

        self.model = model.to(self.device)
        if len(cuda_devices)>1:
            self.model = torch_geometric.nn.DataParallel(self.model)

        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.valid_data = valid_dataloader
        self.test_data = test_dataloader

        # Setting the Adam optimizer with hyper-param

        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        # self.optim = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, betas=betas, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optim, gamma=self.option.lr_decay)
        # Using Negative Log Likelihood Loss function for predicting the masked_token
        self.criterion = nn.NLLLoss()

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def iteration(self, epoch, data_loader, mode='train'):

        avg_loss = []
        total_correct = []
        all_auc = []


        if mode=='train':
            self.model.train()
            for i, data in enumerate(tqdm(data_loader)):
                if len(self.option.gpu)>1:
                    data = Batch.from_data_list(data).to(self.device)
                    output = self.model.forward(data)
                else:
                    data = data.to(self.device)
                    split_size = list(data.sub_nums.cpu().numpy())
                    split_size_edge = (data.sub_nums * (data.sub_nums - 1)).cpu().numpy()
                    split_size_edge = list(np.where(split_size_edge == 0, 1, split_size_edge))

                    sub_nodes = torch.split(data.sub_x,split_size,dim=0)
                    sub_edge_index = torch.split(data.sub_edge_idx,split_size_edge,dim=0)
                    sub_edge_attr = torch.split(data.sub_edge_attr,split_size_edge,dim=0)

                    if np.isnan(data.x.cpu().numpy()).any() :
                        print('data x '+"*"*40)
                        print(data.x.shape)

                        print(data.x.cpu().numpy())
                    for i in range(len(split_size)):
                        if np.isnan(sub_nodes[i].cpu().numpy()).any():
                            print(i)
                            print(sub_nodes.shape)
                            print(sub_nodes[0])
                            print(sub_nodes)

                    sub_graph = []
                    for i in range(len(split_size)):
                        sub_graph.append(Data(x=sub_nodes[i],edge_index=sub_edge_index[i].t(),edge_attr=sub_edge_attr[i]))
                    data2 = Batch.from_data_list(sub_graph).to(self.device)

                    data.sub_edge_idx,data.sub_x,data.sub_edge_attr,data.sub_nums=None,None,None,None
                    output = self.model.forward(data,data2)



                ground_truth = data.y


                loss = self.criterion(output, ground_truth)
                loss.backward()
                if self.clip_norm>0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.clip_norm)
                self.optim.step()
                self.optim.zero_grad()
                avg_loss.append(loss.item())

                prediction = output.argmax(dim=1).cpu().numpy()
                target = ground_truth.cpu().numpy()
                acc = metrics.accuracy_score(target, prediction)
                auc = metrics.roc_auc_score(label_binarize(target, np.arange(4)).T,
                                            label_binarize(prediction, np.arange(4)).T, average='macro')
                all_auc.append(auc)

                # acc = prediction.eq(data.y).sum().item()/output.size(0)
                total_correct.append(acc)

        else:
            self.model.eval()
            with torch.no_grad():
                for i, data in enumerate(tqdm(data_loader)):
                    data = data.to(self.device)
                    split_size = list(data.sub_nums.cpu().numpy())
                    split_size_edge = (data.sub_nums * (data.sub_nums - 1)).cpu().numpy()
                    split_size_edge = list(np.where(split_size_edge == 0, 1, split_size_edge))
                    sub_nodes = torch.split(data.sub_x, split_size, dim=0)
                    sub_edge_index = torch.split(data.sub_edge_idx, split_size_edge, dim=0)
                    sub_edge_attr = torch.split(data.sub_edge_attr, split_size_edge, dim=0)

                    sub_graph = []
                    for i in range(len(split_size)):
                        sub_graph.append(Data(x=sub_nodes[i], edge_index=sub_edge_index[i].t(), edge_attr=sub_edge_attr[i]))
                    data2 = Batch.from_data_list(sub_graph).to(self.device)

                    data.sub_edge_idx, data.sub_x, data.sub_edge_attr, data.sub_nums = None, None, None, None
                    output = self.model.forward(data,data2)
                    ground_truth = data.y
                    loss = self.criterion(output, ground_truth)
                    avg_loss.append(loss.item())

                    prediction = output.argmax(dim=1).cpu().numpy()
                    target = ground_truth.cpu().numpy()

                    acc = metrics.accuracy_score(target, prediction)
                    auc = metrics.roc_auc_score(label_binarize(target, np.arange(4)).T,
                                                label_binarize(prediction, np.arange(4)).T, average='macro')
                    all_auc.append(auc)
                    # acc = prediction.eq(data.y).sum().item()/output.size(0)
                    total_correct.append(acc)
        Acc, Loss,auc = np.mean(total_correct),np.mean(avg_loss),np.mean(all_auc)
        msg ='Epoch:%d, %s, loss:%0.2f, acc:%0.3f, auc:%0.3f'%(epoch, mode, Loss, Acc,auc)
        log = self.msg_with_time(msg)
        print(log)
        self.save_log(log)
        return Acc, auc,Loss

    def train(self):
        cnt=0
        best_train_acc=0
        best_train_auc=0

        for epoch in range(self.epochs):
            print('Current lr is {}.'.format(self.scheduler.get_lr()))
            train_acc,train_auc,train_loss = self.iteration(epoch,self.train_data,mode='train')
            valid_acc,valid_auc,val_loss = self.iteration(epoch, self.valid_data, mode='valid')
            test_acc,test_auc,test_loss = self.iteration(epoch, self.test_data,   mode='test')
            self.scheduler.step()

            self.best_valid_acc =  max(self.best_valid_acc,valid_acc)
            self.best_valid_auc = max(self.best_valid_auc,valid_auc)
            self.best_valid_loss = min(self.best_valid_loss, val_loss)

            best_train_acc = max(best_train_acc, train_acc)
            best_train_auc = max(best_train_auc, train_auc)

            if self.best_valid_acc == valid_acc or self.best_valid_auc == valid_auc:
                self.save_model(epoch)



    def eval(self):
        avg_loss = []
        total_correct = []
        all_auc = []
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(tqdm(self.test_data)):
                data = data.to(self.device)
                split_size = list(data.sub_nums.cpu().numpy())
                split_size_edge = (data.sub_nums * (data.sub_nums - 1)).cpu().numpy()
                split_size_edge = list(np.where(split_size_edge == 0, 1, split_size_edge))
                sub_nodes = torch.split(data.sub_x, split_size, dim=0)
                sub_edge_index = torch.split(data.sub_edge_idx, split_size_edge, dim=0)
                sub_edge_attr = torch.split(data.sub_edge_attr, split_size_edge, dim=0)

                sub_graph = []
                for i in range(len(split_size)):
                    sub_graph.append(Data(x=sub_nodes[i], edge_index=sub_edge_index[i].t(), edge_attr=sub_edge_attr[i]))
                data2 = Batch.from_data_list(sub_graph).to(self.device)

                data.sub_edge_idx, data.sub_x, data.sub_edge_attr, data.sub_nums = None, None, None, None
                output = self.model.forward(data, data2)
                ground_truth = data.y
                loss = self.criterion(output, ground_truth)
                avg_loss.append(loss.item())

                prediction = output.argmax(dim=1).cpu().numpy()
                target = ground_truth.cpu().numpy()

                acc = metrics.accuracy_score(target, prediction)
                auc = metrics.roc_auc_score(label_binarize(target, np.arange(4)).T,
                                            label_binarize(prediction, np.arange(4)).T, average='macro')
                all_auc.append(auc)
                # acc = prediction.eq(data.y).sum().item()/output.size(0)
                total_correct.append(acc)

        Acc, Loss, auc = np.mean(total_correct), np.mean(avg_loss), np.mean(all_auc)
        msg = 'The test dataset acc:%0.3f, auc:%0.3f' % (Acc, auc)
        log = self.msg_with_time(msg)
        print(log)


    def save_model(self, epoch):
        with open(os.path.join(self.save_path,'best_model_{}.pkl'.format(epoch)), 'wb') as f:
            if len(self.option.gpu)>1:
                torch.save(self.model.module.state_dict(),f)
            else:
                torch.save(self.model.state_dict(), f)
        log = 'Model saved at epoch {}'.format(epoch)
        print(log)
        self.save_log(log)

    def save_log(self,strs):
        with open(os.path.join(self.save_path,'log.txt'),'a+') as f:
            f.write(strs)
            f.write('\n')
