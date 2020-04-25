import pandas as pd
import numpy as np
from tqdm import tqdm
import math,os
import torch
import torch.utils.data
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset

import utils
class HierDataset_paper(InMemoryDataset):
    def __init__(self, root, processed=None,gpus=[0]):
        self.processed = processed
        self.device = torch.device("cuda:{}".format(gpus[0]))
        super(HierDataset_paper, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['complex_train_R04_event.csv', 'complex_train_R04_jet.csv', 'complex_train_R04_particle.csv']

    @property
    def processed_file_names(self):
        return [os.path.join(self.processed,'hier_data.pt')]

    @property
    def jet_num_dim(self):
        return 6+1+7 #19

    @property
    def jet_edge_num_dim(self):
        return 3+1

    @property
    def particle_num_dim(self):
        return 5+1 #11

    @property
    def particle_edge_num_dim(self):
        return 3+1

    def download(self):
        pass

    def load_file(self):
        event = pd.read_csv(self.raw_paths[0])
        jet = pd.read_csv(self.raw_paths[1])
        particle = pd.read_csv(self.raw_paths[2])

        event = event.sample(frac=1).reset_index(drop=True)  # shuffle data

        # modify the event dic
        event_dic = {}
        for i, item in enumerate(event['event_id']):
            event_dic[item] = i

        jet_dic = {}
        for i, item in enumerate(jet['jet_id'].values):
            jet_dic[item] = i

        # preprocess
        event_num = []
        for item in jet['event_id'].values:
            event_num.append(event_dic[item])
        jet['event_num'] = event_num

        jet_num = []
        for item in particle['jet_id'].values:
            jet_num.append(jet_dic[item])
        particle['jet_num'] = jet_num

        return event, jet, particle

    def cal_angle(self,m1, m2):
        # m1: N,3; m2: N,3
        m1_norm = torch.norm(m1, p=2, dim=-1)
        m2_norm = torch.norm(m2, p=2, dim=-1)
        m1_m2 = torch.matmul(m1.unsqueeze(1), m2.unsqueeze(2)).squeeze()
        cos = m1_m2 / (m1_norm * m2_norm)
        cos = torch.clamp(cos, -1.0, 1.0)
        angle = torch.acos(cos).view(-1, 1)  # N,1
        return angle

    def process(self):
        event, jet, particle = self.load_file()
        jet_id_array = np.array(range(len(jet)))
        jet_num_in_event = event['number_of_jet_in_this_event'].to_numpy()
        parc_num_in_jet = jet['number_of_particles_in_this_jet'].to_numpy()
        particle_jetnum = particle['jet_num'].to_numpy()
        jet_eventnum = jet['event_num'].to_numpy()

        # jet label preprocess
        jet_label_array = jet['label'].to_numpy()
        jet_label_array = np.where(jet_label_array == 1, 0, jet_label_array)
        jet_label_array = np.where(jet_label_array == 4, 1, jet_label_array)
        jet_label_array = np.where(jet_label_array == 5, 2, jet_label_array)
        jet_label_array = np.where(jet_label_array == 21, 3, jet_label_array)
        print('Files are loaded, there are {} data'.format(len(event)))

        jet_attr = ['jet_px', 'jet_py', 'jet_pz', 'jet_energy', 'jet_mass', 'number_of_particles_in_this_jet']  # The event this jet belong to has n jets
        particle_attr = ['particle_px', 'particle_py', 'particle_pz', 'particle_energy','particle_mass']

        jet_arry = jet[jet_attr].to_numpy()
        particle_arry = particle[particle_attr].to_numpy()

        particle_catg = utils.standard_category(particle['particle_category'].to_numpy())

        jet_arry = utils.z_score(jet_arry,calcute=True)
        particle_arry = np.concatenate([utils.z_score(particle_arry,calcute=True), particle_catg], axis=1)  ###

        jet_arry,particle_arry=torch.tensor(jet_arry,dtype=torch.float32).to(self.device),\
                               torch.tensor(particle_arry,dtype=torch.float32).to(self.device)

        jet_att_array = jet_arry
        jet_pos_array = jet_arry[:, :3]

        particle_attr_array = particle_arry
        particle_pos_array = particle_arry[:, :3]

        data_list = []

        idx_list_for_eventnum = utils.get_grouped_sorted_idx_list(jet_eventnum,jet_num_in_event)
        idx_list_for_particle_jetnum = utils.get_grouped_sorted_idx_list(particle_jetnum,parc_num_in_jet)
        for i,idx in enumerate(tqdm(idx_list_for_eventnum)):

            jet_ids = jet_id_array[idx]
            jet_feat = jet_att_array[idx]  # (n,6)
            targets = jet_label_array[idx]

            if jet_feat.dim() == 1:
                jet_feat = jet_feat.view(1,-1)
                jet_ids = [jet_ids]
                targets = [targets]

            targets=targets[0:1]


            jet_feat = torch.cat([jet_feat, torch.full((len(jet_feat),1),jet_num_in_event[i]).float().to(self.device)],dim=1)
            jet_position = jet_pos_array[idx]

            assert len(jet_feat) > 0, 'No Nodes!'

            jet_edge_att = []
            jet_edge_index = []
            for i in range(len(jet_feat)):
                for j in range(len(jet_feat)):
                    if i != j:
                        jet_edge_att.append([i, j])
                        jet_edge_index.append([i, j])

            sub_nodes = []
            sub_edge_index=[]
            sub_edge_attr = []
            sub_nums = []
            jets_adds = []

            for n in jet_ids:
                par_idx = idx_list_for_particle_jetnum[n]
                nodes = particle_attr_array[par_idx]  # particle[particle['jet_num'] == i][particle_attr].to_numpy()
                position = particle_pos_array[par_idx]
                assert len(nodes) > 0, 'No Nodes!'


                # delete this jet
                if nodes.dim() == 1:
                    nodes = nodes.view(1,-1)
                # new features
                jets_adds.append(torch.tensor([nodes[:, 0].cpu().numpy().std(),
                                  nodes[:, 1].cpu().numpy().std(),
                                  nodes[:, 2].cpu().numpy().std(),
                                  nodes[:, 3].mean(),
                                  nodes[:, 4].mean(),
                                  nodes[:, 5].mean(),
                                  torch.mode(nodes[:, -1].long())[0].float()
                                  ]).view(1,-1).to(self.device)
                                 )
                edge_att = []
                edge_index = []
                for i in range(len(nodes)):
                    for j in range(len(nodes)):
                        if i != j:
                            edge_att.append([i, j])
                            edge_index.append([i, j])


                edge_att = torch.FloatTensor(edge_att).to(self.device)
                edge_index = torch.LongTensor(edge_index).to(self.device)

                # cal distance and append to edge attr
                if len(nodes) > 1:
                    (row, col) = edge_index.t()
                    p1, p2 = position[col], position[row]
                    dist = torch.norm(p1-p2, p=2, dim=-1).view(-1, 1)

                    angle = self.cal_angle(p1,p2)
                    edge_att = torch.cat([edge_att, dist.type_as(edge_att),angle], dim=-1)
                else:
                    edge_index = torch.LongTensor([[0,0]]).to(self.device)
                    edge_att = torch.FloatTensor([[0, 0, 0,0]]).to(self.device)

                # build little graph data
                sub_nodes.append(nodes)
                sub_edge_index.append(edge_index)
                sub_edge_attr.append(edge_att)
                sub_nums.append(len(nodes))

            sub_nodes = torch.cat(sub_nodes).float().to(self.device)
            sub_edge_index = torch.cat(sub_edge_index).long().to(self.device)
            sub_edge_attr = torch.cat(sub_edge_attr).float().to(self.device)
            sub_nums = torch.LongTensor(sub_nums).to(self.device)


            jet_feat = torch.cat([jet_feat,torch.cat(jets_adds).to(self.device)],dim=1)


            jet_edge_att = torch.FloatTensor(jet_edge_att).to(self.device)
            jet_edge_index = torch.LongTensor(jet_edge_index).t().to(self.device)


            # cal distance and append to edge attr
            if len(jet_feat) > 1:
                (row, col) = jet_edge_index
                p1,p2 = jet_position[col],jet_position[row]
                dist = torch.norm(p1 - p2, p=2, dim=-1).view(-1, 1)

                angle = self.cal_angle(p1, p2)
                jet_edge_att = torch.cat([jet_edge_att, dist.type_as(jet_edge_att),angle], dim=-1)
            else:
                jet_edge_index = torch.LongTensor([[0], [0]]).to(self.device)
                jet_edge_att = torch.FloatTensor([[0, 0, 0,0]]).to(self.device)

            data_list.append(Data(x=jet_feat.cpu(), edge_index=jet_edge_index.cpu(), edge_attr=jet_edge_att.cpu(),y=torch.LongTensor(targets),
                                  sub_x=sub_nodes.cpu(), sub_edge_idx=sub_edge_index.cpu(), sub_edge_attr=sub_edge_attr.cpu(),
                                  sub_nums=sub_nums.cpu()))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


if __name__ == '__main__':
    # dataset = HierDataset_event_nonormal(root='data/',processed='event_nonormal')
    dataset_eval = HierDataset_paper(root='data/', processed='trys',gpus=[5])
