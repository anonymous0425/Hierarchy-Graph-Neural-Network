import argparse
import os
import gc
from torch.utils.data import random_split
from model import *
from trainer import EventTrainer_new
from data import  HierDataset_paper
import random
import numpy as np
import time
from torch_geometric.data import DataLoader


class Option(object):
    def __init__(self, d):
        self.__dict__ = d

    def save(self):
        with open(os.path.join(self.this_expsdir, "option.txt"), "w") as f:
            for key, value in sorted(self.__dict__.items(), key=lambda x: x[0]):
                f.write("%s, %s\n" % (key, str(value)))

def train():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=int, default=0)
    parser.add_argument("--data", type=str, default='data/', help="dataset for graph")
    parser.add_argument("--processed_data", type=str, default='event_new', help="dataset for graph")
    parser.add_argument("--hid1", type=int, default=64, help="hidden size of transformer model")
    parser.add_argument("--hid2", type=int, default=128, help="hidden size of transformer model")
    parser.add_argument('--num_step_message_passing', type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument('--seed', default=68, type=int)
    parser.add_argument('--patience', default=100, type=int)
    parser.add_argument('--heads', default=4, type=int)
    parser.add_argument('--gnn',default=None,type=str,help='GCN GraphSAGE GAT EGAT')

    parser.add_argument("--batch_size", type=int, default=128, help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=1000, help="number of epochs")
    parser.add_argument("--num_workers", type=int, default=4, help="dataloader worker size")

    parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: true, or false")
    parser.add_argument("--log_freq", type=int, default=10, help="printing loss every n iter: setting n")
    parser.add_argument("--gpu", type=int, nargs='+', default=None, help="CUDA device ids")

    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument("--lr_decay", type=float, default=0.98, help="learning rate decay")
    parser.add_argument("--adam_weight_decay", type=float, default=0.0001, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")
    parser.add_argument("--clip_norm", type=float, default=0.0)

    parser.add_argument('--load', default=None, type=str)
    parser.add_argument('--load_particle', default=None, type=str)
    parser.add_argument('--eval', default=False, action="store_true")
    parser.add_argument('--competition', default=False, action="store_true")
    parser.add_argument("--exps_dir", default='test',type=str, help="out/")
    parser.add_argument('--exp_name', default='test', type=str)
    parser.add_argument('--max_atom', default=100, type=int)

    d = vars(parser.parse_args())
    args = Option(d)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.exp_name is None:
      args.tag = time.strftime("%y-%m-%d-%H-%M")
    else:
      args.tag = args.exp_name
    args.this_expsdir = os.path.join(args.exps_dir, args.tag)
    if not os.path.exists(args.this_expsdir):
        os.makedirs(args.this_expsdir)
    if not os.path.exists(os.path.join(args.data,'processed/'+args.processed_data)):
        os.mkdir(os.path.join(args.data,'processed/'+args.processed_data))

    dataset = HierDataset_paper(root=args.data, processed=args.processed_data)
    args.particle_features = dataset.particle_num_dim
    args.particle_edge_features = dataset.particle_edge_num_dim
    args.jet_features = dataset.jet_num_dim
    args.jet_edge_features = dataset.jet_edge_num_dim

    num_training = int(len(dataset) * 0.8)
    num_val = int(len(dataset) * 0.1)
    num_test = len(dataset)-num_training-num_val
    training_set, validation_set,test_set = random_split(dataset,[num_training,num_val,num_test])

    train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=False,num_workers=4)
    val_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False,num_workers=4)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    del dataset,training_set,validation_set,test_set
    gc.collect()
    print("Data prepared")



    if args.model==0:
        model = HierMPNN_attention_set(args)
    elif args.model==1:
        model = MPNN(args)
    elif args.model==2:
        model = Henrion_MPNN(args)
    elif args.model==3:
        model = HierEGAT_attention_set(args)
    elif args.model==4:
        model = EGAT(args)
    elif args.model==5:
        model = Baselines(args)
    elif args.model==6:
        model = Baselines_withouthier(args)
    elif args.model==7:
        model = Murat_MPNN(args)


    trainer = EventTrainer_new(args, model, train_loader, val_loader, test_loader)
    print("Model and Trainer are bulit")
    args.save()
    print("Option saved.")
    if args.load is  not None:
        with open(args.load, 'rb') as f:
            model.load_state_dict(torch.load(f))
        if args.eval:
            print('Evaluation Start')
            trainer.eval()

    if not args.eval:
        print("Training Start")
        trainer.train()

if __name__ == '__main__':
    train()
