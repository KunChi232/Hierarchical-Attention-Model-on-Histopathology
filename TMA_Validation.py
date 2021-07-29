
import torch
from torch.utils.data import DataLoader

from util.Dataset import load_label, ClusterDataset, load_TMA_data
from util.TransformerMIL import MIL
from util.EMA import EMA
from util.Epoch import TrainEpoch, ValidEpoch

from sklearn.model_selection import KFold, train_test_split
import argparse
import os, random

parser = argparse.ArgumentParser("TMA Extenral Validation Setting")

parser.add_argument('--level', type=str, default = 'slide',
                    help = 'Prediction level, slide or patient')
parser.add_argument('--hidden_dim', type = int, default = 512,
                    help = 'patch features dimension')
parser.add_argument('--encoder_layer', type = int, default = 1,
                    help = 'Number of Transformer Encoder layer')
parser.add_argument('--k_sample', type = int, default = 2,
                    help = 'Number of top and bottom cluster to be selected')
parser.add_argument('--tau', type = float, default = 0.7)
parser.add_argument('--save_path', type = str,
                    help = 'Model save path')

parser.add_argument('--label', type = str, default = None,
                    help = 'path to label pickle file')

parser.add_argument('--evaluate_mode', type = str, default='holdout',
                    help='holdout or kfold')
parser.add_argument('--kfold', type = int, default = 5)


if __name__ == '__main__':
    args = parser.parse_args()

    if(args.label == None):
        raise ValueError('label pickle file path cannot be empty')    

    lookup_dic = load_label(args.label)
    patches_features, cluster_labels = load_TMA_data()
    available_patient_id = list(lookup_dic.keys())
    
    evaluate_cluster = {}
    for k, v in cluster_labels.items():
        if(k in available_patient_id):
            evaluate_cluster[k] = [[] for i in range(10)]
            for p, c in v.items():
                evaluate_cluster[k][c].append(p)

    evaluate_dataset = ClusterDataset(patches_features, evaluate_cluster, lookup_dic)
    evaluate_loader = DataLoader(evaluate_dataset, batch_size = 1, shuffle = False, num_workers = 1, pin_memory = True, drop_last = True)

    model = MIL(hidden_dim = args.hidden_dim, encoder_layer = args.encoder_layer, k_sample = args.k_sample, tau = args.tau)
    model = EMA(model, 0.999)
    model.eval()

    for i in range(args.kfold):
        model_name = os.path.join(args.save_path, str(i), 'R50TransformerMIL.h5')

        model.load_state_dict(torch.load(model_name))

        val_epoch = ValidEpoch(model, device = 'cuda', stage = 'TMA External Validation', 
                                positive_count = 0, negative_count = 0)


        val_logs = val_epoch.run(evaluate_loader)

        
        if(args.evaluate_mode == 'holdout'):
            break


            