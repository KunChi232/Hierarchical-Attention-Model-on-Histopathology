
import torch
from torch.utils.data import Dataloader

from util.Dataset import load_data, load_label, get_available_id, create_cv_data, msimss_create_cv_data, wgd_create_cv_data
from util.TransformerMIL import MIL
from util.Epoch import TrainEpoch, ValidEpoch

from sklearn.model_selection import KFold
import argparse
import os

parser = argparse.ArgumentParser("Experimental Setting")

parser.add_argument('--cancer_type', type = str, default = 'COAD',
                    help = 'Only including COAD, READ and CRC')
parser.add_argument('--level', type=str, default = 'slide',
                    help = 'Prediction level, slide or patient')
parser.add_argument('--hidden_dim', type = int, default = 2048,
                    help = 'patch features dimension')
parser.add_argument('--encoder_layer', type = int, default = 1,
                    help='Number of Transformer Encoder layer')
parser.add_argument('--k_sample', type = int, default = 2,
                    help='Number of top and bottom cluster to be selected')
parser.add_argument('--save_path', type = str,
                    help='Model save path')
parser.add_argument('--gene', type = str)
parser.add_argument('--mutation_type', type = str, default = None,
                    help='Deletion or Amplification')

parser.add_argument('--lr', type = float, default = 3e-4)
parser.add_argument('--epoch', type = int, default = 60)
parser.add_argument('--tau', type = float, default = 0.7)
parser.add_argument('--kfold', type = int, default = 5)




if __name__ == '__main__':
    args = parser.parse_args()
    
    k_fold = KFold(n_splits = args.kfold)
    

    lookup_dic = load_label(args.gene, args.mutation_type)
    patches_features, cluster_labels = load_data(cancer_type = args.cancer_type,
                                                level = args.level,)

    available_patient_id = get_available_id(lookup_dic, cluster_labels)


    if(args.gene == 'MSI'):
        cv_data_func = msimss_create_cv_data
    elif (args.gene == 'WGD'):
        cv_data_func = wgd_create_cv_data
    else:
        cv_data_func = create_cv_data

    for i, (train_index, val_index) in enumerate((k_fold.split(available_patient_id))):
        train_dataset, test_dataset, positive_count, negative_count = cv_data_func(available_patient_id,
                                                                                    patches_features,
                                                                                    cluster_labels,
                                                                                    train_index,
                                                                                    val_index,
                                                                                    lookup_dic,
                                                                                    level = args.level)


        train_loader = Dataloader(train_dataset, batch_size = 1, shuffle=True,
                                    num_workers = 4, pin_memory = True, drop_last = False)
        val_loader = Dataloader(val_loader, batch_size = 1, shuffle=True,
                                    num_workers = 4, pin_memory = True, drop_last = False)

        model = MIL(hidden_dim = args.hidden_dim, encoder_layer = args.encoder_layer, k_sample = args.k_sample, tau = args.tau)
        optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, weight_decay = 5e-4, momentum = 0.9, nesterov = True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = args.epoch, eta_min = 0, last_epoch = -1)


        train_epoch = TrainEpoch(model, device = 'cuda', stage = 'Train', optimizer = optimizer, 
                                    positive_count = positive_count, negative_count = negative_count)
        val_epoch = ValidEpoch(model, device = 'cuda', stage = 'Valid', 
                                positive_count = positive_count, negative_count = negative_count)


        max_auc = 0

        model_name = os.path.join(args.save_path, str(i), 'R50TransformerMIL.h5')

        for epoch_count in range(1, args.epoch + 1) :
            train_logs = train_epoch.run(train_loader)
            val_logs = val_epoch.run(val_loader)

            if max_auc < val_logs['AUC']:
                max_auc = val_logs['AUC']
                torch.save(model_name)
                print('Model save # {}'.format(model_name))