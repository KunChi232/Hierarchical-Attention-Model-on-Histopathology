from torch.utils.data import DataLoader, Dataset, Sampler
import numpy as np
import joblib
import pickle


def load_data(cancer_type = 'COAD', level = 'slide'):
    if cancer_type == 'COAD':
        all_features = joblib.load('/data/Model/representation_visualize/2021-01-21_15-33-08_ResNet18SimCLR/ResNet50ImageNet_training_feature.pkl')
        all_features.update(joblib.load('/data/Model/representation_visualize/2021-01-21_15-33-08_ResNet18SimCLR/ResNet50ImageNet_testing_feature.pkl'))

        with open('/data/Model/representation_visualize/2021-01-21_15-33-08_ResNet18SimCLR/ResNet50ImageNet_slide_cluster_label_k=10.pkl', 'rb') as f:
            cluster_label = pickle.load(f)
            
        tumor_patch = np.load('/data/Model/512dense_tumor_Zenodo+DrYu_0.5.npy')
        tumor_patch = set([p.split('/')[-1][:-4] for p in tumor_patch])
    else:
        
        with open('/data/READ_Frozen/EXTRACTED_FEATURE/R50ImageNet_feature.pkl', 'rb') as f:
            all_features = pickle.load(f)

        with open('/data/READ_Frozen/EXTRACTED_FEATURE/R50ImageNet_slide_cluster_label_k=10.pkl', 'rb') as f:
            cluster_label = pickle.load(f)
            
        tumor_patch = np.load('/data/msimss_tcga/read_tumor_Zenodo+DrYu_0.5.npy')
        tumor_patch = set([p.split('/')[-1][:-4] for p in tumor_patch])
        
        
    return all_features, cluster_label, tumor_patch


def create_cv_data(patients, all_features, cluster_label, tumor_patch, train_index, test_index, lookup):
    train_patient, test_patient = np.array(patients)[train_index], np.array(patients)[test_index]
    train_cluster = {}
    test_cluster = {}
    
    for k, v in cluster_label.items():
        p_id = k[:12]
        slide_id = k[:23]
        if(slide_id[13] == '1') : continue
        if(p_id in train_patient):
            train_cluster[slide_id] = [[] for i in range(10)]
            for p, c in v.items():
                if(p not in tumor_patch):
                    continue
                train_cluster[slide_id][c].append(p)
        elif(p_id in test_patient):
            test_cluster[slide_id] = [[] for i in range(10)]
            for p, c in v.items():
                if(p not in tumor_patch):
                    continue
                test_cluster[slide_id][c].append(p)    
                
                
    drop = []
    for k, v in train_cluster.items():
        total = sum([len(c) for c in v])
        if(total < 50) :
            drop.append(k)

    for s_id in drop:
        del train_cluster[s_id]


    drop = []
    for k, v in test_cluster.items():
        total = sum([len(c) for c in v])
        if(total < 50) :
            drop.append(k)

    for s_id in drop:
        del test_cluster[s_id]
        
    pos_count = 0    
    for k in train_cluster.keys():
        pos_count += lookup[k[:12]]
        
    test_pos_count = 0    
    for k in test_cluster.keys():
        test_pos_count += lookup[k[:12]]
    print(test_pos_count, len(test_cluster) - test_pos_count)    
    
    train_dataset = ClusterDataset(
        features = all_features,
        cluster = train_cluster,
        cls_lookup = lookup,
    )


    test_dataset = ClusterDataset(
        features = all_features,
        cluster = test_cluster,
        cls_lookup = lookup,
    )
    
    return train_dataset, test_dataset, pos_count, len(train_cluster) - pos_count


def wgd_create_cv_data(patients, all_features, cluster_label, tumor_patch, train_index, test_index, lookup):
    train_patient, test_patient = np.array(patients)[train_index], np.array(patients)[test_index]
    train_cluster = {}
    test_cluster = {}
    
    for k, v in cluster_label.items():
        p_id = k[:15]
        slide_id = k[:23]
        if(slide_id[13] == '1') : continue
        if(p_id in train_patient):
            train_cluster[slide_id] = [[] for i in range(10)]
            for p, c in v.items():
                if(p not in tumor_patch):
                    continue
                train_cluster[slide_id][c].append(p)
        elif(p_id in test_patient):
            test_cluster[slide_id] = [[] for i in range(10)]
            for p, c in v.items():
                if(p not in tumor_patch):
                    continue
                test_cluster[slide_id][c].append(p)    
                
                
    drop = []
    for k, v in train_cluster.items():
        total = sum([len(c) for c in v])
        if(total < 50) :
            drop.append(k)

    for s_id in drop:
        del train_cluster[s_id]


    drop = []
    for k, v in test_cluster.items():
        total = sum([len(c) for c in v])
        if(total < 50) :
            drop.append(k)

    for s_id in drop:
        del test_cluster[s_id]
        
    pos_count = 0    
    for k in train_cluster.keys():
        pos_count += lookup[k[:15]]

    test_pos_count = 0    
    for k in test_cluster.keys():
        test_pos_count += lookup[k[:15]]
    print(test_pos_count, len(test_cluster) - test_pos_count)
    train_dataset = ClusterDataset(
        features = all_features,
        cluster = train_cluster,
        cls_lookup = lookup,
    )


    test_dataset = ClusterDataset(
        features = all_features,
        cluster = test_cluster,
        cls_lookup = lookup,
    )
    
    return train_dataset, test_dataset, pos_count, len(train_cluster) - pos_count


def msimss_create_cv_data(patients, all_features, cluster_label, train_index, test_index, lookup):
    train_patient, test_patient = np.array(patients)[train_index], np.array(patients)[test_index]
    train_cluster = {}
    test_cluster = {}
    
    for k, v in cluster_label.items():
        p_id = k[:12]
        slide_id = k[:23]
        if(p_id in train_patient):
            train_cluster[p_id] = [[] for i in range(10)]
            for p, c in v.items():
#                 if(p not in tumor_patch):
#                     continue
                train_cluster[p_id][c].append(p)
        elif(p_id in test_patient):
            test_cluster[p_id] = [[] for i in range(10)]
            for p, c in v.items():
#                 if(p not in tumor_patch):
#                     continue
                test_cluster[p_id][c].append(p)    
                
                
    drop = []
    for k, v in train_cluster.items():
        total = sum([len(c) for c in v])
        if(total < 50) :
            drop.append(k)

    for s_id in drop:
        del train_cluster[s_id]


    drop = []
    for k, v in test_cluster.items():
        total = sum([len(c) for c in v])
        if(total < 50) :
            drop.append(k)

    for s_id in drop:
        del test_cluster[s_id]
        
    pos_count = 0    
    for k in train_cluster.keys():
        pos_count += lookup[k[:12]]
    test_pos_count = 0    
    for k in test_cluster.keys():
        test_pos_count += lookup[k[:12]]
    print(test_pos_count, len(test_cluster) - test_pos_count)    
           
    train_dataset = ClusterDataset(
        features = all_features,
        cluster = train_cluster,
        cls_lookup = lookup,
    )


    test_dataset = ClusterDataset(
        features = all_features,
        cluster = test_cluster,
        cls_lookup = lookup,
    )
    
    return train_dataset, test_dataset, pos_count, len(train_cluster) - pos_count


class ClusterDataset(Dataset):
    def __init__(self, features, cluster, cls_lookup, score_lookup = None):
        self.patient = list(cluster.keys())
        self.features = features
        self.cluster = cluster
        self.cls_lookup = cls_lookup
        self.score_lookup = score_lookup
    def __getitem__(self, i):
        p_id = self.patient[i]
        cluster = self.cluster[p_id]
        p_id = p_id[:12]
        data = []
        patch_name = []
        for ind, patch in enumerate(cluster):              
            feature = []
            pn = []
            if(len(patch) == 0):
                continue
            for p in patch:
                feature.append(self.features[p])
                pn.append(p)
            feature = np.array(feature)
            data.append(feature)
            patch_name.append(pn)
        if(self.score_lookup is not None):
            score = self.score_lookup[p_id]
            
            return data, np.array([1- score , score]), self.cls_lookup[p_id]
        else:
            return data, patch_name, self.cls_lookup[p_id]
        
        
    def __len__(self):
        return len(self.patient)
    
    
    
