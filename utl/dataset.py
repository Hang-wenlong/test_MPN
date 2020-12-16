import numpy as np
import glob
from sklearn.model_selection import KFold

def load_dataset(dataset_path, n_folds, rand_state):
    """
    Parameters
    --------------------
    :param dataset_path:
    :param n_folds:
    :return: list
        List contains split datasets for K-Fold cross-validation
    """

    # load datapath from path
    c1 = glob.glob(dataset_path+'/0/*M*')
    c2 = glob.glob(dataset_path+'/1/*M*')
    c3 = glob.glob(dataset_path+'/2/*M*')
    c4 = glob.glob(dataset_path+'/3/*M*')

    c1_num = len(c1)
    c2_num = len(c2)
    c3_num = len(c3)
    c4_num = len(c4)
    all_path = c1 + c2 + c3 + c4

    #num_bag = len(all_path)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=rand_state)
    datasets = []
    for train_idx, test_idx in kf.split(all_path):
        dataset = {}
        dataset['train'] = [all_path[ibag] for ibag in train_idx]
        dataset['test'] = [all_path[ibag] for ibag in test_idx]
        datasets.append(dataset)
    return datasets