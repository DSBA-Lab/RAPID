import numpy as np
import pandas as pd

from tqdm import tqdm
import os
import pickle
import json

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, auc, precision_recall_curve
from sklearn.neighbors import NearestNeighbors

import numpy as np

import torch
from torch import matmul

#user warning
import warnings
warnings.filterwarnings("ignore")

import argparse
import sys

from utils import str2bool, load_pickle, save_pickle, set_seed
#for time check
import time

def get_threshold_prc(score, test_label, not_to_numpy=False):
    if not_to_numpy:
        precision, recall, thresholds = precision_recall_curve(test_label, score)
    else:
        precision, recall, thresholds = precision_recall_curve(test_label, score.to_numpy())
    #get best f1score
    f1 = np.array([2 * (pr * re) / (pr + re + 1e-10) for pr, re in zip(precision, recall)])

    ix = np.argmax(f1)
    best_thresh = thresholds[ix]
    return best_thresh
 
def get_threshold_roc(score, test_label, not_to_numpy=False):
    if not_to_numpy:
        fpr, tpr, thresholds =roc_curve(test_label, score)
    else:
        fpr, tpr, thresholds =roc_curve(test_label, score.to_numpy())
    J = tpr - fpr
    ix = np.argmax(J)
    best_thresh = thresholds[ix]
    print('Best Threshold=%f, sensitivity = %.3f, specificity = %.3f, J=%.3f' % (best_thresh, tpr[ix], 1-fpr[ix], J[ix]))
    return best_thresh

def get_detection_score(label, pred, time_list,exp_name = 'current_exp', result_df=None):
    time_for_get_coreSet, time_for_cal_maxsim, time_for_get_adscore_for_all = time_list
    # get detection score
    print(f'confusion_matrix: \n{confusion_matrix(label, pred)}')
    print(f'accuracy_score: {accuracy_score(label, pred)}')
    print(f'f1_score: {f1_score(label, pred)}')
    print(f'precision_score: {precision_score(label, pred)}')
    print(f'recall_score: {recall_score(label, pred)}')
    print(f'roc_auc_score: {roc_auc_score(label, pred)}')
    print(classification_report(label, pred))

    if result_df is None:
        #make new dataframe and make exp_name to be index
        result_df = pd.DataFrame()
        result_df['exp_name'] = [exp_name]
        result_df = result_df.set_index('exp_name')
        result_df['f1_score'] = [f1_score(label, pred)]
        result_df['roc_auc_score'] = [roc_auc_score(label, pred)]
        result_df['precision_score'] = [precision_score(label, pred)]
        result_df['recall_score'] = [recall_score(label, pred)]
        result_df['accuracy_score'] = [accuracy_score(label, pred)]
        result_df['coreSet_time'] = [time_for_get_coreSet]
        result_df['maxsim_time'] = [time_for_cal_maxsim]
        result_df['lookup_all_adscore_time'] = [time_for_get_adscore_for_all]
    else:
        result_df.loc[exp_name] = [f1_score(label, pred),roc_auc_score(label, pred), precision_score(label, pred), recall_score(label, pred), accuracy_score(label, pred), 
                                   time_for_get_coreSet, time_for_cal_maxsim, time_for_get_adscore_for_all]
    return result_df

def get_threshold_pred_distance(score, test_label, desc='make maxsim_ori using lookup'):
    # get score of all data from unique data
    time_adscore_all=time.time()
    score_ori=np.zeros((test_label.shape[0]))
    for i in tqdm(range(test_label.shape[0]), desc=desc):
        score_ori[i] = score[test_unique_lookup_table[i]]
    time_adscore_all=time.time()-time_adscore_all
    best_thresh = threshold_function(score_ori, test_label['label'], True)
    pred=(score_ori >= best_thresh).astype(int)
    return pred, time_adscore_all


def get_colbert_score(a_test_rep, train_representations, maxsim_metric='cos'): #maxsim_metric: cosine, dot
    if maxsim_metric=='cos':                
        test_score = torch.sum(torch.max(torch.div(
                                                    matmul(a_test_rep, train_representations.transpose(1,2)),
                                                    torch.mul(torch.norm(a_test_rep,dim=1).unsqueeze(0).unsqueeze(-1),
                                                            torch.norm(train_representations,dim=2).unsqueeze(1))
                                                ), dim=2).values, dim=1)

        maxsim_score=torch.max(test_score)
        mean_maxsim_score=torch.mean(test_score)
        return maxsim_score, mean_maxsim_score

    elif maxsim_metric=='dot':
        pass

def divide_cal(test_rep_chunk, train_representations, train_neighbor_index, test_idx, coreSet, maxsim_metric='cos'):
    test_rep_chunk_cuda = test_rep_chunk.cuda()
    test_scores_log_chunk = torch.Tensor([]).to(test_rep_chunk_cuda.device)
    test_mean_coreSet_scores_chunk = torch.Tensor([]).to(test_rep_chunk_cuda.device)
    train_representations=train_representations.cuda()

    for a_test in test_rep_chunk_cuda:
        if coreSet==0:
            maxsim_score, mean_coreSet_score =get_colbert_score(a_test, train_representations, maxsim_metric=maxsim_metric)
            # if test_idx==0:
            #     print(f'모든 train data와 비교,{train_representations.shape[0]}')
        else:
            maxsim_score, mean_coreSet_score =get_colbert_score(a_test, train_representations[train_neighbor_index[test_idx]], maxsim_metric=maxsim_metric)
            # if test_idx==0:
            #     print(f'coreSet만 비교,{train_representations[train_neighbor_index[test_idx]].shape[0]}')
        test_scores_log_chunk = torch.cat((test_scores_log_chunk, 
                                    maxsim_score.unsqueeze(0)), dim=0) 
        test_mean_coreSet_scores_chunk = torch.cat((test_mean_coreSet_scores_chunk,
                                    mean_coreSet_score.unsqueeze(0)), dim=0)
        test_idx+=1

    test_scores_log_chunk=test_scores_log_chunk.detach().cpu().numpy()
    test_mean_coreSet_scores_chunk=test_mean_coreSet_scores_chunk.detach().cpu().numpy()
    return test_scores_log_chunk, test_mean_coreSet_scores_chunk, test_idx

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plm', type=str, default='bert-base-uncased')
    parser.add_argument('--seed', type=int, default=1234, help='random seed (default: 1234)')
   
    #dataset.
    parser.add_argument('--dataset', type=str, default='bgl', help='bgl, tbird, hdfs') 
    parser.add_argument("--sample", help=[0.1, 0.05, 100000], default=1, type=lambda x: int(x) if x.isdigit() else float(x))
    parser.add_argument("--test_size", help="test_size", default=0.2, type=float)
    
    # core set
    parser.add_argument('--coreSet', default=0, type=lambda x: int(x) if x.isdigit() else float(x), help='0:all unique, 1, 1000, 0.1')

    parser.add_argument('--maxsim_metric', type=str, default='cos', help='cos, dot')

    #extra Experiment
    parser.add_argument('--only_cls', default=False, type=str2bool, help='only cls colbert')
    parser.add_argument('--train_ratio', type=float, default=1.0, help='for using exp(train ratio)')
    parser.add_argument("--only_in_test", default=False, type=str2bool, help='only_in_test')
    parser.add_argument('--threshold_function', type=str, default='prc', help='prc, roc')

    args = parser.parse_args()
    set_seed(args.seed)

    # directory setting
    if args.sample != 1:
        root_data_path = os.path.join(os.getcwd(), 'processed_data', f'{args.dataset}_sample_{str(args.sample)}')
        processed_data_path = os.path.join(root_data_path, f'{args.test_size}', f'{args.plm}')

    else:
        root_data_path = os.path.join(os.getcwd(), 'processed_data', f'{args.dataset}')
        processed_data_path = os.path.join(root_data_path, f'{args.test_size}', f'{args.plm}') 

    #save result
    save_path=os.path.join(processed_data_path, 'results')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if args.only_in_test:
        save_path=os.path.join(save_path, f'only_in_test')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    #set experiment name start with data information
    exp_log_file_name = f'{args.dataset}_sample-{str(args.sample)}_trainRatio-{str(args.train_ratio)}'

    #exp setting
    exp_log_file_name = exp_log_file_name+f'_thrSearch-{args.threshold_function}'
    
    if args.only_cls: 
        exp_log_file_name = exp_log_file_name+f'_C-Onlycls-{args.maxsim_metric}'
    else:
        exp_log_file_name = exp_log_file_name+f'_C-Wcls-{args.maxsim_metric}'

    exp_log_file_name = exp_log_file_name+f'_coreSet-{str(args.coreSet)}'

    if args.threshold_function == 'roc':
        threshold_function=get_threshold_roc
    elif args.threshold_function == 'prc':
        threshold_function=get_threshold_prc

    if 'hdfs' in args.dataset:
        result_name='(session)'
    else:
        result_name='(all_each)'

    if os.path.exists(os.path.join(save_path, f'{exp_log_file_name}_dict.json')):
        # end of experiment
        print(f'{exp_log_file_name} is already exist')
        sys.exit()

    # load preprocessed data
    # train, val : all normal data
    train_representations = load_pickle(os.path.join(processed_data_path,'train_representations'))
    test_label = load_pickle(os.path.join(processed_data_path,'test_label'))
    test_representations = load_pickle(os.path.join(processed_data_path,'test_representations'))
    test_unique_lookup_table = load_pickle(os.path.join(processed_data_path,'test_unique_lookup_table'))  

    if args.train_ratio != 1:
        print(f'original unique train size: {train_representations.shape}')
        print(f'train_ratio: {args.train_ratio}')
        train_unique_lookup_table=load_pickle(os.path.join(processed_data_path,'train_unique_lookup_table'))        
        sampled_train = np.random.choice(train_unique_lookup_table.shape[0], int(train_unique_lookup_table.shape[0]*args.train_ratio), replace=False)
        train_representations=train_representations[np.unique(train_unique_lookup_table[sampled_train]),:,:]
        print(f'sampled_unique_train_size: {train_representations.shape}')

    if args.only_in_test:
        test_label['lookup_table']=test_unique_lookup_table
        test_label['label']=test_label['label'].astype(int)
        print('only_in_test')
        #compare train_representations and test_representations to find only in test
        # to easy calcuration compare only cls
        train_representations_cls=train_representations[:,0,:].numpy()
        test_representations_cls=test_representations[:,0,:].numpy()

        # get only in test
        only_in_test_idx=[]
        for i, cls in tqdm(enumerate(test_representations_cls.tolist()), desc='only_in_test', total=len(test_representations_cls.tolist())):
            if cls not in train_representations_cls.tolist():
                only_in_test_idx.append(i)

        only_in_test_idx=np.array(only_in_test_idx)
        new_idx_dict={}
        for i in range(len(only_in_test_idx)):
            new_idx_dict[only_in_test_idx[i]]=np.arange(0, len(only_in_test_idx))[i]

        only_test_test_label=test_label[test_label['lookup_table'].isin(only_in_test_idx)].reset_index(drop=True)
        only_test_test_label['lookup_table']=only_test_test_label['lookup_table'].map(new_idx_dict)
        
        test_unique_lookup_table=only_test_test_label['lookup_table'].values
        test_label=only_test_test_label[['timestamp','label']]
        test_representations=test_representations[only_in_test_idx]
        print(f'only_in_test: {test_representations.shape}')


    exp_log_file_name=exp_log_file_name.split('_')
    exp_log_file_name[2]=exp_log_file_name[2]+'-'+str(train_representations.shape[0])

    exp_log_file_name='_'.join(exp_log_file_name)
    
    # if coreSet is nor integer, then it is ratio
    # check args.coreSet is ratio or not
    if (args.coreSet > 0) and (args.coreSet < 1):
        coreSet = int(train_representations.shape[0]*args.coreSet)
        coreSet = max(coreSet, 1)
        print(f'{args.coreSet} = {coreSet}')
        exp_log_file_name='-'.join(exp_log_file_name.split('-')[:-1])
        exp_log_file_name = exp_log_file_name+f'-{args.coreSet}-{coreSet}'
    elif args.coreSet <= train_representations.shape[0]:
        #여기에 coreSet=0인 경우도 포함됨
        coreSet = int(args.coreSet)
    else:
        coreSet = int(train_representations.shape[0])
        print(f'train_representations.shape[0] < coreSet: {train_representations.shape[0]} < {args.coreSet}')
        print('use all unique_train')
    
    if train_representations.shape[0] < coreSet:
        exp_log_file_name='-'.join(exp_log_file_name.split('-')[:-1])
        exp_log_file_name = exp_log_file_name+f'-{train_representations.shape[0]}'

    if os.path.exists(os.path.join(save_path, f'{exp_log_file_name}_dict.json')):
        # end of experiment
        print(f'{exp_log_file_name} is already exist')
        sys.exit()
        
    with open(os.path.join(save_path, f'{exp_log_file_name}.txt'), 'w') as f:
        sys.stdout = f
        
        # get label
        # new_label from unique to all log
        test_label['lookup_table']=test_unique_lookup_table
        test_label['label']=test_label['label'].astype(int)

        #fit knn with unique training data 
        time_for_get_coreSet=time.time()
        if coreSet==0:
            #use all unique_train & 여기선 그냥 3
            knn_cuml_cls = NearestNeighbors(n_neighbors=1)
        else:
            # 앞에서 coreSet이 전체 보다 큰 경우에는 전체를 사용하도록 설정 
            knn_cuml_cls = NearestNeighbors(n_neighbors=coreSet)
        knn_cuml_cls.fit(train_representations[:,0,:].numpy())

        knn_D, train_neighbor_index = knn_cuml_cls.kneighbors(test_representations[:,0,:].numpy())
        time_for_get_coreSet=time.time()-time_for_get_coreSet

        del knn_cuml_cls

        # check the score is ordered by the score
        if (np.sort(knn_D[10])==knn_D[10]).all():
            print('score is ordered by the score')
        else:
            print('*'*50)
            print('score is not ordered by the score')
            print('*'*50)

        # make D of original test data by using test_unique_lookup_table
        knn_D = knn_D[:,0]
        knn_pred, knn_time=get_threshold_pred_distance(knn_D, test_label, desc='make D_ori using lookup')

        if args.only_cls:
            train_representations=train_representations[:,0,:].unsqueeze(1)
            test_representations=test_representations[:,0,:].unsqueeze(1)
            print('colbert with only cls')
            print('*'*50)
        
        print('='*50)
        print('start calculating colbert score')

        test_scores_log = np.array([])
        test_mean_coreSet_scores = np.array([])
        num_chunk = 100
        print(f'train: {train_representations.shape}, test: {test_representations.shape}')

        test_chunks = torch.chunk(test_representations, num_chunk, dim=0)
        print(f'num_chunk: {num_chunk}, num_each_chunk: {test_chunks[0].shape[0]}')

        time_for_cal_maxsim=time.time()
        test_idx=0
        for i in tqdm(range(num_chunk), desc='colbert score by chunk'):
            if (len(test_chunks) != num_chunk) and (i >= len(test_chunks)):
                print(f'{i}th chunk is not exist: number of unique is less than num_chunk')
                break
            test_scores_log_chunk, test_mean_coreSet_scores_chunk, new_test_idx = divide_cal(
                test_rep_chunk=test_chunks[i], test_idx=test_idx, train_representations=train_representations, 
                train_neighbor_index=train_neighbor_index, coreSet=coreSet, maxsim_metric=args.maxsim_metric)
            test_idx=new_test_idx
            
            test_scores_log = np.concatenate((test_scores_log, test_scores_log_chunk), axis=0)
            test_mean_coreSet_scores = np.concatenate((test_mean_coreSet_scores, test_mean_coreSet_scores_chunk), axis=0)

        del test_chunks

        test_scores_log=-test_scores_log+np.max(test_scores_log)
        test_mean_coreSet_scores=-test_mean_coreSet_scores+np.max(test_mean_coreSet_scores)
        time_for_cal_maxsim=time.time()-time_for_cal_maxsim

        save_pickle(test_scores_log, os.path.join(save_path, f'{exp_log_file_name}_test_scores_log'))
        save_pickle(test_mean_coreSet_scores, os.path.join(save_path, f'{exp_log_file_name}_test_mean_coreSet_scores'))

        # for maxsim_ori version: main result
        maxsim_pred, time_for_get_adscore_for_all = get_threshold_pred_distance(test_scores_log, test_label, desc='make maxsim_ori using lookup')

        # for mean_coreSet_score version     
        mean_coreSet_maxsim_pred, mean_coreSet_time = get_threshold_pred_distance(test_mean_coreSet_scores, test_label, desc='make mean maxsim using lookup')


        print('='*50)
        print('save times')
        print('time_for_get_coreSet:', time_for_get_coreSet)
        print('time_for_cal_maxsim:', time_for_cal_maxsim)
        print('time_for_get_adscore_for_all:', time_for_get_adscore_for_all)
        print('='*50)
        time_list=(time_for_get_coreSet,time_for_cal_maxsim, time_for_get_adscore_for_all)

        print('-'*50)
        print('K=1')
        results_df=get_detection_score(test_label['label'], knn_pred,time_list, exp_name = f'K=1{result_name}', result_df=None)
        print('='*50)
        print('only ColBERT by all test')
        results_df=get_detection_score(test_label['label'], maxsim_pred,time_list, exp_name = f'ColBERT{result_name}', result_df=results_df)
        print('='*50)
        print('mean_coreSet_score version')   
        results_df=get_detection_score(test_label['label'], mean_coreSet_maxsim_pred,time_list, exp_name = f'mean ColBERT{result_name}', result_df=results_df)
    # Restore the standard output
    sys.stdout = sys.__stdout__

    # Close the file object
    f.close()

    #save result
    results_df.to_csv(os.path.join(save_path, f'{exp_log_file_name}_df.csv'))
    #also save result_df as dict
    result_dict={f'{exp_log_file_name}':results_df.to_dict()}

    with open(os.path.join(save_path, f'{exp_log_file_name}_dict.json'), 'w') as f:
        json.dump(result_dict, f, indent=4)
        