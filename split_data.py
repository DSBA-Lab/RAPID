import sys
import os
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
import json

import re
from collections import defaultdict

import random
from sklearn.model_selection import train_test_split

import parmap
import multiprocessing
num_processors = multiprocessing.cpu_count()

from utils import str2bool, set_seed, get_parsed_log, get_unique_log, label_parsed_log

def save_processed_log(data, path, need_newline=False):
    if not need_newline:
        with open(path, 'w') as f:
            for log in data:
                f.write(log)
    else:
        with open(path, 'w') as f:
            for log in data:
                f.write(log)
                f.write('\n')

if __name__ == '__main__':
    set_seed(1234)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help=["hdfs", "bgl", "tbird"], default="tbird")
    parser.add_argument("--shuffle", help="shuffle data", default=True, type=str2bool)
    parser.add_argument("--sample", help=[0.1, 0.05], default=1, type=lambda x: int(x) if x.isdigit() else float(x))
    parser.add_argument("--test_size", help="test_size", default=0.2, type=float)
    args = parser.parse_args()
    current_dir = os.path.dirname(os.path.abspath(__file__))

    if args.dataset == "bgl":
        data_dir = os.path.join(current_dir, 'dataset', 'bgl')
        log_file = "BGL.log"
        output_dir = os.path.join(current_dir, 'processed_data', f'bgl')
    elif args.dataset == "tbird":
        data_dir = os.path.join(current_dir, 'dataset', 'tbird')
        log_file = "Thunderbird.log"
        if args.sample != 1:
            output_dir = os.path.join(current_dir, 'processed_data', f'tbird_sample_{str(args.sample)}')
        else:
            output_dir = os.path.join(current_dir, 'processed_data', f'tbird')
    
    elif args.dataset == "hdfs":
        # we don't split hdfs dataset with this code
        data_dir = os.path.join(current_dir, 'dataset', 'hdfs')
        output_dir = os.path.join(current_dir, 'processed_data', f'hdfs')

        log_file = "HDFS.log"
        blk_label_file = os.path.join(data_dir,"anomaly_label.csv")     
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # load dataset and get normal & abnormal
    if os.path.exists(os.path.join(output_dir, f'train_{args.test_size}')) and os.path.exists(os.path.join(output_dir, f'test_{args.test_size}')):
        print("Already split dataset")
        sys.exit()  


    print("Split dataset")
    if args.dataset != "hdfs":
        #open data_dir + log_file
        with open(os.path.join(data_dir, log_file), 'r', errors='ignore') as f:
            labels = []
            data=[]
            normal_data = []
            abnormal_data = []
            idx = 0
            for line in tqdm(f, desc='get data'):
                labels.append(line.split()[0] != '-')
                if labels[-1]:
                    abnormal_data.append(line)
                else:
                    normal_data.append(line)
                data.append(line)
                idx += 1
    else:
        #hdfs
        if os.path.exists(os.path.join(data_dir, 'preprocessed_data_df.csv')):
            print("preprocessed hdfs:preprocessed_data_df.csv exists")
            import ast
            def str_to_list(s):
                return ast.literal_eval(s)
            data_df=pd.read_csv(os.path.join(data_dir, 'preprocessed_data_df.csv'), converters={'Raw':str_to_list,'labeled_Raw': str_to_list, 'parsed_unique_log': str_to_list})

        else:
            print("preprocess hdfs:preprocessed_data_df.csv")
            
            with open(os.path.join(data_dir, log_file), 'r', errors='ignore') as f:
                data=[]
                for line in tqdm(f, total=11175629, desc='get data'):
                    data.append(line)
            #list to dataframe
            df = pd.DataFrame(data, columns=['Raw']) #raw data

            data_dict = defaultdict(list) #preserve insertion order of items
            for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc='find blk_id'):
                blkId_list = re.findall(r'(blk_-?\d+)', row['Raw']) #find all block ids in log Content
                blkId_set = set(blkId_list)
                for blk_Id in blkId_set:
                    data_dict[blk_Id].append(row["Raw"])

            data_df = pd.DataFrame(list(data_dict.items()), columns=['BlockId', 'Raw'])
            # make dataframe:blk_df to dict:blk_label_dict
            blk_df=pd.read_csv(blk_label_file)
            blk_label_dict = dict(zip(blk_df.BlockId, blk_df.Label))
            blk_label_dict = {k: 1 if v == 'Anomaly' else 0 for k, v in blk_label_dict.items()}

            data_df["Label"] = data_df["BlockId"].apply(lambda x: blk_label_dict.get(x)) #add label to the sequence of each blockid
            
            parsed_unique_log=parmap.map(get_parsed_log, data_df['Raw'], pm_pbar=True, pm_processes=num_processors-2)
            parsed_unique_log=parmap.map(get_unique_log, parsed_unique_log, pm_pbar=True, pm_processes=num_processors-2)

            data_df['parsed_unique_log']=parsed_unique_log
            data_df=label_parsed_log(data_df)
            data_df.to_csv(os.path.join(data_dir, 'preprocessed_data_df.csv'), index=False)

        normal_data = data_df[data_df['Label'] == 0]['labeled_parsed_unique_concat'].tolist()
        abnormal_data = data_df[data_df['Label'] == 1]['labeled_parsed_unique_concat'].tolist()

    #split dataset
    if args.sample != 1:
        # sample == float or int
        # sample data with max_num
        #get normal, abnormal data ratio and get # of each max
        ab_ratio=len(abnormal_data)/(len(normal_data)+len(abnormal_data))

        if isinstance(args.sample, float):
            normal_data = random.sample(normal_data, int(len(normal_data)*args.sample))
            abnormal_data = random.sample(abnormal_data, int(len(abnormal_data)*args.sample))
            normal_train_val, normal_test = train_test_split(normal_data, test_size=args.test_size, random_state=1234, shuffle=args.shuffle)

        elif isinstance(args.sample, int):
            print("sample data with specific integer num")
            normal_data = random.sample(normal_data, int(args.sample*(1-ab_ratio)))
            abnormal_data = random.sample(abnormal_data, int(args.sample*ab_ratio))
            normal_train_val, normal_test = train_test_split(normal_data, test_size=args.test_size, random_state=1234, shuffle=args.shuffle)
    else:
        normal_train_val, normal_test = train_test_split(normal_data, test_size=args.test_size, random_state=1234, shuffle=args.shuffle)

    test = normal_test + abnormal_data

    if args.dataset == "hdfs":
        need_newline=True
    else:
        need_newline=False

    save_processed_log(normal_train_val, os.path.join(output_dir, f'train_{args.test_size}'), need_newline)
    save_processed_log(test, os.path.join(output_dir, f'test_{args.test_size}'),need_newline) 

    data_size={}
    data_size['train_normal']=len(normal_train_val)
    data_size['test_normal']=len(normal_test)
    data_size['test_abnormal']=len(abnormal_data)
    with open(os.path.join(output_dir, f'data_size_dict_{args.test_size}.json'), 'w') as f:
        json.dump(data_size, f, indent=4)
