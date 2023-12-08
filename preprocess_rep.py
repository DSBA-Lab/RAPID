import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import pandas as pd
import numpy as np
import argparse 

import torch
from torch.nn import DataParallel

from transformers import (
    BertModel,
    BertTokenizer,
    AutoModel,
    AutoTokenizer,
)

import parmap
import multiprocessing
num_processors = multiprocessing.cpu_count()

#for time check
from tqdm import tqdm
import time
from utils import str2bool, set_seed, load_pickle, save_pickle, normalizeString, bgl_regex, tb_regex, hdfs_regex

def preprocess(sentence, flag, dataset):
    if "bgl" in dataset:
        timestamp, sentence = bgl_regex(sentence)
    elif "tbird" in dataset:
        timestamp, sentence = tb_regex(sentence)
    elif "hdfs" in dataset:
        timestamp, sentence = hdfs_regex(sentence)#hdfs: timestamp=block_id

    if flag=='test':
        if sentence.split()[0] == '-':
            test_label=0
        else:
            test_label=1
            if ("bgl" in dataset) or ("tbird" in dataset): #bgl, tbird: abnormal case has error category, don't need this
                sentence = " ".join(sentence.split()[1:])
    sentence = normalizeString(sentence)
    if ("bgl" in dataset) or ("tbird" in dataset):
        sentence = " ".join(sentence.split()[3:]) #useless part remove
    elif "hdfs" in dataset:
        sentence = " ".join(sentence.split()[1:])

    if flag=='test':
        return timestamp,sentence,test_label
    else:
        return timestamp,sentence

def get_time_data(raw_data, flag, args): 
    #multiprocessing
    timestamp_representation_testLabel =np.array(parmap.map(preprocess, raw_data, flag, args.dataset, pm_pbar=True, pm_processes=num_processors-2))
    timestamps=timestamp_representation_testLabel[:,0]
    sentence_list=timestamp_representation_testLabel[:,1]
    if flag=='test':
        test_labels=timestamp_representation_testLabel[:,2]
    if flag=='test':
        return timestamps, sentence_list, test_labels
    else:
        return timestamps, sentence_list
    
def get_representation(model, sentence_list, batch_size=20000, max_length=512, pooling_strategy='all'):
    all_representations=torch.tensor([], dtype=torch.float32)
    for batch_sentences in tqdm(batch(sentence_list, batch_size), total=len(sentence_list)//batch_size+1):
        tokens=tokenizer(batch_sentences, add_special_tokens=True, return_tensors='pt', padding='max_length', max_length=max_length, truncation=True).to(device) #max_lenth=128, truncation=True하지 않으면 특정한 문장이 299 길이가 나오나봐

        with torch.no_grad():
            if pooling_strategy=='all':
                representations=model(**tokens).last_hidden_state.detach().cpu()

        all_representations=torch.cat((all_representations, representations), dim=0)
    return all_representations

def batch(iterable, n = 1):
   current_batch = []
   for item in iterable:
       current_batch.append(item)
       if len(current_batch) == n:
           yield current_batch
           current_batch = []
   if current_batch:
       yield current_batch

def get_unique_values_table(sentence_in_list, unique_sentence_list):
    indices = np.where(unique_sentence_list == sentence_in_list)[0][0]
    return indices

def process(model, flag, args):
    # flag: train, validation, test
    if not args.need_split:
        if os.path.exists(os.path.join(temp_data_path,f'{flag}_timestamps.pkl')):
            print(f'{flag}은 이미 존재합니다.')
            timestamps = load_pickle(os.path.join(temp_data_path,f'{flag}_timestamps'))
            sentence_list = load_pickle(os.path.join(temp_data_path,f'{flag}_sentence_list'))
        else:
            time_data=get_time_data(globals()[f'raw_{flag}'],flag, args)
            timestamps=time_data[0]
            sentence_list=time_data[1]
            save_pickle(timestamps, os.path.join(temp_data_path,f'{flag}_timestamps'))
            save_pickle(sentence_list, os.path.join(temp_data_path,f'{flag}_sentence_list'))
            if flag == 'test':
                test_label=time_data[2]
                save_pickle(test_label, os.path.join(temp_data_path,f'test_label'))

        if flag =='test':
            test_label_df=pd.DataFrame({'timestamp':timestamps,'label':test_label})
            save_pickle(test_label_df, os.path.join(temp_data_path,'test_label'))
        print('timestamps, sentence_list 완료')

        sentence_array = np.array(sentence_list)
        unique_sentence = np.unique(sentence_array)

        unique_lookup_table= np.array(parmap.map(get_unique_values_table, sentence_array, unique_sentence, pm_pbar=True, pm_processes=num_processors-2))
        print('unique_lookup_table 완료')
        unique_sentence=unique_sentence.tolist()
        save_pickle(unique_lookup_table, os.path.join(temp_data_path,f'{flag}_unique_lookup_table'))
        save_pickle(unique_sentence, os.path.join(temp_data_path,f'{flag}_unique_sentence'))

        if os.path.exists(os.path.join(temp_data_path,f'{flag}_representations.pkl')):
            print(f'{flag}은 이미 존재합니다.')
        else:
            representations=get_representation(model, unique_sentence, batch_size=args.batch_size, max_length = args.max_token_len, pooling_strategy=args.pooling_strategy)
            save_pickle(representations, os.path.join(temp_data_path,f'{flag}_representations'))
        print('representations 완료')
    else:
        def save_sentence_list(flag, args):
            for i in tqdm(range(args.split_num), desc=f'{flag} time_data split'):
                print(f'{flag} time_data {i}번째 get_time_data')
                if i == args.split_num-1:
                    time_data=get_time_data(globals()[f'raw_{flag}'][int(len(globals()[f'raw_{flag}'])/args.split_num)*i:],flag, args)
                else:
                    time_data=get_time_data(globals()[f'raw_{flag}'][int(len(globals()[f'raw_{flag}'])/args.split_num)*i:int(len(globals()[f'raw_{flag}'])/args.split_num)*(i+1)],flag, args)
                
                timestamps=time_data[0]
                sentence_list=time_data[1]
                if flag == 'test':
                    test_label=time_data[2]
                    save_pickle(test_label, os.path.join(temp_data_path,f'test_label_{i}'))

                    test_label_df=pd.DataFrame({'timestamp':timestamps,'label':test_label})
                    save_pickle(test_label_df, os.path.join(temp_data_path,f'test_label_{i}'))

                save_pickle(timestamps, os.path.join(temp_data_path,f'{flag}_timestamps_{i}'))
                save_pickle(sentence_list, os.path.join(temp_data_path,f'{flag}_sentence_list_{i}'))
                print(f'{flag}_timestamps_{i}, {flag}_sentence_list_{i} 완료')

        print(f'{args.split_num} 만큼 나누어 전처리 수행 후 다시 결합해 rep 저장합니다.')

        save_sentence_list(flag, args)
        del globals()[f'raw_{flag}']
        print('조각을 다시 모아 전체 데이터 생성')
        #label
        if flag == 'test':
            test_label_df=pd.DataFrame({'timestamp':[],'label':[]})
            for i in range(args.split_num):
                test_label_df=pd.concat([test_label_df, load_pickle(os.path.join(temp_data_path,f'test_label_{i}'))], axis=0, ignore_index=True)
            save_pickle(test_label_df, os.path.join(temp_data_path,'test_label'))
            print('test_label 완료')

        sentence_list=np.array([])
        for i in range(args.split_num):
            # concat all sentence_list
            sentence_list=np.append(sentence_list, load_pickle(os.path.join(temp_data_path,f'{flag}_sentence_list_{i}')))
        
        print('unique시작')
        sentence_array = np.array(sentence_list)
        unique_sentence = np.unique(sentence_array)
        print(f'{flag} unique sentence len :{unique_sentence.shape}')
        
        unique_lookup_table= np.array(parmap.map(get_unique_values_table, sentence_array, unique_sentence, pm_pbar=True, pm_processes=num_processors-2))
        unique_sentence=unique_sentence.tolist()

        save_pickle(unique_lookup_table, os.path.join(temp_data_path,f'{flag}_unique_lookup_table'))
        save_pickle(unique_sentence, os.path.join(temp_data_path,f'{flag}_unique_sentence'))
        print('unique_lookup_table 완료')
            
        print('get_representation 시작')
        representations=get_representation(model, unique_sentence, batch_size=args.batch_size, max_length = args.max_token_len, pooling_strategy=args.pooling_strategy)
        save_pickle(representations, os.path.join(temp_data_path,f'{flag}_representations'))
        print(f'{flag}_representations 완료')
                    

if __name__ == '__main__':
    set_seed(1234)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help=["hdfs", "bgl", "tbird"], default="tbird")
    parser.add_argument("--sample", help=[0.1, 0.05, 100000], default=1, type=lambda x: int(x) if x.isdigit() else float(x))
    parser.add_argument("--shuffle", help=["True", "False"], default=True, type=str2bool)
    parser.add_argument("--test_size", help="test_size", default=0.2, type=float)
    parser.add_argument("--need_split", help=["True", "False"], default=False, type=str2bool)
    parser.add_argument("--split_num", help=[5, 10], default=10, type=int)
    parser.add_argument('--plm', type=str, default='bert-base-uncased')
    parser.add_argument("--batch_size", default=8192, type=int)
    parser.add_argument("--max_token_len", default=128, type=int, help='bgl, tbrid:128, hdfs:512')
    parser.add_argument("--pooling_strategy", help=["cls", "mean", 'all'], default="all")

    args = parser.parse_args()

    if args.sample != 1:
        output_path=os.path.join(os.getcwd(), 'processed_data' ,f'{args.dataset}_sample_{str(args.sample)}')
        raw_file_path=os.path.join(os.getcwd(),'processed_data', f'{args.dataset}_sample_{str(args.sample)}')
    else:
        output_path=os.path.join(os.getcwd(), 'processed_data' ,f'{args.dataset}')
        raw_file_path=os.path.join(os.getcwd(),'processed_data', f'{args.dataset}')

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    #save temp files
    if args.plm == 'pretrained_bgl':
        temp_data_path=os.path.join(output_path, f'{args.test_size}','temp_bgl')
    else :
        temp_data_path=os.path.join(output_path, f'{args.test_size}',f'{args.plm}')

    if not os.path.exists(temp_data_path):
        os.makedirs(temp_data_path)
        
    if os.path.exists(os.path.join(temp_data_path, 'rep_time.txt')):
        print('이미 전처리 완료된 데이터입니다.')
        sys.exit()

    #device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if (device.type == 'cuda') and (torch.cuda.device_count() > 1):
        print("Let's use", torch.cuda.device_count(), "GPUs!")

    # 일반 사전학습 bert 이용하기
    if args.plm == 'pretrained_bgl':
        model = BertModel.from_pretrained("./final_bert_model")
    else :
        model = AutoModel.from_pretrained(args.plm)
    model= DataParallel(model)
    model.to(device)
    model.eval()
    if args.plm == 'pretrained_bgl':
        tokenizer = BertTokenizer.from_pretrained("./tokenizer/BGL_lanobert-vocab.txt")
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.plm)

    # open file
    with open(os.path.join(raw_file_path,f'train_{args.test_size}'), 'r', encoding='utf-8') as f:
        raw_train = f.readlines()

    with open(os.path.join(raw_file_path,f'test_{args.test_size}'), 'r', encoding='utf-8') as f:
        raw_test = f.readlines()

    # time check
    start_time = time.time()
    print('get train data & representation')
    process(model, 'train', args)

    train_end_time = time.time()
    print(f'train time: {train_end_time-start_time}')

    print('get test data & representation')
    process(model, 'test', args)
    test_end_time = time.time()
    print(f'test time: {test_end_time-train_end_time}')

    print('train_test time: ', test_end_time-start_time)
   
    with open(os.path.join(temp_data_path, f'rep_time.txt'), 'w') as f:
        sys.stdout = f
        print(f'train time: {train_end_time-start_time}')
        print(f'test time: {test_end_time-train_end_time}')
        print(f'train_test time: {test_end_time-start_time}')

    # Restore the standard output
    sys.stdout = sys.__stdout__

    # Close the file object
    f.close()
