import argparse
import os
import random
import numpy as np
import torch
import pickle

import re
import unicodedata
from datetime import datetime

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def load_pickle(path):
    with open(path+'.pkl', 'rb') as f:
        data = pickle.load(f)
    return data

def save_pickle(data, path):
    with open(path+'.pkl', 'wb') as f:
        pickle.dump(data, f)

def unicodeToAscii(s):
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"[^a-zA-Z<>]+", r" ", s) # only english, del: num, special char
    s = re.sub(r"\s+", r" ", s).strip() # del white space
    return s

#for dataset preprocessing
# for bgl
def bgl_regex(log):
    date_time_regex = re.compile(
        "\d{1,4}\-\d{1,2}\-\d{1,2}-\d{1,2}.\d{1,2}.\d{1,2}.\d{1,6}"
    )
    date_regex = re.compile("\d{1,4}\.\d{1,2}\.\d{1,2}")
    ip_regex = re.compile("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d{1,5})?")
    server_regex = re.compile("\S+(?=.*[0-9])(?=.*[a-zA-Z])(?=[:]+)\S+")
    server_regex2 = re.compile("\S+(?=.*[0-9])(?=.*[a-zA-Z])(?=[-])\S+")
    ecid_regex = re.compile("[A-Z0-9]{28}")
    serial_regex = re.compile("[a-zA-Z0-9]{48}")
    memory_regex = re.compile("0[xX][0-9a-fA-F]\S+")
    path_regex = re.compile(".\S+(?=.[0-9a-zA-Z])(?=[/]).\S+")
    iar_regex = re.compile("[0-9a-fA-F]{8}")
    num_regex = re.compile("(\d+)")
    
    timestamp = (np.array([str(datetime.strptime(re.findall(date_time_regex, log)[0],'%Y-%m-%d-%H.%M.%S.%f'))])).item()
    tmp = re.sub(date_time_regex, " TIME ", log)
    tmp = re.sub(ip_regex, " IP ", tmp)
    tmp = re.sub(date_regex, " TIME ", tmp)
    tmp = re.sub(path_regex, " PATH ", tmp)
    tmp = re.sub(server_regex, " SERVER ", tmp)
    tmp = re.sub(server_regex2, " SERVER ", tmp)
    tmp = re.sub(ecid_regex, " ECID ", tmp)
    tmp = re.sub(serial_regex, " SERIAL ", tmp)
    tmp = re.sub(memory_regex, " MEMORY ", tmp)
    tmp = re.sub(iar_regex, " IAR ", tmp)
    tmp = re.sub(num_regex, " NUM ", tmp)
    return timestamp, tmp

def tb_regex(log):
    date_regex = re.compile("\d{2,4}\.\d{1,2}\.\d{1,2}\s")
    date_regex2 = re.compile(
        "(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{1,2})\s+"
    )
    time_regex = re.compile("\d{1,2}\:\d{1,2}\:\d{1,2}")
    id_regex = re.compile(r"DATE\s.*\sDATE")

    account_regex = re.compile("(\w+[\w\.]*)@(\w+[\w\.]*)\-(\w+[\w\.]*)")
    account_regex2 = re.compile("(\w+[\w\.]*)@(\w+[\w\.]*)")
    account_regex3 = re.compile(r"TIME\s\S+")

    dir_regex = re.compile(r'[a-zA-Z0-9_\-\.\/]+\/[a-zA-Z0-9_\-\.\/]+\/[a-zA-Z0-9_\-\.\/]*') # /로 안시작하고 /가 두겹이상인 경우
    dir_regex2 = re.compile(r'\/[a-zA-Z0-9_\-\.\/]+\/[a-zA-Z0-9_\-\.\/]*') # /로 시작하고 /가 한겹인 경우
    iar_regex = re.compile("[0-9a-fA-F]{10}")
    ip_regex = re.compile("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d{1,5})?")
    num_regex = re.compile("(\[\d+\])")

    date_time_str=re.findall(date_regex, log)[0]+" "+re.findall(time_regex, log)[0]
    timestamp = (np.array([str(datetime.strptime(date_time_str,'%Y.%m.%d %H:%M:%S'))])).item()
    tmp = re.sub(date_regex, "DATE ", log)
    tmp = re.sub(date_regex2, "DATE ", tmp)
    tmp = re.sub(id_regex, "DATE ID DATE", tmp)
    tmp = re.sub(time_regex, "TIME", tmp)
    tmp = re.sub(account_regex3, "TIME ACCOUNT", tmp) ## TIME / TIME ACCOUNT
    tmp = re.sub(account_regex, "ACCOUNT", tmp)
    tmp = re.sub(account_regex2, "ACCOUNT", tmp)
    tmp = re.sub(dir_regex, " DIR ", tmp)
    tmp = re.sub(dir_regex2, " DIR ", tmp)
    tmp = re.sub(ip_regex, "IP", tmp)
    tmp = re.sub(iar_regex, "IAR", tmp)
    tmp = re.sub(num_regex, " NUM ", tmp)

    return timestamp, tmp

def hdfs_regex(log):
    id_regex = re.compile("blk_.\d+")
    ip_regex = re.compile("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d{1,5})?")
    num_regex = re.compile("\d*\d")

    block_id = re.findall(id_regex, log)[0]
    tmp = re.sub(id_regex, "BLK", log) # already parsed in dataset preprocessing, del block_id
    tmp = re.sub(ip_regex, "IP", tmp)
    tmp = re.sub(num_regex, "NUM", tmp)
    return block_id, tmp

# for hdfs data split
def concat_list_str(row):
    # delete \n & concatenate
    return ' '.join(list(map(lambda x: (x.replace('\n','')),row)))

def add_label_Raw_blk(row):
    blk = concat_list_str(row)
    blk = "- "+blk
    return blk

def get_parsed_log(df_row):
    blk_log=[]
    for i, log in enumerate(df_row):
        parsed=hdfs_regex(' '.join(log.split()[3:]))
        if i ==0:
            blk_log.append(parsed[0])
        blk_log.append(normalizeString(parsed[1]))        
    return blk_log
    
def get_unique_log(df_row):
    return np.unique(df_row).tolist()

def label_parsed_log(data_df):
    data_df['labeled_parsed_unique_concat']=data_df.apply(lambda row: add_label_Raw_blk(row['parsed_unique_log']) if (row['Label'] == 0) else concat_list_str(row['parsed_unique_log']), axis=1)
    return data_df