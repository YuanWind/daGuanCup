import logging
import random
import numpy as np
import torch
import pickle
import json
def set_logger(log_file,to_console=True,to_file=True):
     logger = logging.getLogger()  # 不加名称设置root logger
     logger.setLevel(logging.DEBUG)
     formatter = logging.Formatter(
     '%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
     datefmt='%Y-%m-%d %H:%M:%S')
     # 使用FileHandler输出到文件
     if to_file:
         fh = logging.FileHandler(log_file)
         fh.setLevel(logging.DEBUG)
         fh.setFormatter(formatter)
         logger.addHandler(fh)
     # 使用StreamHandler输出到屏幕
     if to_console:
         ch = logging.StreamHandler()
         ch.setLevel(logging.DEBUG)
         ch.setFormatter(formatter)
         logger.addHandler(ch)
def set_seed(seed):
     random.seed(seed)
     np.random.seed(seed)
     torch.manual_seed(seed)
     if torch.cuda.is_available():
         torch.cuda.manual_seed_all(seed)
def write_pkl(data, f_name):
     with open(f_name, 'wb') as f:
         pickle.dump(data, f)
def load_pkl(f_name):
     with open(f_name, 'rb') as f:
         return pickle.load(f)
def write_json(data, f_name):
     with open(f_name, 'w', encoding='utf-8') as json_file:
         json.dump(data, json_file, ensure_ascii=False, indent=4)
def load_json(f_name):
     with open(f_name, 'r', encoding='utf-8') as fr:
         data = json.load(fr)
     return data