import sys
sys.path.extend(['../../','./','../'])
import zipfile
import json
import fire
from tqdm import tqdm
def sample_num(save_file='data/ori_data/unlabel_{}.json',num=100000):
    unlabel_zip=zipfile.ZipFile(r'data/ori_data/datagrand_2021_unlabeled_data.zip','r')
    files=unlabel_zip.infolist()
    unlabel_json=unlabel_zip.open('datagrand_2021_unlabeled_data.json','r')
    with open(save_file.format(num),'wb') as f:
        for i in tqdm(range(num)):
            line_data=unlabel_json.readline()
            f.write(line_data)
if __name__ == '__main__':
    fire.Fire(sample_num)
    