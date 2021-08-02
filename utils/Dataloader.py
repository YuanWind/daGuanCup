import logging

from torch.utils.data import Dataset


class Instance():
    def __init__(self):
        self.sentence = None
        self.label_1=None
        self.label_2=None
        self.label_12=None
        self.idx=None

class MyDataSet(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]

class Vocab():
    def __init__(self) -> None:
        
        self.id2label=['PAD','UNK']
        self.label2id={}

    def build(self,data):
        """构建词表

        :param data: 训练数据 List，元素类型为 Instance
        :type data: list
        """        
        label=set()
        for inst in data:
            label= label | {inst.label_12}
        self.id2label.extend(sorted(list(label)))
        
        for idx,v in enumerate(self.id2label):
            self.label2id[v]=idx
        assert len(self.id2label)==len(self.label2id)
        logging.info('labels num:{}')
        for k ,v in self.label2id.items():
            logging.info('{}\t{}'.format(v,k))



