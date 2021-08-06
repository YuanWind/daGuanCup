from collections import Counter
from utils.utils import load_pkl
import matplotlib.pyplot as plt
def analysis():
    train_data=load_pkl('./data/processed/train.pkl')
    dev_data=load_pkl('./data/processed/dev.pkl')
    test_data=load_pkl('./data/processed/test.pkl')
    sentence_len=[]
    for inst in train_data+dev_data+test_data:
        sentence_len.append(len(inst.sentence))
    cnt=Counter(sentence_len).most_common()

    return cnt
cnt=analysis()
x=[]
y=[]
for length,num in cnt:
    x.append(length)
    y.append(num)
plt.bar(x,y)
plt.show()