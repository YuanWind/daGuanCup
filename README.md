## 第五届“达观杯” 基于大规模预训练模型的风险事件标签识别

1. 比赛链接：https://www.datafountain.cn/competitions/512

2. 验证集线上线下分数对照：
    | 线下         | 线上       | 备注                                                         |
    | ------------ | ---------- | ------------------------------------------------------------ |
    | 0.5415717674 | 0.54136122 | !python main.py train --train_batch_size 32 --num_train_epochs 20 --eval_batch_size 32 |
    | 0.54418902   | 0.55100852 | !python main.py train --train_batch_size 16 --num_train_epochs 20 --eval_batch_size 64 |
    ​    
    
    当前只用了训练和测试数据进行语言模型预训练，然后微调分类得到结果