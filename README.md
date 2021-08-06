## 第五届“达观杯” 基于大规模预训练模型的风险事件标签识别

1. 比赛链接：https://www.datafountain.cn/competitions/512
2. code：https://github.com/YuanWind/daGuanCup
3. 白嫖移动V100 300小时：https://ecloud.10086.cn/home/market/ai
4. 验证集线上线下分数对照：
    | 线下         | 线上       | 备注                                                         |
    | ------------ | ---------- | ------------------------------------------------------------ |
    | 0.5415717674 | 0.54136122 | !python main.py train --train_batch_size 32 --num_train_epochs 20 --eval_batch_size 32 |
    | 0.54418902   | 0.55100852 | !python main.py train --train_batch_size 16 --num_train_epochs 20 --eval_batch_size 64 |
     以上结果为组队之前的提交，只用了训练和测试数据进行语言模型预训练，然后微调分类得到结果
5. 需要下载 chinese-bert-wwm-ext 模型到pretrained目录下，下载地址：https://www.aliyundrive.com/s/eCxtJokoT79
6. data/ori_data/sample_unlabel_data_10000.json数据为从19G压缩无标注数据中抽出来了10000条json数据得到的,下载地址：https://www.aliyundrive.com/s/eCxtJokoT79
7. 未完待续......
    