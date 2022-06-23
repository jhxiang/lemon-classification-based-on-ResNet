resNet34-pre.pth:模型预训练权重
resNet34_8.pth:epoch为8时模型权重，其余同理，后缀数字表示epoch
resNet34_noise50.pth:扩充训练集后epoch为50时模型权重
img_process.py:注释掉的部分是根据标签将柠檬图片分成ABCD四类
               其余部分将原来的训练集以8:2比例划分为新训练集和验证集
dataset_addnoise.py:扩充训练集程序(加高斯噪声)
train.py、model.py、predict.py:训练、模型和预测
注：扩充后的数据集放在dataset_noise文件夹下
               