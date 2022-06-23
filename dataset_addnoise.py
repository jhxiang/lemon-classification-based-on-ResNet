import matplotlib.pyplot as plt
import numpy as np
import os
import random
from PIL import Image

# 程序说明：扩充数据集，给训练集和验证集图片中随机加入不同标准差的高斯噪声
# 给图片添加高斯噪声函数
def Gaussnoise_func(img, mean=0, var=0.005):
    img = np.asarray(img)
    noise = np.random.normal(mean, var ** 0.5, np.shape(img))     #产生高斯噪声
    img = img/255
    out = img + noise
    out = np.clip(out, 0, 1)
    out = Image.fromarray(np.uint8(out*255))
    return out

def main():
    train_path = os.path.join(os.getcwd(), 'dataset_noise', 'train')  # 含噪声图片的训练集路径
    val_path = os.path.join(os.getcwd(), 'dataset_noise', 'val')  # 含噪声图片的验证集路径
    print(train_path)
    print(val_path)
    label_set = [label_i for label_i in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, label_i))]  # 类别标签

    for label in label_set:
        root = os.path.join(train_path, label)
        image_set = os.listdir(root)
        # 给每张图片随机添加不同标准差的高斯噪声
        for i, j in enumerate(image_set):
            origin_path = os.path.join(root, j)
            var = random.choice([0.01, 0.04, 0.09])
            img = Image.open(origin_path)
            img_n = Gaussnoise_func(img, 0, var)  # 添加高斯噪声
            save_path = root + '/'+ j[0:10] + '_noise' + '.jpg'  # 保存路径为当前文件夹
            img_n.save(save_path)

    for label in label_set:
        root = os.path.join(val_path, label)
        image_set = os.listdir(root)
        # 给每张图片随机添加不同标准差的高斯噪声
        for i, j in enumerate(image_set):
            origin_path = os.path.join(root, j)
            var = random.choice([0.01, 0.04, 0.09])
            img = Image.open(origin_path)
            img_n = Gaussnoise_func(img, 0, var)  # 添加高斯噪声
            save_path = root +'/'+ j[0:10] + '_noise' + '.jpg'  # 保存路径为当前文件夹
            img_n.save(save_path)

if __name__ == '__main__':
    main()