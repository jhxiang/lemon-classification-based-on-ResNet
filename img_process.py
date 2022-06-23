import os
import random
import shutil

# 注释程序说明：根据标签将柠檬图片分成ABCD四类
# filename = 'train_images.csv'
# raw_path = './dataset/train_images'
# new_path = './dataset/trainset'
# with open(filename,'rt') as raw_data:
# 	for row in raw_data:
# 		row = row.strip('\n')
# 		row = row.split(',')
# 		src = raw_path + '/' + row[0]
# 		print(row[1])
# 		if row[1] == '0':
# 			dst = new_path+'/'+'A'+'/'
# 			shutil.move(src,dst)
# 		elif row[1] == '1':
# 			dst = new_path+'/'+'B'+'/'
# 			shutil.move(src,dst)
# 		elif row[1] == '2':
# 			dst = new_path+'/'+'C'+'/'
# 			shutil.move(src,dst)
# 		elif row[1] == '3':
# 			dst = new_path+'/'+'D'+'/'
# 			shutil.move(src,dst)

# 程序说明：将原来的训练集以8:2比例划分为新训练集和验证集
def make_file(path):
    if os.path.exists(path):
        os.removedirs(path)
        os.makedirs(path)
    else:
        os.makedirs(path)


def main():
    path = r'./dataset'  # 训练集和验证集的存储路径
    par_path = os.path.join(os.getcwd(), 'dataset','trainset')  # 原始数据集路径
    print(par_path)
    label_set = [label_i for label_i in os.listdir(par_path) if os.path.isdir(os.path.join(par_path, label_i))]  # 类别标签
    # 训练集和验证集的比例
    rate = 0.8
    # 创建训练集/测试集文件夹
    train_path = os.path.join(path, 'train')  # 训练集路径
    make_file(train_path)
    val_path = os.path.join(path, 'val')  # 验证集路径
    make_file(val_path)

    for label in label_set:
        # 创建训练集/验证集内的种类文件
        os.mkdir(os.path.join(train_path, label))
        os.mkdir(os.path.join(val_path, label))

        root = os.path.join(par_path, label)
        image_set = os.listdir(root)
        sample_index = random.sample(image_set, k=int(rate * len(image_set)))

        # 分配数据
        for i, j in enumerate(image_set):
            origin_path = os.path.join(root, j)
            tgt_train_path = os.path.join(train_path, label)
            tgt_val_path = os.path.join(val_path, label)
            if j in sample_index:
                shutil.copy(origin_path, tgt_train_path)
            else:
                shutil.copy(origin_path, tgt_val_path)

            if i == len(image_set) - 1:
                print(f'< {label} > processing | {i + 1}/{len(image_set)} |')

    print('Finished !')


if __name__ == '__main__':
    main()


