import os
import json
import time
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from model import resnet34

# 给图片添加高斯噪声
def Gaussnoise_func(img, mean=0, var=0.005):
    img = np.asarray(img)
    noise = np.random.normal(mean, var ** 0.5, np.shape(img))     #产生高斯噪声
    img = img/255
    out = img + noise
    out = np.clip(out, 0, 1)
    out = Image.fromarray(np.uint8(out*255))
    return out


def main():
    start = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    # 指向需要遍历预测的图像文件夹
    imgs_root = "./dataset/test"
    assert os.path.exists(imgs_root), f"file: '{imgs_root}' dose not exist."
    # 读取指定文件夹下所有jpg图像路径
    img_path_list = [os.path.join(imgs_root, i) for i in os.listdir(imgs_root) if i.endswith(".jpg")]

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), f"file: '{json_path}' dose not exist."

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    model = resnet34(num_classes=4).to(device)

    # load model weights
    weights_path = "resNet34_noise50.pth"
    assert os.path.exists(weights_path), f"file: '{weights_path}' dose not exist."
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # prediction
    model.eval()
    batch_size = 13  # 每次预测时将多少张图片打包成一个batch
    predict_res = []  # 存放预测结果
    with torch.no_grad():
        for ids in range(0, len(img_path_list) // batch_size):
            img_list = []
            for img_path in img_path_list[ids * batch_size: (ids + 1) * batch_size]:
                assert os.path.exists(img_path), f"file: '{img_path}' dose not exist."
                img = Image.open(img_path)
                # img = Gaussnoise_func(img,0,0.01)   # 添加高斯噪声
                img = data_transform(img)
                img_list.append(img)

            # batch img
            # 将img_list列表中的所有图像打包成一个batch
            batch_img = torch.stack(img_list, dim=0)
            # predict class
            output = model(batch_img.to(device)).cpu()
            predict = torch.softmax(output, dim=1)
            probs, classes = torch.max(predict, dim=1)

            for idx, (pro, cla) in enumerate(zip(probs, classes)):
                print("image: {}  class: {}  prob: {:.3}".format(img_path_list[ids * batch_size + idx],
                                                                 str(cla.numpy()),
                                                                 pro.numpy()))
                predict_res.append(str(cla.numpy()))

    # 计算测试集正确率和运行时间
    right_num = 0   # 正确预测数
    filename = 'sample_submit.csv'
    with open(filename,'rt') as raw_data:
        for i,row in enumerate(raw_data):
            row = row.strip('\n')
            row = row.split(',')
            if row[1] == predict_res[i]:
                right_num = right_num+1
    end = time.time()
    print("Acc: {%.3f} pro_time: {%.3f}",right_num/len(img_path_list),end-start)

if __name__ == '__main__':
    main()
