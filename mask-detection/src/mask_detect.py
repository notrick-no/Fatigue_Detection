import os
import zipfile
import random
import json
import paddle
import sys
import numpy as np
from PIL import Image
from PIL import ImageEnhance
import paddle.fluid as fluid
from multiprocessing import cpu_count
import matplotlib.pyplot as plt
'''
参数配置
'''
train_parameters = {
    "input_size": [3, 128, 128],                              #输入图片的shape
    "class_dim": -1,                                          #分类数
    "src_path":"D:\吕良伟毕业论文+PPT+算法程序\吕良伟毕业论文+PPT+算法程序\Fatigue Detector\mask-detection\otherthing\Dataset.zip",#原始数据集路径
    "target_path":"D:\吕良伟毕业论文+PPT+算法程序\吕良伟毕业论文+PPT+算法程序\Fatigue Detector\mask-detection\data",                     #要解压的路径
    "train_list_path": "D:\吕良伟毕业论文+PPT+算法程序\吕良伟毕业论文+PPT+算法程序\Fatigue Detector\mask-detection\datatrain.txt",       #train.txt路径
    "eval_list_path": "D:\吕良伟毕业论文+PPT+算法程序\吕良伟毕业论文+PPT+算法程序\Fatigue Detector\mask-detection\data\eval.txt",         #eval.txt路径
    "readme_path": "D:\吕良伟毕业论文+PPT+算法程序\吕良伟毕业论文+PPT+算法程序\Fatigue Detector\mask-detection\data/readme.json",         #readme.json路径
    "label_dict":{},                                          #标签字典
    "train_batch_size":20,                                    #训练大小
}
def unzip_data(src_path,target_path):
    '''
    解压原始数据集，将src_path路径下的zip包解压至data目录下
    '''
    if(not os.path.isdir(target_path + "maskDetect")):
        z = zipfile.ZipFile(src_path, 'r')
        z.extractall(path=target_path)
        z.close()


def get_data_list(target_path, train_list_path, eval_list_path):
    '''
    生成数据列表
    '''
    # 存放所有类别的信息
    class_detail = []
    # 获取所有类别保存的文件夹名称
    data_list_path = target_path + "maskDetect/"
    class_dirs = os.listdir(data_list_path)
    # 总的图像数量
    all_class_images = 0
    # 存放类别标签
    class_label = 0
    # 存放类别数目
    class_dim = 0
    # 存储要写进eval.txt和train.txt中的内容
    trainer_list = []
    eval_list = []
    # 读取每个类别，['maskimages', 'nomaskimages']
    for class_dir in class_dirs:
        if class_dir != ".DS_Store":
            class_dim += 1
            # 每个类别的信息
            class_detail_list = {}
            eval_sum = 0
            trainer_sum = 0
            # 统计每个类别有多少张图片
            class_sum = 0
            # 获取类别路径
            path = data_list_path + class_dir
            # 获取所有图片
            img_paths = os.listdir(path)
            for img_path in img_paths:  # 遍历文件夹下的每个图片
                name_path = path + '/' + img_path  # 每张图片的路径
                if class_sum % 10 == 0:  # 每10张图片取一个做验证数据
                    eval_sum += 1  # test_sum为测试数据的数目
                    eval_list.append(name_path + "\t%d" % class_label + "\n")
                else:
                    trainer_sum += 1
                    trainer_list.append(name_path + "\t%d" % class_label + "\n")  # trainer_sum测试数据的数目
                class_sum += 1  # 每类图片的数目
                all_class_images += 1  # 所有类图片的数目

            # 说明的json文件的class_detail数据
            class_detail_list['class_name'] = class_dir  # 类别名称，如jiangwen
            class_detail_list['class_label'] = class_label  # 类别标签
            class_detail_list['class_eval_images'] = eval_sum  # 该类数据的测试集数目
            class_detail_list['class_trainer_images'] = trainer_sum  # 该类数据的训练集数目
            class_detail.append(class_detail_list)
            # 初始化标签列表
            train_parameters['label_dict'][str(class_label)] = class_dir
            class_label += 1

            # 初始化分类数
    train_parameters['class_dim'] = class_dim

    # 乱序
    random.shuffle(eval_list)
    with open(eval_list_path, 'a') as f:
        for eval_image in eval_list:
            f.write(eval_image)

    random.shuffle(trainer_list)
    with open(train_list_path, 'a') as f2:
        for train_image in trainer_list:
            f2.write(train_image)

            # 说明的json文件信息
    readjson = {}
    readjson['all_class_name'] = data_list_path  # 文件父目录
    readjson['all_class_images'] = all_class_images
    readjson['class_detail'] = class_detail
    jsons = json.dumps(readjson, sort_keys=True, indent=4, separators=(',', ': '))
    with open(train_parameters['readme_path'], 'w') as f:
        f.write(jsons)
    print('生成数据列表完成！')

'''
自定义数据集
'''
import paddle
from paddle.io import Dataset





class MyDataset(paddle.io.Dataset):
    """
    步骤一：继承paddle.io.Dataset类
    """
    def __init__(self, mode='train'):
        """
        步骤二：实现构造函数，定义数据集大小
        """
        super(MyDataset, self).__init__()
        self.data = []
        self.label = []
        if mode == 'train':
            #遍历数据文件
            with open(train_list_path, 'r') as f:
                lines = [line.strip() for line in f]
                for line in lines:
                    #读入图片文件地址和标签
                    img_path, lab = line.strip().split('\t')
                    #读入图片
                    img = Image.open(img_path)
                    #判断图片格式
                    if img.mode != 'RGB':
                        #如果图片格式不是RGB则改成RGB
                        img = img.convert('RGB')
                    #将图片大小改变为(224, 224)
                    img = img.resize((224, 224), Image.BILINEAR)
                    #将图片数据转为数组
                    img = np.array(img).astype('float32')
                    # HWC to CHW
                    img = img.transpose((2, 0, 1))
                    # 像素值归一化
                    img = img/255
                    #将数据统一添加到data和label中
                    self.data.append(img)
                    self.label.append(np.array(lab).astype('int64'))
        else:
            #测试集同上
            with open(eval_list_path, 'r') as f:
                lines = [line.strip() for line in f]
                for line in lines:
                    img_path, lab = line.strip().split('\t')
                    img = Image.open(img_path)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img = img.resize((224, 224), Image.BILINEAR)
                    img = np.array(img).astype('float32')
                    img = img.transpose((2, 0, 1))
                    img = img/255
                    self.data.append(img)
                    self.label.append(np.array(lab).astype('int64'))
    def __getitem__(self, index):
        """
        步骤三：实现__getitem__方法，定义指定index时如何获取数据，并返回单条数据（训练数据，对应的标签）
        """
        #返回单一数据和标签
        data = self.data[index]
        label = self.label[index]
        #注：返回标签数据时必须是int64
        return data, np.array(label, dtype='int64')
    def __len__(self):
        """
        步骤四：实现__len__方法，返回数据集总数目
        """
        #返回数据总数
        return len(self.data)
# 测试定义的数据集
train_dataset = MyDataset(mode='train')
eval_dataset = MyDataset(mode='val')
print('=============train_dataset =============')
#输出数据集的形状和标签
print(train_dataset.__getitem__(1)[0].shape,train_dataset.__getitem__(1)[1])
#输出数据集的长度
print(train_dataset.__len__())
print('=============eval_dataset =============')
#输出数据集的形状和标签
for data, label in eval_dataset:
    print(data.shape, label)
    break
#输出数据集的长度