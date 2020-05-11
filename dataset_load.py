from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch

#重写dataset父类，自行导入数据集

class mnist_data(Dataset):

    def __init__(self,root):
        #加载对应路径数据集
        self.pic, self.label = torch.load(root)
      
    def __getitem__(self,index):
        #设置下标可访问、图片数组转化以及图片向量转化
        pic, label= self.pic[index],int(self.label[index])
        pic = Image.fromarray(pic.numpy(),mode='L')
        pic = transforms.ToTensor()(pic)
        #字典存储
        pair={'pic':pic,'label':label}
        return pair

    def __len__(self):
        #返回数据集大小
        return len(self.data)

'''
train_set='./training.pt'
test_set='./test.pt'
train_data=mnist_data(train_set)
train_data=mnist_data(test_set)
'''
