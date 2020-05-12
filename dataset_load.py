from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch

#重写dataset类，自行导入数据集

class mnist_data(Dataset):

    def __init__(self,root):
        self.pic, self.label = torch.load(root)
      
    def __getitem__(self,index):
        pic, label= self.pic[index],int(self.label[index])
        pic = Image.fromarray(pic.numpy(),mode='L').convert('RGB')
        pic = transforms.ToTensor()(pic)

        pair={'pic':pic,'label':label}
        return pair

    def __len__(self):
        return len(self.pic)


train_set='./training.pt'
test_set='./test.pt'
train_data=mnist_data(train_set)
test_data=mnist_data(test_set)
trainlen=train_data.__len__()