import torch
import random
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from method.LinearNet import new_linearNet
from configparser import ConfigParser
from method.LinearNet import LinearNet
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

def tensor_to_PIL(tensor):
    
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image

config=ConfigParser()
config.read("config.ini",encoding="utf-8")

checkpoint_path=config.get("PATH","MODEL_PATH")
pretrained_model_state=torch.load(checkpoint_path)


model=new_linearNet()
model.load_state_dict(pretrained_model_state)
print(pretrained_model_state,model)
test_data = datasets.MNIST(
    root=config.get("PATH","DATA_PATH"), 
    train=False, 
    transform=transforms.ToTensor())

test_set=DataLoader(
    dataset=test_data,
    shuffle=False,
    batch_size=config.getint("LINEAR_ARGS","TEST_BATCHSIZE"))

random_num=random.randint(0,2000)
print(random_num)

num=random_num
for i,(x,y) in enumerate(test_set):
    if i == num:
        print(x.size(),type(y))
        x1,x2,x3,x4,x5=torch.split(x,1,0) #切割
        img=x1
        print(x1.size())
        x = Variable(x1.view(-1, 28*28)) #变tensor
        y=Variable(y[0])  #变tensor
        outputs = model(x)
        print(x.size())
        _, predicted = torch.max(outputs.data, 1)  #predicted=tensor([x])
        predicted=predicted.squeeze(0)             #predicted=tensor(x)
        print(predicted.squeeze(0),y)
        if predicted.equal(y):
            print("identify successfully!")
            #输出图像和标签
        ax = plt.subplot(1, 1, 1)
        img=tensor_to_PIL(img)
        ax.imshow(img)
        ax.set_title("predicted is "+str(predicted)+" while its label is "+str(y))
        plt.pause(100)



