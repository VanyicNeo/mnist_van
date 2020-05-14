import os
import torch
from acc_calculate import acc_cal
from torch import nn,optim
from torch.autograd import Variable
from torchvision import datasets,transforms
from method.LinearNet import new_linearNet
from torch.utils import data
from torch.utils.data import DataLoader
from configparser import ConfigParser
from sklearn import preprocessing


#引入配置文件
config = ConfigParser()
config.read("config.ini",encoding="utf-8")

#导入数据集
train_data = datasets.MNIST(
    root=config.get("PATH","DATA_PATH"), 
    train=True,
    transform=transforms.ToTensor(), 
    download=False)

test_data = datasets.MNIST(
    root=config.get("PATH","DATA_PATH"), 
    train=False, 
    transform=transforms.ToTensor())

#包装数据集
train_set=DataLoader(
    dataset=train_data,
    shuffle=True,
    batch_size=config.getint("LINEAR_ARGS","TRAIN_BATCHSIZE"))

test_set=DataLoader(
    dataset=test_data,
    shuffle=False,
    batch_size=config.getint("LINEAR_ARGS","TEST_BATCHSIZE"))

#构建线性回归器
net=new_linearNet()

#构建损失函数
loss=nn.CrossEntropyLoss()

#构建优化器
optimizer = optim.Adam(net.parameters(),lr=0.001)

#模型参数定义
iter=0
loss_now=0
epoch_num=config.getint("LINEAR_ARGS","EPOCH")
MODEL_PATH=config.get("PATH","MODEL_PATH")

#输出网络、损失函数、优化器详细参数
print(net)
print(loss)
print(optimizer)
print("Training begins")

#epoch循环
for epoch in range(1,epoch_num+1):  
    print("This is epoch"+str(epoch))
    for i,(x,y) in enumerate(train_set):
        #更改维度
        x=Variable(x.view(-1,28*28))
        y=Variable(y)
        #梯度下降、损失函数、反向、迭代
        optimizer.zero_grad()
        output=net(x)
        l=loss(output,y)
        l.backward()
        optimizer.step()

        #计算准确率
        #print("acc cal")
        #accuracy=acc_cal(iter,test_set)
        
        iter+=1
        if iter%500==0:
            correct = 0
            total = 0
            for images, labels in test_set:
                images = Variable(images.view(-1, 28*28))
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total+= labels.size(0)
                correct+= (predicted == labels).sum()
            accuracy = 100 * correct/total
            
    loss_now=l.item()
    if epoch==1:
        loss_min=loss_now
        #保存模型
    if loss_now < loss_min:
        loss_min=loss_now
        torch.save(net.state_dict(),MODEL_PATH)
        print("model saved!")    
    print('epoch %d, loss: %f,acc: %f' % (epoch,l.item(),accuracy))
    

