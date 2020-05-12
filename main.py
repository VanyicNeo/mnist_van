import os
from torch import nn,optim
from method.LinearNet import LinearNet
from dataset_load import mnist_data
from torch.utils.data import DataLoader
from configparser import ConfigParser

#引入配置文件
config = ConfigParser()
config.read("config.ini",encoding="utf-8")

#导入数据集
train_data=mnist_data(config.get("PATH","TRAIN_PATH"))
test_data=mnist_data(config.get("PATH","TEST_PATH"))
#pic,lab=train_data[1]
#print(pic.,lab)

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
net=nn.Sequential(
    nn.Linear(
    #in_features
    config.getint("DATA_ARGS","INPUT_SIZE1"),
    config.getint("DATA_ARGS","INPUT_SIZE2"),
    #out_features
    config.getint("DATA_ARGS","OUTPUT_CLASSES_NUM")
    ))
#print(model.size())

#构建损失函数
loss=nn.MSELoss()
print(loss)

#构建优化器
optimizer = optim.Adam(net.parameters(),lr=0.1)
print(optimizer)
'''for i in train_data:
    print('The images size is {}',format(i['pic'].size())) 
    break #本循环就是执行一次
'''

#训练模型
epoch_num=config.getint("LINEAR_ARGS","EPOCH")
for epoch in range(1,epoch_num+1):  #epoch循环
    for i in train_set:
        #print(i.get("pic"))
        output=net(i.get("pic"))
        l=loss(output,i['label'])
        optimizer.zero_grad() #梯度清零
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch,l.item()))