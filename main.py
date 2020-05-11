import os
from dataset_load import mnist_data
from torch.utils.data import DataLoader
from configparser import ConfigParser

#引入配置文件
config = ConfigParser()
config.read("config.ini",encoding="utf-8")

#导入数据集
train_data=mnist_data(config.get("PATH","TRAIN_PATH"))
train_data=mnist_data(config.get("PATH","TEST_PATH"))

#包装数据集
train_set=DataLoader(
    dataset=train_data,
    shuffle=True,
    batch_size=config.getint("ARGS","TRAIN_BATCHSIZE"))

test_set=DataLoader(
    dataset=train_data,
    shuffle=False,
    batch_size=config.getint("ARGS","TEST_BATCHSIZE"))

