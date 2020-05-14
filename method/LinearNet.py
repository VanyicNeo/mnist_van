from torch import nn
from configparser import ConfigParser

config = ConfigParser()
config.read("config.ini",encoding="utf-8")

class LinearNet(nn.Module):
    '''Applies a linear transformation to the incoming data: 
       :math:`y = xA^T + b`'''


    def __init__(self,in_feature,out_feature):
        super(LinearNet,self).__init__()
        self.linear=nn.Linear(in_feature,out_feature)

    def forward(self,x):
        output=self.linear(x)
        return output

'''
    def printdata(dataset):
        for i in enumerate(dataset):
            print(i)
            break
        '''

def new_linearNet():
    net=LinearNet(
    #in_features=28*28
    config.getint("DATA_ARGS","INPUT_SIZE1")*config.getint("DATA_ARGS","INPUT_SIZE2"),
    #out_features=10
    config.getint("DATA_ARGS","OUTPUT_CLASSES_NUM")
    )
    return net


    