from dataset_load import *
import matplotlib.pyplot as plt


train_set='./training.pt'
test_set='./test.pt'
train_data=mnist_data(train_set)
train_data=mnist_data(test_set)


def data_show(data):
    #数据集展示
    for (cnt,i) in enumerate(data):
        img = i['pic']
        lab = i['label']
        ax = plt.subplot(4, 4, cnt+1)
    # ax.axis('off')
        ax.imshow(img.squeeze(0))
        ax.set_title(lab)
        plt.pause(0.001)
        if cnt ==15:
            break

data_show(train_data)
