# AlexNet-
using cat&amp;dog image,visualization kernal //
其中一共有4个.py文件
alexnet_inference.py 是用pytorch提供的预训练Alexnet网络对1000类物体进行识别，并根据论文输出top5的结果 
alexnet_visualization.py 对卷积核进行一个可视化，并对第一层卷积后的特征图进行可视化 
train_cat_dog_alexnet.py 采用AlexNet对猫狗数据集进行一个训练，并得出结果，简单5个epoch可以达到95&以上的正确率

tool_dataset.py 编写一个继承Dataset的类，完成图片的读写工作

其中cat&dog的数据集可以在kaggle上面进行下载；
AlexNet训练好的网络可以在pytorch上面的model进行下载；

data文件夹里用来存放train set和test set，图片可以来自网络也可以来自自己拍摄

其中前两个程序在cpu也是可以运行的，最后一个最好是选择GPU 毕竟有25000张图片

本次实验是在 win10 + python3.7 + pytorch1.12 + tensorboard2.0
