"""
使用alexnet训练猫狗数据集
预训练地址：https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth
数据集地址： kaggle网站下载
"""

#导入需要的包
import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import torchvision.models as models
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from tool_dataset import CDDataset
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_model(path_state_dict, visual_model=False):
    model = models.alexnet()
    pretrain = torch.load(path_state_dict)
    model.load_state_dict(pretrain)
    #可视化模型
    if visual_model:
        from torchsummary import summary
        summary(model, input_size=(3,224,224), dim=1)
    model.to(device)
    return model


if __name__ == '__main__':

    path_state_dict = os.path.join(BASE_DIR,'data','alexnet-owt-4df8aa71.pth')
    data_dir = os.path.join(BASE_DIR,'data','train','train')
    num_class = 2   #可根据你的数据集修改分类数量
    #基本参数设置
    max_epoch = 5
    batsh_size = 128
    learning_rate = 0.001

    # ----------- 1/5 数据预处理及加载 -------------
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    # 依据论文中的要求进行预处理 --> attention 长短边 Resize的操作  32^2*2 = 2048
    train_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop((256,256)),
        transforms.RandomCrop((224,224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_mean),
    ])
    # 验证集处理 tencrop裁剪10张图片
    normalize = transforms.Normalize(norm_mean, norm_std)
    valid_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.TenCrop((224,224), vertical_flip=False),
        transforms.Lambda(lambda imgs: torch.stack([normalize(transforms.ToTensor()(img)) for img in imgs])),
    ])
    #构建Dataset
    train_data = CDDataset(data_dir=data_dir, mode='train', transform=train_transforms)
    valid_data = CDDataset(data_dir=data_dir, mode='valid', transform=valid_transforms)
    #构建Dataloader
    train_loader = DataLoader(dataset=train_data, batch_size=batsh_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=8)

    #---------------------- 2/5 model --------------------
    alexnet_model = get_model(path_state_dict, False)

    #修改为二分类 原始代码为1000
    num = alexnet_model.classifier._modules["6"].in_features
    alexnet_model.classifier._modules["6"] = nn.Linear(num, num_class)
    alexnet_model.to(device)
    #--------------------- 3/5 loss -----------------------
    loss_fn = nn.CrossEntropyLoss()
    #---------------------- 4/5 optim ---------------------
    fc_param_id =list(map(id, alexnet_model.classifier.parameters()))
    base_param =filter(lambda p: id(p) not in fc_param_id, alexnet_model.parameters())
    optimzer = optim.SGD([
        {'params': base_param, 'lr':learning_rate*0.1},
        {'params': alexnet_model.classifier.parameters(), 'lr': learning_rate}], momentum=0.9)
    #学习率下降策略
    scheduler =optim.lr_scheduler.StepLR(optimzer, step_size=1, gamma=0.1) #每一个epoch 学习率下降0.1
    #----------------------- 5/5 train ----------------------
    #保存数据 画图用
    train_loss_list = list()
    train_acc_list = list()
    valid_loss_list = list()
    valid_acc_list = list()

    for epoch in range(max_epoch):
        loss_mean = 0.
        correct = 0.
        total = 0.

        alexnet_model.train()

        for i,data in enumerate(train_loader):

            #前向传播
            input, label =data
            input = input.to(device)
            label = label.to(device)
            output = alexnet_model(input)
            #反向传播
            optimzer.zero_grad()
            loss = loss_fn(output, label)
            loss.backward()
            #更新参数
            optimzer.step()
            #统计分类
            _,pred = torch.max(output.data, 1)
            total += label.size(0)
            correct += (pred == label).squeeze().cpu().sum().numpy()
            acc = (correct / total) * 100.0
            train_acc_list.append(acc)

            #打印信息
            loss_mean += loss.item()
            train_loss_list.append(loss_mean)
            if (i+1) % 1 == 0:
                print("train:\t epoch [{:0>3}/{:0>3}] iteration [{:0>3}/{:0>3}] loss {:.4f} accuracy {:.2f}% ".format(
                    epoch+1, max_epoch, i+1,len(train_loader), loss_mean, acc
                ))
                loss_mean = 0.
        #更新学习率
        scheduler.step()

        #验证模型
        if (epoch+1) % 1 == 0:

            alexnet_model.eval()
            valid_correct = 0
            valid_loss = 0.
            valid_total = 0
            valid_accuracy = 0.
            with torch.no_grad():
                for j,data in enumerate(valid_loader):
                    input, label = data
                    input = input.to(device)
                    label = label.to(device)

                    b, nrow, c, k_w, k_h = input.size()
                    output = alexnet_model(input.view(-1,c ,k_w, k_h))
                    output_avg = output.view(b, nrow, -1).mean(1)

                    loss = loss_fn(output_avg, label)

                    _,pred = torch.max(output_avg.data, 1)
                    valid_total += label.size(0)
                    valid_correct += (pred == label).squeeze().cpu().sum().numpy()

                    valid_loss += loss.item()
                    valid_loss_list.append(valid_loss)
                    valid_acc_list.append(valid_accuracy)
                    valid_accuracy = (valid_correct / valid_total) * 100.0
                print("valid:\t epoch[{:0>3}/{:0>3}]  loss {:.4f}  accuracy {:.2f}%".format(
                        epoch+1,max_epoch, valid_loss/len(valid_loader), valid_accuracy
                    ))
            alexnet_model.train()

    train_x = range(len(train_acc_list))
    train_y1 = train_acc_list
    train_y2 = train_loss_list

    train_iters = len(train_loader)
    valid_x = np.arange(1, len(valid_acc_list)+1) * train_iters
    valid_y1 = valid_acc_list
    valid_y2 = valid_loss_list

    plt.subplot(1,2,1)
    plt.plot(train_x, train_y1)
    plt.plot(valid_x, valid_y1)
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.legend(loc='upper right')
    plt.title("Accuracy and Iteration")

    plt.subplot(1,2,2)
    plt.plot(train_x, train_y2)
    plt.plot(valid_x, valid_y2)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend(loc='upper right')
    plt.title("Loss and Iteration")
    plt.show()
    plt.imsave("loss_and_acc.jpg")
