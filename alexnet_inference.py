import torch
import time
import json
import os
import torchvision.models as models
from torchvision import transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_names(path_class, path_class_cn):
    with open(path_class,'r') as f:
        class_names = json.load(f)
    with open(path_class_cn, encoding="UTF-8") as f:
        class_names_cn = f.readlines()
    return class_names,class_names_cn

def process(path_img):
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    imagetransforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean,norm_std),
    ])
    img_rgb = Image.open(path_img).convert('RGB')
    img_tensor = imagetransforms(img_rgb)
    img_tensor.unsqueeze_(0)
    img_tensor = img_tensor.to(device)

    return img_tensor,img_rgb

def get_model(path_model, vis_model=False):
    model = models.alexnet()
    pretrain = torch.load(path_module)
    model.load_state_dict(pretrain)
    model.eval()

    if vis_model:
        from torchsummary import summary
        summary(model, input_size=(3,224,224), device='cpu')
    model.to(device)
    return model

if __name__ == "__main__":
    #load filename_dir
    path_img = os.path.join(BASE_DIR,"data","huanggua.jpg")
    path_module = os.path.join(BASE_DIR,"data","alexnet-owt-4df8aa71.pth")
    path_class_cn = os.path.join(BASE_DIR,"data","imagenet_classnames.txt")
    path_class = os.path.join(BASE_DIR,"data","imagenet1000.json")

    # load class_name
    classes, classes_cn = get_names(path_class, path_class_cn)

    # 1/5 load img
    img_tensor, img_rgb = process(path_img)

    # 2/5 load model
    alexnet_model = get_model(path_module, True)

    # 3/5 inference
    with torch.no_grad():
        output = alexnet_model(img_tensor)

    # 4/5 tensor -- > index
    _,pre = torch.max(output.data,1)
    _,top5 = torch.topk(output.data,5,dim=1)

    pre_index =int(pre.cpu().numpy())
    pre_str,pre_str_cn = classes[pre_index], classes_cn[pre_index]

    #visilization
    plt.imshow(img_rgb)
    plt.title("the predict is {}".format(pre_str))
    top5_num =top5.cpu().numpy().squeeze()
    top5_str = [classes[t] for t in top5_num]
    for i in range(len(top5_str)):
        plt.text(5,12+i*30,"top {} is {}".format(i+1,top5_str[i]),bbox=dict(fc='yellow'))
    plt.show()

