import torch
import os
import torch.nn as nn
import torchvision.utils as vutils
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    log_dir = os.path.join(BASE_DIR,"results")

#-------------kernel visualization------
    writer = SummaryWriter(log_dir=log_dir, filename_suffix='_kernel')

    path_state_dict = os.path.join(BASE_DIR,"data","alexnet-owt-4df8aa71.pth")
    alexnet = models.alexnet()
    pretrain_state_dict = torch.load(path_state_dict)
    alexnet.load_state_dict(pretrain_state_dict)

    kernel_num = -1
    vis_max = 1

    for sub_module in alexnet.modules():
        if not isinstance(sub_module,nn.Conv2d):
            continue
        kernel_num += 1

        if kernel_num > vis_max:
            break

        kernels = sub_module.weight
        c_out, c_in ,k_w, k_h = tuple(kernels.shape)

        for o_idx in range(c_out):
            kernel_idx = kernels[o_idx,:,:,:].unsqueeze(1)
            kernel_grid = vutils.make_grid(kernel_idx, normalize=True, scale_each=True, nrow=c_in)
            writer.add_image('{} convlayer split in channels'.format(kernel_num), kernel_grid, global_step=o_idx)

        kernel_all =kernels.view(-1, 3, k_h, k_w)
        kernel_grid = vutils.make_grid(kernel_all, normalize=True, scale_each=True,nrow=8)
        writer.add_image('{}_all'.format(kernel_num),kernel_grid)

        print("{} convlayer shape".format(kernel_num, tuple(kernels.shape)))

#--------------------------feature map visualization---------

    writer = SummaryWriter(log_dir=log_dir, filename_suffix='_feature_map')
    path_img = os.path.join(BASE_DIR,"data","hanxue.jpg")

    img_rgb = Image.open(path_img).convert("RGB")
    normMean = [0.49139968, 0.48215827, 0.44653124]
    normStd = [0.24703233, 0.24348505, 0.26158768]
    norm_transform = transforms.Normalize(normMean, normStd)
    img_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        norm_transform,
    ])

    img_tensor = img_transforms(img_rgb)
    img_tensor.unsqueeze_(0)

    convlayer1 = alexnet.features[0]
    fmap1 = convlayer1(img_tensor)

    fmap1.transpose_(0,1)
    fmap1_grid = vutils.make_grid(fmap1, normalize=True, scale_each=True, nrow=8)
    writer.add_image("feature map in convlayer1",fmap1_grid, global_step=620)
    writer.close()
