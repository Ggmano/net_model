import torch

box_loss_scale_x = mask = torch.zeros(4, 3, 4, 4, requires_grad=False)
mask = torch.zeros(4, 3, 4, 4, requires_grad=False)
mask[0, :, 1, :] = 1
box_loss_scale_x[0, 1, 1, 1] = 5
# 将非零值筛选出来，一维展开
print(box_loss_scale_x[mask.bool()])
