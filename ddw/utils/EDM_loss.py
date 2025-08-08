import torch
import torch.nn.functional as F
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=1.0, data_scale=1.0, data_loc=0.0):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.data_scale = 1.0
        self.data_loc = 0.0

    def normalize(self, x):
        return (x - self.data_loc) / self.data_scale
    def denormalize(self, x):
        return x * self.data_scale + self.data_loc
    
    def __call__(self, net, images, **kwargs):
        loss_weight = 10
        # images = self.normalize(images)
        
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y = images.unsqueeze(1)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma)
        # smooth l1 loss
        loss = weight * F.smooth_l1_loss(D_yn, y, reduction="none")
        # l2 loss
        # loss = weight * ((D_yn - y) ** 2)
        # normalize by the value of the data
        # data_weight = 1+(images ** 2)
        # loss = loss / data_weight
        return loss.mean() * loss_weight , sigma.squeeze()
