import torch
import numpy as np
import tomosipo as ts
from ts_algorithms import fbp

@torch.no_grad()
def noise_fbp(angle = 60,size = 96):
    # spectral noise
    angle = angle[0].item()
    angles = np.linspace(angle/2, 180-angle/2, 40, endpoint=True)/180 * np.pi
    vg = ts.volume(shape=(size, size, size), size=(1, 1, 1))
    pg = ts.parallel(angles=angles,shape=(int(size*1.5), int(size*1.5)),size=(1.5, 1.5))
    A = ts.operator(vg, pg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    y = torch.randn(int(size*1.5), 40, int(size*1.5), device=device) 
    rec_fbp = fbp(A, y).permute(2,0,1) 
    # normalize to 0 mean and 1 std
    spec_noise = rec_fbp / rec_fbp.std()
    # real space noise
    real_noise = torch.randn(size, size, size, device=device)
    # random ratio 
    ratio = torch.rand(1, device=device)
    noise = torch.sqrt(ratio) * spec_noise + torch.sqrt(1-ratio) * real_noise
    return noise
