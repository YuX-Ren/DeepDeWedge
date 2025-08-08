import torch
import numpy as np
from ddw.utils.unet import LitUnet3D
import mrcfile
@torch.no_grad()
def edm_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):


    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float32, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([torch.as_tensor(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = latents.to(torch.float32) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = torch.as_tensor(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step.
        denoised = net(x_hat, t_hat).to(torch.float32)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, t_next).to(torch.float32)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next

if __name__ == "__main__":
    from ddw.utils.mrctools import load_mrc_data, save_mrc_data
    model_checkpoint_file = "tutorial/tutorial_project/logs/version_8/checkpoints/fitting_loss/epoch=237-fitting_loss=0.15652.ckpt"
    # model_checkpoint_file = "logs/version_1/checkpoints/epoch/epoch=499.ckpt"
    gpu = 0
    device = "cpu" if gpu is None else f"cuda:{gpu}"

    print(f"Loading model from {model_checkpoint_file}")
    net = (
        LitUnet3D.load_from_checkpoint(model_checkpoint_file).to(device).eval()
    )
    data_path = "tutorial/tutorial_project/subtomos/val_subtomos/subtomo0/0.pt"
    latents = torch.load(data_path).clip(0, 3).to(device).unsqueeze(0)
    # load mrc file
    # latents = mrcfile.open("/root/CryoGEN-CR/dataset/tomograms_deconv/TS01-wbp.rec").data.astype(np.float32)
    # latents = torch.from_numpy(latents)
    print(latents.shape)
    # normalize
    # latents = (latents - latents.mean()) / latents.std()
    noised_latents = latents.to(device)
    # noised_latents = noised_latents[:,100:100+64, 100:100+64, 100:100+64]
    print(noised_latents.shape)
    # add noise
    noised_latents = latents + torch.randn_like(latents) * 1
    noised_latents = noised_latents
    output = edm_sampler(net, noised_latents).detach().cpu()
    # # large noise
    # onestep_output = net(noised_latents, torch.tensor(8).to(device)).detach().cpu()
    # save_mrc_data(latents.clip(-6, 6).squeeze(0).cpu(), "latents.mrc")
    save_mrc_data(noised_latents.squeeze(0).cpu(), "unet_noised_latents.mrc")
    save_mrc_data(output, "unet_output_restormer.mrc")
    # save_mrc_data(onestep_output, "onestep_output.mrc")