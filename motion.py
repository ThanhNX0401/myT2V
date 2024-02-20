
import torch
import torch.nn.functional as F
from torch.nn.functional import grid_sample


def rearrange_3(tensor, f):
    F, D, C = tensor.size()
    return torch.reshape(tensor, (F // f, f, D, C))


def rearrange_4(tensor):
    B, F, D, C = tensor.size()
    return torch.reshape(tensor, (B * F, D, C))

def coords_grid(batch, ht, wd, device):
    # Adapted from https://github.com/princeton-vl/RAFT/blob/master/core/utils/utils.py
    coords = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wd, device=device)) 
    coords = torch.stack(coords[::-1], dim=0).float() 

    return coords[None].repeat(batch, 1, 1, 1) # (b, 2, h, w)

def warp_single_latent(latent, reference_flow):
    _, _, H, W = reference_flow.size()
    _, _, h, w = latent.size()
    coords0 = coords_grid(1, H, W, device=latent.device).to(latent.dtype) # (1, 2, H, W)

    coords_t0 = coords0.clone() # (1, 2, H, W)
    coords_t0[:, 0] -= reference_flow[:, 0] # subtract for x
    coords_t0[:, 1] += reference_flow[:, 1] # add for y
    coords_t0[:, 0] /= W                # normalize
    coords_t0[:, 1] /= H

    coords_t0 = coords_t0 * 2.0 - 1.0 # to [-1, 1]
    coords_t0 = F.interpolate(coords_t0, size=(h, w), mode="bilinear") #resize to (64,64)
    coords_t0 = torch.permute(coords_t0, (0, 2, 3, 1)) # (1, 64, 64, 2)

    warped = grid_sample(latent, coords_t0, mode="nearest", padding_mode="reflection") # (1, 4, 64, 64)
    #warped = grid_sample(latent, coords_t0, mode="nearest", padding_mode="zeros")
    if (latent.size(1) == 1):
        warped = (warped > 0.5).to(latent.dtype)
    return warped


def create_motion_field(motion_field_strength_x, motion_field_strength_y, device, dtype):
    reference_flow = torch.zeros((1, 2, 512, 512), device=device, dtype=dtype)
    reference_flow[0, 0, :, :] = motion_field_strength_x
    reference_flow[0, 1, :, :] = motion_field_strength_y
    return reference_flow 


def warp_latents(motion_field_strength_x, motion_field_strength_y, latents):
    motion_field = create_motion_field(
        motion_field_strength_x=motion_field_strength_x,
        motion_field_strength_y=motion_field_strength_y,
        device=latents.device,
        dtype=latents.dtype,
    )
    warped_latents = latents.clone().detach()# latents: [7,4,64,64], detach =  don't want to compute gradients
    warped_latents = warp_single_latent(latents, motion_field)
    return warped_latents
