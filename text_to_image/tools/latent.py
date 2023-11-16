import torch
from diffusers import EulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor


def create_latents(
    batch_size,
    num_channels_latents,
    height,
    width,
    dtype,
    generator,
    vae_scale_factor,
    scheduler,
):
    shape = (
        batch_size,
        num_channels_latents,
        height // vae_scale_factor,
        width // vae_scale_factor,
    )
    latents = randn_tensor(shape, generator=generator, dtype=dtype)


    return latents


if __name__ == "__main__":
    batch_size = 1
    num_channels_latents = 4
    height = 1024
    width = 1024
    dtype = torch.float32
    seed = 0
    generator = torch.Generator(device='cpu').manual_seed(seed)
    vae_scale_factor = 8
    scheduler = EulerDiscreteScheduler.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler"
    )
    save_path = "latents.pt"

    latents = create_latents(
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        generator,
        vae_scale_factor,
        scheduler,
    )
    torch.save(latents, save_path)
