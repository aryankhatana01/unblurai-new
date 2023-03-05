import argparse
import os

import cv2
import torch
from torch import nn

import preprocess
import model
from utils import load_state_dict

model_names = sorted(
    name for name in model.__dict__ if
    name.islower() and not name.startswith("__") and callable(model.__dict__[name]))


def choice_device(device_type: str) -> torch.device:
    # Select model processing equipment type
    if device_type == "cuda":
        device = torch.device("cuda", 0)
    else:
        device = torch.device("cpu")
    return device


def build_model(model_arch_name: str, device: torch.device) -> nn.Module:
    # Initialize the super-resolution model
    print(model.__dict__)
    g_model = model.__dict__[model_arch_name](in_channels=3,
                                              out_channels=3,
                                              channels=64,
                                              growth_channels=32,
                                              num_blocks=23)
    g_model = g_model.to(device=device)

    return g_model


def main(
    device_type, 
    model_arch_name, 
    model_weights_path, 
    inputs_path, 
    output_path
    ):
    device = choice_device(device_type)

    # Initialize the model
    g_model = build_model(model_arch_name, device)
    print(f"Build `{model_arch_name}` model successfully.")

    # Load model weights
    g_model = load_state_dict(g_model, model_weights_path)
    print(f"Load `{model_arch_name}` model weights `{os.path.abspath(model_weights_path)}` successfully.")

    # Start the verification mode of the model.
    g_model.eval()

    lr_tensor = preprocess.preprocess_one_image(inputs_path, device)

    # Use the model to generate super-resolved images
    with torch.no_grad():
        sr_tensor = g_model(lr_tensor)

    # Save image
    sr_image = preprocess.tensor_to_image(sr_tensor, False, False)
    sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, sr_image)

    print(f"SR image save to `{output_path}`")


if __name__ == "__main__":
    main(
        device_type="cuda" if torch.cuda.is_available() else "cpu",
        model_arch_name="rrdbnet_x4",
        model_weights_path="./pretrained-models/ESRGAN_x4-DFO2K-25393df7.pth.tar",
        inputs_path="/Users/0x4ry4n/Desktop/Dev/unblurai-new/esrgan/test_img/test.jpeg",
        output_path="./test-img/sr.jpg"
    )