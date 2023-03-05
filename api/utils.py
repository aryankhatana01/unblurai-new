import os

import cv2

import preprocess
import model

import shutil
from enum import Enum
from typing import Any

import torch
from torch import nn
from torch.nn import Module
from torch.optim import Optimizer

__all__ = [
    "load_state_dict", "make_directory", "save_checkpoint",
    "Summary", "AverageMeter", "ProgressMeter"
]


def load_state_dict(
        model: nn.Module,
        model_weights_path: str,
        ema_model: nn.Module = None,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler = None,
        load_mode: str = None,
) -> tuple[Module, Module, Any, Any, Any, Optimizer | None, Any] | tuple[Module, Any, Any, Any, Optimizer | None, Any] | Module:
    # Load model weights
    checkpoint = torch.load(model_weights_path, map_location=lambda storage, loc: storage)

    if load_mode == "resume":
        # Restore the parameters in the training node to this point
        start_epoch = checkpoint["epoch"]
        best_psnr = checkpoint["best_psnr"]
        best_ssim = checkpoint["best_ssim"]
        # Load model state dict. Extract the fitted model weights
        model_state_dict = model.state_dict()
        state_dict = {k: v for k, v in checkpoint["state_dict"].items() if k in model_state_dict.keys()}
        # Overwrite the model weights to the current model (base model)
        model_state_dict.update(state_dict)
        model.load_state_dict(model_state_dict)
        # Load the optimizer model
        optimizer.load_state_dict(checkpoint["optimizer"])

        if scheduler is not None:
            # Load the scheduler model
            scheduler.load_state_dict(checkpoint["scheduler"])

        if ema_model is not None:
            # Load ema model state dict. Extract the fitted model weights
            ema_model_state_dict = ema_model.state_dict()
            ema_state_dict = {k: v for k, v in checkpoint["ema_state_dict"].items() if k in ema_model_state_dict.keys()}
            # Overwrite the model weights to the current model (ema model)
            ema_model_state_dict.update(ema_state_dict)
            ema_model.load_state_dict(ema_model_state_dict)

        return model, ema_model, start_epoch, best_psnr, best_ssim, optimizer, scheduler
    else:
        # Load model state dict. Extract the fitted model weights
        model_state_dict = model.state_dict()
        state_dict = {k: v for k, v in checkpoint["state_dict"].items() if
                      k in model_state_dict.keys() and v.size() == model_state_dict[k].size()}
        # Overwrite the model weights to the current model
        model_state_dict.update(state_dict)
        model.load_state_dict(model_state_dict)

        return model


def make_directory(dir_path: str) -> None:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def save_checkpoint(
        state_dict: dict,
        file_name: str,
        samples_dir: str,
        results_dir: str,
        best_file_name: str,
        last_file_name: str,
        is_best: bool = False,
        is_last: bool = False,
) -> None:
    checkpoint_path = os.path.join(samples_dir, file_name)
    torch.save(state_dict, checkpoint_path)

    if is_best:
        shutil.copyfile(checkpoint_path, os.path.join(results_dir, best_file_name))
    if is_last:
        shutil.copyfile(checkpoint_path, os.path.join(results_dir, last_file_name))


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.2f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.2f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.2f}"
        else:
            raise ValueError(f"Invalid summary type {self.summary_type}")

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


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


def predict_one_image(
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

