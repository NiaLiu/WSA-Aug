import os
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import random

MNIST_PATH = os.environ.get("MNIST_DATA_PATH", "/home/ge73zuh2/DA_WSE/dataset/")

AUG_BRIGHTNESS = 0
AUG_NOISE = 1
AUG_VFLIP = 2
AUG_CONTRAST = 3
AUG_HFLIP = 4
AUG_ROTATION = 5
AUG_INVERT = 6
AUG_PARAM_DIM = 7


def get_dataset_stats(dataset_name: str):
    stats = {
        "M": ((0.1307,), (0.3081,)),
        "F": ((0.2860,), (0.3530,)),
        "C10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        "C100": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        "S": ((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        "I": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    }
    if dataset_name not in stats:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")
    return stats[dataset_name]



def get_default_crop_padding(dataset_name: str) -> int:
    if dataset_name in {"M", "F"}:
        return 2
    if dataset_name in {"C10", "C100", "S"}:
        return 4
    if dataset_name == "I":
        return 8
    raise ValueError(f"Unsupported dataset_name: {dataset_name}")



def normalize_image_tensor(x: torch.Tensor, dataset_name: str) -> torch.Tensor:
    mean, std = get_dataset_stats(dataset_name)
    mean = torch.tensor(mean, dtype=x.dtype, device=x.device).view(-1, 1, 1)
    std = torch.tensor(std, dtype=x.dtype, device=x.device).view(-1, 1, 1)
    return (x - mean) / std



def _get_normalized_zero_fill(dataset_name: str, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    mean, std = get_dataset_stats(dataset_name)
    mean = torch.tensor(mean, dtype=dtype, device=device)
    std = torch.tensor(std, dtype=dtype, device=device)
    return (-mean / std).view(-1, 1, 1)



def _get_normalized_inversion_bias(dataset_name: str, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    mean, std = get_dataset_stats(dataset_name)
    mean = torch.tensor(mean, dtype=dtype, device=device)
    std = torch.tensor(std, dtype=dtype, device=device)
    return ((1.0 - 2.0 * mean) / std).view(-1, 1, 1)

def _inversion(x: torch.Tensor) -> torch.Tensor:
    return 1.0-x



def sample_aug_params(
    dataset_name: str,
    model_name: str,
) -> torch.Tensor:
    
    brightness = random.uniform(0.001, 0.3)
    noise = random.uniform(0.001, 0.1)
    contrast = random.randint(0, 1)
    hflip_flag = random.randint(0, 1)
    vflip_flag = random.randint(0, 1)
    invert_flag = random.randint(0, 1)
    rot_flag = random.randint(0, 1)
    
    out = torch.zeros(AUG_PARAM_DIM, dtype=torch.float32)
    out[AUG_BRIGHTNESS] = brightness
    out[AUG_NOISE] = noise
    out[AUG_VFLIP] = vflip_flag
    out[AUG_CONTRAST] = contrast
    out[AUG_HFLIP] = hflip_flag
    out[AUG_ROTATION] = rot_flag
    out[AUG_INVERT] = invert_flag
    return out



def _pad_image_single(img: torch.Tensor, dataset_name: str, crop_pad: int) -> torch.Tensor:
    if crop_pad <= 0:
        return img
    c, h, w = img.shape
    fill = _get_normalized_zero_fill(dataset_name, dtype=img.dtype, device=img.device)
    padded = fill.expand(c, h + 2 * crop_pad, w + 2 * crop_pad).clone()
    padded[:, crop_pad : crop_pad + h, crop_pad : crop_pad + w] = img
    return padded



def _pad_image_batch(x: torch.Tensor, dataset_name: str, crop_pad: int) -> torch.Tensor:
    if crop_pad <= 0:
        return x
    b, c, h, w = x.shape
    fill = _get_normalized_zero_fill(dataset_name, dtype=x.dtype, device=x.device).view(1, c, 1, 1)
    padded = fill.expand(b, c, h + 2 * crop_pad, w + 2 * crop_pad).clone()
    padded[:, :, crop_pad : crop_pad + h, crop_pad : crop_pad + w] = x
    return padded


def _horizontal_flip_single(x: torch.Tensor) -> torch.Tensor:
    w = x.shape[2]
    return torch.cat([x[:, :, i : i + 1] for i in range(w - 1, -1, -1)], dim=2)



def _vertical_flip_single(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[1]
    return torch.cat([x[:, i : i + 1, :] for i in range(h - 1, -1, -1)], dim=1)



def _horizontal_flip_batch(x: torch.Tensor) -> torch.Tensor:
    w = x.shape[3]
    return torch.cat([x[:, :, :, i : i + 1] for i in range(w - 1, -1, -1)], dim=3)



def _vertical_flip_batch(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[2]
    return torch.cat([x[:, :, i : i + 1, :] for i in range(h - 1, -1, -1)], dim=2)


def apply_aug_image(
    img: torch.Tensor,
    aug_params: torch.Tensor,
    dataset_name: str,
    model_name: str,
    DA_layer: int,
) -> torch.Tensor:
    # img is already normalized.
    x = img
    aug_params = aug_params.to(dtype=x.dtype)

    if DA_layer >= 1:
        brightness = aug_params[AUG_BRIGHTNESS]
        x = brightness * x

    if DA_layer >= 2:
        x = x + aug_params[AUG_NOISE]


    if DA_layer >= 3:
        condition = aug_params[AUG_HFLIP] > 0.5
        x_flip = _horizontal_flip_single(x)
        x = torch.where(condition, x_flip, x)

    if DA_layer >= 4:
        mean = x.mean(dim=(1, 2), keepdim=True)
        x = (x - mean) * aug_params[AUG_CONTRAST] + mean

    if DA_layer >= 5:
        condition = aug_params[AUG_VFLIP] > 0.5
        x_flip = _vertical_flip_single(x)
        x = torch.where(condition, x_flip, x)

    if DA_layer >= 6:
        condition = aug_params[AUG_ROTATION] > 0.5
        x_rot = _vertical_flip_single(x)
        x = torch.where(condition, x_rot, x)

    if DA_layer >= 7:
        condition = aug_params[AUG_INVERT] > 0.5
        x_inv = _inversion(x)
        x = torch.where(condition, x_inv, x)

    return x.contiguous()


def apply_aug_batch(
    x: torch.Tensor,
    aug_params: torch.Tensor,
    dataset_name: str,
    model_name: str,
    DA_layer: int,
    crop_pad: Optional[int] = None,
) -> torch.Tensor:
    b = x.shape[0]
    out = x
    aug_params = aug_params.to(dtype=x.dtype)

    if DA_layer>=0:
        brightness = aug_params[:, AUG_BRIGHTNESS].view(b, 1, 1, 1)
        x = brightness * x

    if DA_layer>=1:
        noise = aug_params[:, AUG_NOISE].view(b, 1, 1, 1)
        x = x + noise

    if DA_layer>=2:
        vflip_mask = aug_params[:, AUG_VFLIP].view(b, 1, 1, 1) > 0.5
        x_flip = torch.cat([x[:, :, :, i : i + 1] for i in range(w - 1, -1, -1)], dim=3)
        x = torch.where(vflip_mask, x_flip, x)

    if DA_layer>=3:
        mean = x.mean(dim=(2, 3), keepdim=True)
        x = (x - mean) * aug_params[:, AUG_CONTRAST].view(b, 1, 1, 1) + mean

    if DA_layer>=4:
        hflip_mask = aug_params[:, AUG_HFLIP].view(b, 1, 1, 1) > 0.5
        x_flip = torch.cat([x[:, :, i : i + 1, :] for i in range(h - 1, -1, -1)], dim=2)
        x = torch.where(hflip_mask, x_flip, x)

    if DA_layer>=5:
        hflip_mask = aug_params[:, AUG_ROTATION].view(b, 1, 1, 1) > 0.5
        x_flip = torch.cat([x[:, :, :, i : i + 1] for i in range(w - 1, -1, -1)], dim=3)
        x_flip = torch.cat([x[:, :, i : i + 1, :] for i in range(h - 1, -1, -1)], dim=2)
        x = torch.where(hflip_mask, x_flip, x)

    if DA_layer>=6:
        hflip_mask = aug_params[:, AUG_INVERT].view(b, 1, 1, 1) > 0.5
        x_flip = 1.0-x
        x = torch.where(hflip_mask, x_flip, x)

    return out.contiguous()


class dataset_init(Dataset):
    def __init__(
        self,
        dataset_name: str = "M",
        path: str = MNIST_PATH,
        train: bool = True,
        download: bool = True,
        DA: bool = False,
        DA_layer: int = 1,
        model_name: str = "FNN",
    ):
        self.dataset_name = dataset_name
        self.DA = DA
        self.DA_layer = DA_layer
        self.model_name = model_name

        if dataset_name == "M":
            transform = transforms.Compose([transforms.ToTensor()])
            self.dataset = datasets.MNIST(path, train=train, download=download, transform=transform)
        elif dataset_name == "C10":
            transform = transforms.Compose([transforms.ToTensor()])
            self.dataset = datasets.CIFAR10(path, train=train, download=download, transform=transform)
        elif dataset_name == "C100":
            transform = transforms.Compose([transforms.ToTensor()])
            self.dataset = datasets.CIFAR100(path, train=train, download=download, transform=transform)
        elif dataset_name == "S":
            split = "train" if train else "test"
            transform = transforms.Compose([transforms.ToTensor()])
            self.dataset = datasets.SVHN(path, split=split, download=download, transform=transform)
        elif dataset_name == "F":
            transform = transforms.Compose([transforms.ToTensor()])
            self.dataset = datasets.FashionMNIST(path, train=train, download=download, transform=transform)
        elif dataset_name == "I":
            split = "train" if train else "val"
            transform = transforms.Compose(
                [
                    transforms.Resize((128, 128)),
                    transforms.CenterCrop((128, 128)),
                    transforms.ToTensor(),
                ]
            )
            self.dataset = datasets.ImageNet(path, split=split, transform=transform)
        else:
            raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img, target = self.dataset[index]
        target = torch.as_tensor(target, dtype=torch.int32)

        # Always normalize in the dataloader so the compiled WSE graph stays simple.
        img = normalize_image_tensor(img, self.dataset_name)

        aug_params = sample_aug_params(
            dataset_name=self.dataset_name,
            model_name=self.model_name,
        )

        if self.DA:
            img = apply_aug_image(
                img=img,
                aug_params=aug_params,
                dataset_name=self.dataset_name,
                model_name=self.model_name,
                DA_layer=self.DA_layer,
            )

        return img, target, aug_params



def get_torch_dataloader(
    dataset_name,
    batch_size,
    train,
    DA_WSE,
    DA_layer,
    model_name,
    num_workers=0,
):
    dataset = dataset_init(
        dataset_name=dataset_name,
        path=MNIST_PATH,
        train=train,
        download=True,
        DA=not DA_WSE,
        DA_layer=DA_layer,
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        drop_last=True,
        num_workers=num_workers,
    )
