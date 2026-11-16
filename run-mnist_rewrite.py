import time
import argparse
import os

import cerebras.pytorch as cstorch
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_loader_rewrite import (
    apply_aug_batch,
    get_default_crop_padding,
    get_torch_dataloader,
    _horizontal_flip_batch,
    AUG_BRIGHTNESS,
    AUG_NOISE,
    AUG_VFLIP,
    AUG_CONTRAST,
    AUG_HFLIP,
    AUG_ROTATION,
    AUG_INVERT,
)
from cerebras.modelzoo.models.vision.vision_transformer.ViTClassificationModel import (
    ViTClassificationModel,
    ViTClassificationModelConfig,
)

start_time = time.time()
data_path = os.environ["MNIST_DATA_PATH"]
mount_path = os.environ["MOUNT_PATH"]



def main():
    parser = argparse.ArgumentParser(description="Parameter Processing")
    parser.add_argument("--dataset", type=str, default="C10", help="dataset: C10, C100, S, F, M, I")
    parser.add_argument("--model", type=str, default="FNN", help="network: FNN, CNN, ViT")
    parser.add_argument("--model_layer", type=int, default=2, help="How many layers of a model needed?")
    parser.add_argument("--model_width", type=int, default=256, help="width/channels in the model")
    parser.add_argument("--DA_WSE", action="store_true", default=False, help="Implement DA on WSE")
    parser.add_argument(
        "--DA_layer",
        type=int,
        default=2,
        help="1: brightness, 2: +noise, 3: +hflip, 4: +contrast, 5: +vflip, 6: +rotation, 7: +inversion",
    )
    parser.add_argument("--iteration", type=int, default=100, help="Number of iterations to train")
    parser.add_argument("--bs", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--half_dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "cbfloat16"],
        help="Cerebras half dtype",
    )
    parser.add_argument("--enable_hflip", action="store_true", default=False, help="Enable random horizontal flip")
    parser.add_argument("--enable_vflip", action="store_true", default=False, help="Enable random vertical flip")
    parser.add_argument("--enable_crop", action="store_true", default=False, help="Enable random crop with same output size")
    parser.add_argument("--enable_inversion", action="store_true", default=False, help="Enable random image inversion")
    parser.add_argument(
        "--crop_pad",
        type=int,
        default=-1,
        help="Padding used by random crop. Use -1 to select a dataset-specific default.",
    )
    parser.add_argument("--hflip_prob", type=float, default=0.5, help="Probability for horizontal flip")
    parser.add_argument("--vflip_prob", type=float, default=0.5, help="Probability for vertical flip")
    parser.add_argument("--inversion_prob", type=float, default=0.5, help="Probability for inversion")
    parser.add_argument("--num_workers", type=int, default=0, help="number of workers to load data")
    args = parser.parse_args()

    for name in ["hflip_prob", "vflip_prob", "inversion_prob"]:
        value = getattr(args, name)
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"{name} must be in [0, 1], got {value}")

    crop_pad = get_default_crop_padding(args.dataset) if args.crop_pad < 0 else args.crop_pad

    # Avoid default float16 unless you specifically want it.
    cstorch.amp.set_half_dtype(args.half_dtype)

    print("DA on WSE." if args.DA_WSE else "DA on host.")
    print(f"half dtype: {args.half_dtype}")
    print(
        "Enabled DA options: "
        f"hflip={args.enable_hflip}, "
        f"vflip={args.enable_vflip}, "
        f"rotation=True, "
        f"inversion={args.enable_inversion}"
    )

    class AugmentationLayer(nn.Module):
        def __init__(self, dataset_name: str, model_name: str, crop_pad: int):
            super().__init__()
            self.dataset_name = dataset_name
            self.model_name = model_name
            self.crop_pad = crop_pad

        def forward(self, x, aug_params, DA_layer):
            x = apply_aug_batch(
                x=x,
                aug_params=aug_params,
                dataset_name=self.dataset_name,
                model_name=self.model_name,
                DA_layer=DA_layer,
                crop_pad=self.crop_pad,
            )
            return x.contiguous()

    class ModelFNN(nn.Module):
        def __init__(self, size, layer, width, num_classes, dataset_name, model_name, crop_pad):
            super().__init__()
            self.layer = layer
            self.width = width
            self.augment = AugmentationLayer(dataset_name, model_name, crop_pad)
            self.fc1 = nn.Linear(size, self.width)
            self.middle = nn.Linear(self.width, self.width)
            self.fc2 = nn.Linear(self.width, num_classes)

        def forward(self, inputs, aug_params=None, DA_layer=0):
            x = inputs
            if self.training and aug_params is not None:
                x = self.augment(x, aug_params, DA_layer)
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            if self.layer > 2:
                for _ in range(self.layer - 2):
                    x = F.relu(self.middle(x))
            return self.fc2(x)
        
    class ModelCNN(nn.Module):
        def __init__(self, in_channels, input_size, num_layers, channels, num_classes, dataset_name, model_name, crop_pad):
            super().__init__()
            self.augment = AugmentationLayer(dataset_name, model_name, crop_pad)
            self.convs = nn.ModuleList()
            current_channels = in_channels

            for _ in range(num_layers - 1):
                self.convs.append(nn.Conv2d(current_channels, channels, kernel_size=3))
                current_channels = channels

            conv_out_size = input_size - (num_layers - 1) * 2
            flattened_dim = channels * conv_out_size * conv_out_size
            self.fc = nn.Linear(flattened_dim, num_classes)

        def forward(self, inputs, aug_params=None, DA_layer=0):
            x = inputs
            if self.training and aug_params is not None:
                x = self.augment(x, aug_params, DA_layer)
            for conv in self.convs:
                x = F.relu(conv(x))
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    class ModelViT(nn.Module):
        def __init__(self, config, dataset_name, model_name, crop_pad):
            super().__init__()
            if isinstance(config, dict):
                config = ViTClassificationModelConfig(**config)
            self.augment = AugmentationLayer(dataset_name, model_name, crop_pad)
            self.model = ViTClassificationModel(config)
            self.use_dinov2_classifier = self.model.use_dinov2_classifier

        def forward(self, inputs, aug_params=None, DA_layer=0):
            x = inputs
            if self.training and aug_params is not None:
                x = self.augment(x, aug_params, DA_layer)
            return self.model(x)

    class ModelFNN_aug1(nn.Module):
        def __init__(self, size, layer, width, num_classes, dataset_name, model_name, crop_pad):
            super().__init__()
            self.layer = layer
            self.width = width
            self.augment = AugmentationLayer(dataset_name, model_name, crop_pad)
            self.fc1 = nn.Linear(size, self.width)
            self.middle = nn.Linear(self.width, self.width)
            self.fc2 = nn.Linear(self.width, num_classes)

        def forward(self, inputs, aug_params=None, DA_layer=0):
            x = inputs
            b, c, h, w = x.shape

            if self.training and aug_params is not None:
                brightness = aug_params[:, AUG_BRIGHTNESS].view(b, 1, 1, 1)
                x = brightness * x

                noise = aug_params[:, AUG_NOISE].view(b, 1, 1, 1)
                x = x + noise

                vflip_mask = aug_params[:, AUG_VFLIP].view(b, 1, 1, 1) > 0.5
                x_flip = torch.cat([x[:, :, :, i : i + 1] for i in range(w - 1, -1, -1)], dim=3)
                x = torch.where(vflip_mask, x_flip, x)

                mean = x.mean(dim=(2, 3), keepdim=True)
                x = (x - mean) * aug_params[:, AUG_CONTRAST].view(b, 1, 1, 1) + mean

                hflip_mask = aug_params[:, AUG_HFLIP].view(b, 1, 1, 1) > 0.5
                x_flip = torch.cat([x[:, :, i : i + 1, :] for i in range(h - 1, -1, -1)], dim=2)
                x = torch.where(hflip_mask, x_flip, x)

                hflip_mask = aug_params[:, AUG_ROTATION].view(b, 1, 1, 1) > 0.5
                x_flip = torch.cat([x[:, :, :, i : i + 1] for i in range(w - 1, -1, -1)], dim=3)
                x_flip = torch.cat([x[:, :, i : i + 1, :] for i in range(h - 1, -1, -1)], dim=2)
                x = torch.where(hflip_mask, x_flip, x)

                hflip_mask = aug_params[:, AUG_INVERT].view(b, 1, 1, 1) > 0.5
                x_flip = 1.0-x
                x = torch.where(hflip_mask, x_flip, x)



            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            if self.layer > 2:
                for _ in range(self.layer - 2):
                    x = F.relu(self.middle(x))
            return self.fc2(x)

    class ModelCNN_aug1(nn.Module):
        def __init__(self, in_channels, input_size, num_layers, channels, num_classes, dataset_name, model_name, crop_pad):
            super().__init__()
            self.augment = AugmentationLayer(dataset_name, model_name, crop_pad)
            self.convs = nn.ModuleList()
            current_channels = in_channels

            for _ in range(num_layers - 1):
                self.convs.append(nn.Conv2d(current_channels, channels, kernel_size=3))
                current_channels = channels

            conv_out_size = input_size - (num_layers - 1) * 2
            flattened_dim = channels * conv_out_size * conv_out_size
            self.fc = nn.Linear(flattened_dim, num_classes)

        def forward(self, inputs, aug_params=None, DA_layer=0):
            x = inputs
            b, c, h, w = x.shape
            if self.training and aug_params is not None:
                brightness = aug_params[:, AUG_BRIGHTNESS].view(b, 1, 1, 1)
                x = brightness * x

                noise = aug_params[:, AUG_NOISE].view(b, 1, 1, 1)
                x = x + noise

                vflip_mask = aug_params[:, AUG_VFLIP].view(b, 1, 1, 1) > 0.5
                x_flip = torch.cat([x[:, :, :, i : i + 1] for i in range(w - 1, -1, -1)], dim=3)
                x = torch.where(vflip_mask, x_flip, x)

                mean = x.mean(dim=(2, 3), keepdim=True)
                x = (x - mean) * aug_params[:, AUG_CONTRAST].view(b, 1, 1, 1) + mean

                hflip_mask = aug_params[:, AUG_HFLIP].view(b, 1, 1, 1) > 0.5
                x_flip = torch.cat([x[:, :, i : i + 1, :] for i in range(h - 1, -1, -1)], dim=2)
                x = torch.where(hflip_mask, x_flip, x)

                hflip_mask = aug_params[:, AUG_ROTATION].view(b, 1, 1, 1) > 0.5
                x_flip = torch.cat([x[:, :, :, i : i + 1] for i in range(w - 1, -1, -1)], dim=3)
                x_flip = torch.cat([x[:, :, i : i + 1, :] for i in range(h - 1, -1, -1)], dim=2)
                x = torch.where(hflip_mask, x_flip, x)

                hflip_mask = aug_params[:, AUG_INVERT].view(b, 1, 1, 1) > 0.5
                x_flip = 1.0-x
                x = torch.where(hflip_mask, x_flip, x)
            for conv in self.convs:
                x = F.relu(conv(x))
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    class ModelViT_aug1(nn.Module):
        def __init__(self, config, dataset_name, model_name, crop_pad):
            super().__init__()
            if isinstance(config, dict):
                config = ViTClassificationModelConfig(**config)
            self.augment = AugmentationLayer(dataset_name, model_name, crop_pad)
            self.model = ViTClassificationModel(config)
            self.use_dinov2_classifier = self.model.use_dinov2_classifier

        def forward(self, inputs, aug_params=None, DA_layer=0):
            x = inputs
            b, c, h, w = x.shape
            if self.training and aug_params is not None:
                brightness = aug_params[:, AUG_BRIGHTNESS].view(b, 1, 1, 1)
                x = brightness * x

                noise = aug_params[:, AUG_NOISE].view(b, 1, 1, 1)
                x = x + noise

                vflip_mask = aug_params[:, AUG_VFLIP].view(b, 1, 1, 1) > 0.5
                x_flip = torch.cat([x[:, :, :, i : i + 1] for i in range(w - 1, -1, -1)], dim=3)
                x = torch.where(vflip_mask, x_flip, x)

                mean = x.mean(dim=(2, 3), keepdim=True)
                x = (x - mean) * aug_params[:, AUG_CONTRAST].view(b, 1, 1, 1) + mean

                hflip_mask = aug_params[:, AUG_HFLIP].view(b, 1, 1, 1) > 0.5
                x_flip = torch.cat([x[:, :, i : i + 1, :] for i in range(h - 1, -1, -1)], dim=2)
                x = torch.where(hflip_mask, x_flip, x)

                hflip_mask = aug_params[:, AUG_ROTATION].view(b, 1, 1, 1) > 0.5
                x_flip = torch.cat([x[:, :, :, i : i + 1] for i in range(w - 1, -1, -1)], dim=3)
                x_flip = torch.cat([x[:, :, i : i + 1, :] for i in range(h - 1, -1, -1)], dim=2)
                x = torch.where(hflip_mask, x_flip, x)

                hflip_mask = aug_params[:, AUG_INVERT].view(b, 1, 1, 1) > 0.5
                x_flip = 1.0-x
                x = torch.where(hflip_mask, x_flip, x)
            return self.model(x)

    if args.dataset in {"C10", "C100", "S"}:
        size, in_channels, num_classes, input_size = 3 * 32 * 32, 3, 10, 32
        patch = 4
        if args.dataset == "C100":
            num_classes = 100
    elif args.dataset in {"F", "M"}:
        size, in_channels, num_classes, input_size = 28 * 28, 1, 10, 28
        patch = 7
    elif args.dataset == "I":
        size, in_channels, num_classes, input_size = 3 * 128 * 128, 3, 10, 128
        patch = 16
    else:
        raise ValueError(f"Unsupported dataset {args.dataset}")

    if args.model == "FNN":
        max_wgt_servers, max_act_per_csx=100, 1
        model = ModelFNN_aug1(
            size=size,
            layer=args.model_layer,
            width=args.model_width,
            num_classes=num_classes,
            dataset_name=args.dataset,
            model_name=args.model,
            crop_pad=crop_pad,
        )     
    elif args.model == "CNN":
        max_wgt_servers, max_act_per_csx=6, 1
        model = ModelCNN_aug1(
            in_channels=in_channels,
            input_size=input_size,
            num_layers=args.model_layer,
            channels=args.model_width,
            num_classes=num_classes,
            dataset_name=args.dataset,
            model_name=args.model,
            crop_pad=crop_pad,
        )
    elif args.model == "ViT":
        max_wgt_servers, max_act_per_csx=1, 1
        config = ViTClassificationModelConfig(
            image_size=(input_size, input_size),
            num_channels=in_channels,
            num_classes=num_classes,
            patch_size=(patch, patch),
            hidden_size=64,
            num_hidden_layers=args.model_layer,
            num_heads=4,
            filter_size=128,
            dropout_rate=0.1,
            attention_dropout_rate=0.1,
        )
        model = ModelViT_aug1(config, dataset_name=args.dataset, model_name=args.model, crop_pad=crop_pad)
    else:
        raise ValueError("Model must be one of FNN, CNN, ViT")
    
    # cstorch.backends.csx.debug.compile_crd_memory_gi = 90
    backend = cstorch.backend(
        "CSX",
        cluster_config=cstorch.distributed.ClusterConfig(
            mount_dirs=[mount_path],
            python_paths=[mount_path],
            max_wgt_servers=max_wgt_servers,
            max_act_per_csx=max_act_per_csx,
            num_workers_per_csx=1,
            disable_version_check=True,
        ),
    )

    compiled_model = cstorch.compile(model, backend=backend)

    if args.model == "ViT":
        optimizer = cstorch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    else:
        optimizer = cstorch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    training_dataloader = cstorch.utils.data.DataLoader(
        get_torch_dataloader,
        dataset_name=args.dataset,
        batch_size=args.bs,
        train=True,
        DA_WSE=args.DA_WSE,
        DA_layer=args.DA_layer,
        model_name=args.model,
        num_workers=args.num_workers,
    )

    loss_fn = nn.CrossEntropyLoss()

    @cstorch.trace
    def training_step(inputs, targets, aug_params):
        if not args.DA_WSE:
            aug_params = None 
        outputs = compiled_model(inputs, aug_params, DA_layer=args.DA_layer)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss

    @cstorch.step_closure
    def print_loss(loss: torch.Tensor, step: int, train_time_start: float):
        if step == 2:
            train_time_start = time.time()
        if step % 1000 == 0 or step == args.iteration:
            print(f"Train Loss {step}: {loss.item()}")
            print("Per iteration runtime: {}".format((time.time()-train_time_start)/(step-2)))


    global_step = 0
    train_executor = cstorch.utils.data.DataExecutor(
        training_dataloader,
        num_steps=args.iteration,
        checkpoint_steps=None,
    )

    model.train()
    train_time_start = time.time()
    for inputs, targets, aug_params in train_executor:
        loss = training_step(inputs, targets, aug_params)
        global_step += 1
        print_loss(loss, global_step, train_time_start)
    


if __name__ == "__main__":
    main()
    print("Used actually time is {}".format(time.time() - start_time))
