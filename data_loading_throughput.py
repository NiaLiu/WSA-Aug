import time
from data_loader_rewrite import get_torch_dataloader
import argparse

def main():
    parser = argparse.ArgumentParser(description="Parameter Processing")
    parser.add_argument("--dataset", type=str, default="C10", help="dataset: C10, C100, S, F, M, I")
    parser.add_argument("--DA_WSE", action="store_true", default=False, help="Implement DA on WSE")
    parser.add_argument(
        "--DA_layer",
        type=int,
        default=2,
        help="1: contrast, 2: +hflip, 3: +brightness, 4: +noise, 5: +vflip, 6: +crop, 7: +inversion",
    )
    parser.add_argument("--iteration", type=int, default=100, help="Number of iterations to train")
    parser.add_argument("--bs", type=int, default=16, help="Batch size")

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
    # Set these parameters to match the ones you use in your main.py
    batch_size = 800  # You can increase this to push the CPU harder
    num_iterations = 500
    dataset_name = "C10"
    num_workers = 0  # Change this to test multi-core scaling
    args.hflip_prob = True
    args.vflip_prob = True

    
    # We set DA_WSE=False so the CPU is forced to compute the augmentations[cite: 1, 2]
    dataloader = get_torch_dataloader(
        dataset_name=dataset_name,
        batch_size=batch_size,
        train=True,
        DA_WSE=False,     
        DA_layer=7,       # Use a heavy DA layer to stress the CPU
        model_name="CNN",
        enable_hflip=True,
        enable_vflip=True,
        enable_crop=True,
        enable_inversion=True,
        num_workers=num_workers 
    )

    data_iter = iter(dataloader)

    # 1. Warmup Phase: 
    # Dataloaders have startup overhead (spinning up worker processes). 
    # We run a few batches first so this initialization time isn't included in our metric.
    print(f"Warming up {num_workers} dataloader workers...")
    for _ in range(10):
        try:
            next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            next(data_iter)

    # 2. Measurement Phase:
    print("Starting CPU throughput measurement...")
    start_time = time.time()

    for _ in range(num_iterations):
        try:
            inputs, targets, aug_params = next(data_iter)
        except StopIteration:
            # If the dataset runs out, restart it seamlessly
            data_iter = iter(dataloader)
            inputs, targets, aug_params = next(data_iter)

    end_time = time.time()
    
    # 3. Calculate Metrics
    total_time = end_time - start_time
    total_images = num_iterations * batch_size
    images_per_second = total_images / total_time

    print("-" * 30)
    print(f"Total time for {num_iterations} batches: {total_time:.4f} seconds")
    print(f"Total images processed: {total_images}")
    print(f"CPU Throughput: {images_per_second:.2f} images/second")

if __name__ == "__main__":
    main()