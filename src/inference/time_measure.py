import torch
from torch import nn
import typing
import gc
from src.inference import gpu_utils
from torch.nn import parallel
from torch.utils.data.distributed import DistributedSampler
from torch.utils import data
from torch import distributed as dist
from src.inference import base
import numpy

class MultiGPUInferenceMeasurer(base.InferenceMeasurer):
    """
    Module, that measures inference of the network
    on multiple GPUs.
    
    Parameters:
    -----------
        network - nn.Module (neural network to use)
        input_img - torch.Tensor mxn image
        batch_size - number of images in a single batch, that can be processed by 
        a group of gpus in parallel. Recommended to put actual number of images,
        that will be utilized by the network during actual inference.

        total_repetitions (int) - total number of times 
        inference_devices - list of gpu devices, that will be used during inference measure
        warmup_steps - number of warmup steps to use for warming up gpus
        output_device - device to deploy results to after computation on GPU.
        dist_rank (int) - rank to use for distributed inference.
        dist_world_size (int) - number of gpus utilized during inference.
        dist_backend (str) - distributed backend to use for multi gpu inference. Either "nccl" or "golo"
        dist_group_name (str) - distributed group name to use for processing group.

    """
    def __init__(self, 
        network: nn.Module, 
        input_img: torch.Tensor, 
        batch_size: int, 
        total_repetitions: int, 
        inference_devices: typing.List[torch.DeviceObjType],
        warmup_steps: int,
        output_device: torch.DeviceObjType,
        dist_rank: int,
        dist_world_size: int,
        dist_backend: str,
        dist_group_name: str
    ):
        self.configure_process_group(group_name=dist_group_name)
        self.dist_group_name = dist_group_name
        self.inference_devices = inference_devices
        self.network = parallel.DistributedDataParallel(
            network, 
            device_ids=inference_devices,
            output_device=output_device
        )
        # measurement prerequisites
        self.input_images = torch.stack(input_img.repeat(repeats=batch_size))
        self.batch_size: int = batch_size 
        self.total_repetitions: int = total_repetitions
        self.warmup_steps: int = warmup_steps

        # distributed parameters
        self.dist_rank = dist_rank
        self.dist_world_size: int = dist_world_size
        self.dist_backend = dist_backend
        
    def flush_gpu_cache():
        torch.cuda.empty_cache()
        gc.collect()

    def load_kernel_ops(self, ops: int = 1_000_000):
        """
        Minimizes cost of launching 
        cuda kernels to provide more precise
        inference measurement. Waits until specific 
        number of operations will be loaded to cuda stream.
        """
        torch.cuda._sleep(ops)

    def configure_process_group(self, group_name: str):

        dist.init_process_group(
            backend=self.dist_backend, 
            init_method=None, 
            rank=self.dist_rank,
            world_size=self.dist_world_size,
            group_name=group_name
        ) 

    def destroy_process_group(self, group_name: str):
        dist.destroy_process_group(group=group_name)

    def configure_loader(self):
        sampler = DistributedSampler(
            dataset=self.input_images,
            num_replicas=2,
            rank=self.dist_rank,
            shuffle=False,
            seed=self.dist_seed
        )
        return data.DataLoader(
            dataset=self.input_images,
            shuffle=False,
            sampler=sampler
        )

    def fix_gpus_clock_speed(self, clock_speed: int):
        # fixating gpus clock speed 
        for gpu in self.inference_devices:
            gpu_utils.fix_gpu_clock_speed(
                clock_speed=clock_speed, 
                device_id=gpu 
            )

    def reset_gpus_clock_speed(self):
        # reset gpus clock speed back
        for gpu in self.inference_devices:
            gpu_utils.release_gpu_clock_speed(device_id=gpu)

    def measure_inference_time_ms(self):
        """
        Measures inference time on multiple GPU 
        instances per batch of data.
        Return result in miliseconds.
        """
        for _ in range(self.warmup_steps):
            for images in self.loader:
                _ = self.network.forward(images)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        avg_times = []

        for _ in range(self.total_repetitions):
            # small functions to measure inference more precisely
            self.flush_gpu_cache()
            self.load_kernel_ops()

            start.record()
            batches = 1
            for images in self.loader:
                _ = self.network.forward(images)
                batches += 1
            end.record()
            torch.cuda.synchronize()
            el_time = end.elapsed_time(other=start) / batches
            avg_times.append(el_time)
        return numpy.mean(avg_times)
