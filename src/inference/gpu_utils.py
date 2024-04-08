import subprocess 

def fix_gpu_clock_speed(clock_speed: int, device_id: str):
    """
    Fixates GPU clock speed using
    NVIDIA-SIM command tool.
    NOTE:
        only works for nvidia based GPUs
    """

def release_gpu_clock_speed(device_id: str):
    """
    Releasess GPU clock speed back
    to default using NVIDIA-SIM command util.
    NOTE:
        only works for nvidia based GPUs.
    """