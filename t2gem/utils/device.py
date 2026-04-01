import os
import torch
import torch.distributed as dist
import socket
import subprocess
from contextlib import closing
from t2gem.utils.logger import get_logger
from torch.backends import cudnn
from torch import nn
logger = get_logger(__name__)

def get_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("localhost", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]

def get_master_addr():
    cmd = "scontrol show hostname $SLURM_NODELIST | head -n1"
    return subprocess.check_output(cmd, shell=True).decode().strip()

def slurm_ddp_setup():
    rank = int(os.environ['SLURM_PROCID'])
    world_size = int(os.environ['SLURM_NPROCS'])
    local_rank = int(os.environ["SLURM_LOCALID"])
    dist.init_process_group(backend='nccl',
                            world_size=world_size,
                            rank=rank)
    torch.cuda.set_device(local_rank)
    print(f"Rank {rank}/{world_size} (local_rank {local_rank} on {socket.gethostname()}) initialized. Using GPU: {local_rank}")
    return local_rank, rank, world_size

def ddp_setup():
    dist.init_process_group(backend='nccl', init_method='env://')

    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])

    torch.cuda.set_device(local_rank)
    
    print(
        f"Rank {rank}/{world_size} (local_rank {local_rank} on {socket.gethostname()}) "
        f"initialized via torchrun. Using GPU: {local_rank}"
    )
    
    return local_rank, rank, world_size

def get_amp_dtype_and_device() -> tuple[torch.dtype, torch.device]:
    """Get automatic mixed precision dtype and device.

    Returns:
        amp_dtype: automatic mixed precision dtype.
        device: device.
    """
    amp_dtype = torch.float16
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        device = torch.device("cuda")
        # enable cuDNN auto-tuner, https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
        cudnn.benchmark = True
        if torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
            logger.info("Using bfloat16 for automatic mixed precision.")
    else:
        logger.info("CUDA is not available, using CPU.")
        device = torch.device("cpu")

    return amp_dtype, device

def print_model_info(model: nn.Module) -> None:
    """Print model information.

    Args:
        model: Model to print information.
    """
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"number of parameters: {n_params:,}")
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of trainable parameters: {n_trainable_params:,}")
