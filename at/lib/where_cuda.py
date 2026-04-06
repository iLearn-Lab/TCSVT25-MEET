import os

import torch

def _resolve_cuda_device():
    if not torch.cuda.is_available():
        return torch.device("cpu"), -1, "cpu"

    requested = os.environ.get("LIRONG_CUDA_DEVICE")
    if requested is None:
        requested = os.environ.get("LOCAL_RANK")

    try:
        device_id = int(requested) if requested is not None else 0
    except ValueError:
        device_id = 0

    device_count = torch.cuda.device_count()
    if device_count <= 0:
        return torch.device("cpu"), -1, "cpu"

    if device_id < 0 or device_id >= device_count:
        device_id = 0

    torch.cuda.set_device(device_id)
    device_string = f"cuda:{device_id}"
    return torch.device(device_string), device_id, device_string

device, device_id, device_string = _resolve_cuda_device()
