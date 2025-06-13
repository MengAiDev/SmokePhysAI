# Utility functions
# Contains helper functions for physics modeling and simulation

import torch
import contextlib

@contextlib.contextmanager
def cuda_timer(msg="", enabled=True):
    """CUDA性能计时器"""
    if not enabled or not torch.cuda.is_available():
        yield
        return
        
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    
    yield
    
    end.record()
    torch.cuda.synchronize()
    print(f"{msg} 耗时: {start.elapsed_time(end):.2f}ms")

def optimize_cuda_cache():
    """优化CUDA缓存"""
    if torch.cuda.is_available():
        # 清理缓存
        torch.cuda.empty_cache()
        # 设置内存分配器
        torch.cuda.memory.set_per_process_memory_fraction(0.8)  # 预留20%系统内存
        torch.cuda.memory._set_allocator_settings("max_split_size_mb:512")

def create_pinned_buffer(shape, dtype=torch.float32):
    """创建固定内存缓冲区"""
    return torch.zeros(shape, dtype=dtype, pin_memory=True)

def async_gpu_transfer(tensor, device, stream=None):
    """异步GPU数据传输"""
    if stream is None:
        stream = torch.cuda.current_stream()
    with torch.cuda.stream(stream):
        return tensor.cuda(device, non_blocking=True)

def batch_to_gpu(batch_data, device):
    """批量数据高效传输到GPU"""
    if isinstance(batch_data, torch.Tensor):
        return batch_data.to(device, non_blocking=True)
    elif isinstance(batch_data, (list, tuple)):
        return [batch_to_gpu(item, device) for item in batch_data]
    elif isinstance(batch_data, dict):
        return {k: batch_to_gpu(v, device) for k, v in batch_data.items()}
    return batch_data