import cProfile
import pstats
import time
import torch
import torch.profiler
from functools import wraps
from typing import Callable, Dict, Any
import numpy as np

def profile_memory_usage(func: Callable) -> Callable:
    """Decorator to track peak memory usage of a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start_mem = torch.cuda.memory_allocated()
        result = func(*args, **kwargs)
        end_mem = torch.cuda.memory_allocated()
        peak_mem = torch.cuda.max_memory_allocated()
        print(f"\nMemory Profile for {func.__name__}:")
        print(f"Peak Memory Usage: {peak_mem / 1024**2:.2f} MB")
        print(f"Net Memory Change: {(end_mem - start_mem) / 1024**2:.2f} MB")
        return result
    return wrapper

def profile_operation_timing(func: Callable) -> Callable:
    """Decorator to measure detailed timing of key operations."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            result = func(*args, **kwargs)
        
        print(f"\nDetailed Profile for {func.__name__}:")
        print(prof.key_averages().table(
            sort_by="cuda_time_total", 
            row_limit=10))
        return result
    return wrapper

class FFTProfiler:
    """Class to perform comprehensive profiling of FFT operations."""
    
    def __init__(self):
        self.timing_stats: Dict[str, list] = {}
        self.memory_stats: Dict[str, list] = {}
    
    def benchmark_fft(self, n: int, batch_size: int = None, num_runs: int = 10):
        """Benchmark FFT performance for given parameters."""
        from algebraist.fourier import sn_fft, sn_ifft
        
        # Generate test data
        factorial_n = np.math.factorial(n)
        if batch_size:
            data = torch.randn(batch_size, factorial_n)
        else:
            data = torch.randn(factorial_n)
            
        # Warmup run
        _ = sn_fft(data, n)
        
        # Profile FFT
        fft_times = []
        fft_memory = []
        
        for _ in range(num_runs):
            torch.cuda.empty_cache()
            start_mem = torch.cuda.memory_allocated()
            
            start_time = time.perf_counter()
            ft = sn_fft(data, n)
            end_time = time.perf_counter()
            
            peak_mem = torch.cuda.max_memory_allocated()
            
            fft_times.append(end_time - start_time)
            fft_memory.append(peak_mem / 1024**2)  # Convert to MB
            
            # Profile inverse FFT
            start_time = time.perf_counter()
            _ = sn_ifft(ft, n)
            end_time = time.perf_counter()
            
        # Store statistics
        self.timing_stats[f'n={n},batch={batch_size}'] = {
            'mean_time': np.mean(fft_times),
            'std_time': np.std(fft_times),
            'min_time': np.min(fft_times),
            'max_time': np.max(fft_times)
        }
        
        self.memory_stats[f'n={n},batch={batch_size}'] = {
            'mean_memory': np.mean(fft_memory),
            'peak_memory': np.max(fft_memory)
        }
    
    def profile_with_cprofile(self, n: int, batch_size: int = None):
        """Profile using cProfile for Python-level insights."""
        from algebraist.fourier import sn_fft
        
        factorial_n = np.math.factorial(n)
        if batch_size:
            data = torch.randn(batch_size, factorial_n)
        else:
            data = torch.randn(factorial_n)
            
        profiler = cProfile.Profile()
        profiler.enable()
        _ = sn_fft(data, n)
        profiler.disable()
        
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        stats.print_stats(20)  # Print top 20 time-consuming functions
    
    def print_summary(self):
        """Print summary of profiling results."""
        print("\nTiming Summary:")
        for config, stats in self.timing_stats.items():
            print(f"\nConfiguration: {config}")
            print(f"Mean execution time: {stats['mean_time']:.4f}s")
            print(f"Std deviation: {stats['std_time']:.4f}s")
            print(f"Min/Max time: {stats['min_time']:.4f}s / {stats['max_time']:.4f}s")
        
        print("\nMemory Summary:")
        for config, stats in self.memory_stats.items():
            print(f"\nConfiguration: {config}")
            print(f"Mean memory usage: {stats['mean_memory']:.2f} MB")
            print(f"Peak memory usage: {stats['peak_memory']:.2f} MB")