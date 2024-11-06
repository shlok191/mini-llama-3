import torch
import torch.nn as nn
import numpy as np
import time
from embedding_layer import CUDAEmbedding
import matplotlib.pyplot as plt
from typing import List, Tuple
import pandas as pd

class EmbeddingTester:
    def __init__(self, seed: int = 42):
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Make sure CUDA is available
        assert torch.cuda.is_available(), "CUDA is not available!"
        
        # For storing benchmark results
        self.benchmark_results = []
    
    def test_correctness(self, 
                        batch_size: int = 32, 
                        seq_length: int = 128, 
                        num_embeddings: int = 50000, 
                        embedding_dim: int = 1024,
                        padding_idx: int = -1):
        """Test correctness by comparing with PyTorch's nn.Embedding"""
        print("\nTesting correctness...")
        
        # Create both embedding layers
        custom_embed = CUDAEmbedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        ).cuda()
        
        torch_embed = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        ).cuda()
        
        # Use same weights for fair comparison
        torch_embed.weight.data = custom_embed.weight.data.clone()
        
        # Create random indices
        indices = torch.randint(
            0, num_embeddings, 
            (batch_size, seq_length), 
            device='cuda'
        )
        
        # Add some padding indices
        indices[torch.rand_like(indices.float()) < 0.1] = padding_idx
        
        # Forward pass
        with torch.no_grad():
            custom_output = custom_embed(indices)
            torch_output = torch_embed(indices)
        
        # Check outputs
        max_diff = (custom_output - torch_output).abs().max().item()
        print(f"Maximum difference between custom and PyTorch: {max_diff}")
        assert torch.allclose(custom_output, torch_output, atol=1e-6), \
            "Outputs don't match!"
        
        # Test backward pass
        custom_output.sum().backward()
        custom_grad = custom_embed.weight.grad.clone()
        
        torch_embed.weight.grad = None
        torch_output.sum().backward()
        torch_grad = torch_embed.weight.grad.clone()
        
        grad_diff = (custom_grad - torch_grad).abs().max().item()
        print(f"Maximum gradient difference: {grad_diff}")
        assert torch.allclose(custom_grad, torch_grad, atol=1e-6), \
            "Gradients don't match!"
        
        print("All tests passed!")
    
    def benchmark_forward(self, 
                         batch_size: int,
                         seq_length: int,
                         num_embeddings: int,
                         embedding_dim: int,
                         num_warmup: int = 10,
                         num_repeats: int = 100) -> Tuple[float, float]:
        """Benchmark forward pass and compare with PyTorch"""
        
        # Create embedding layers
        custom_embed = CUDAEmbedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim
        ).cuda()
        torch_embed = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim
        ).cuda()
        
        # Create input
        indices = torch.randint(
            0, num_embeddings, 
            (batch_size, seq_length), 
            device='cuda'
        )
        
        # Warmup
        for _ in range(num_warmup):
            custom_output = custom_embed(indices)
            torch_output = torch_embed(indices)
            torch.cuda.synchronize()
        
        # Benchmark custom implementation
        custom_times = []
        torch.cuda.synchronize()
        for _ in range(num_repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            custom_output = custom_embed(indices)
            end.record()
            
            torch.cuda.synchronize()
            custom_times.append(start.elapsed_time(end))
        
        # Benchmark PyTorch implementation
        torch_times = []
        torch.cuda.synchronize()
        for _ in range(num_repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            torch_output = torch_embed(indices)
            end.record()
            
            torch.cuda.synchronize()
            torch_times.append(start.elapsed_time(end))
        
        custom_avg = np.mean(custom_times)
        torch_avg = np.mean(torch_times)
        
        result = {
            'batch_size': batch_size,
            'seq_length': seq_length,
            'embed_dim': embedding_dim,
            'custom_time': custom_avg,
            'torch_time': torch_avg,
            'speedup': torch_avg / custom_avg
        }
        self.benchmark_results.append(result)
        
        return custom_avg, torch_avg
    
    def run_benchmarks(self):
        """Run a series of benchmarks with different configurations"""
        print("\nRunning benchmarks...")
        
        configs = [
            # (batch_size, seq_length, num_embeddings, embedding_dim)
            (32, 128, 50000, 1024),
            (64, 128, 50000, 1024),
            (128, 128, 50000, 1024),
            (32, 256, 50000, 1024),
            (32, 512, 50000, 1024),
            (32, 128, 50000, 2048),
            (32, 128, 50000, 4096)
        ]
        
        for config in configs:
            print(f"\nBenchmarking configuration: {config}")
            custom_time, torch_time = self.benchmark_forward(*config)
            print(f"Custom implementation: {custom_time:.2f} ms")
            print(f"PyTorch implementation: {torch_time:.2f} ms")
            print(f"Speedup: {torch_time/custom_time:.2f}x")
    
    def plot_results(self):
        """Plot benchmark results"""
        if not self.benchmark_results:
            print("No benchmark results to plot!")
            return
        
        df = pd.DataFrame(self.benchmark_results)
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot timing comparison
        axes[0, 0].bar(['Custom', 'PyTorch'], 
                      [df['custom_time'].mean(), df['torch_time'].mean()],
                      yerr=[df['custom_time'].std(), df['torch_time'].std()])
        axes[0, 0].set_title('Average Execution Time')
        axes[0, 0].set_ylabel('Time (ms)')
        
        # Plot speedup by batch size
        df.plot(x='batch_size', y='speedup', ax=axes[0, 1], marker='o')
        axes[0, 1].set_title('Speedup vs Batch Size')
        axes[0, 1].set_ylabel('Speedup Factor')
        
        # Plot speedup by sequence length
        df.plot(x='seq_length', y='speedup', ax=axes[1, 0], marker='o')
        axes[1, 0].set_title('Speedup vs Sequence Length')
        axes[1, 0].set_ylabel('Speedup Factor')
        
        # Plot speedup by embedding dimension
        df.plot(x='embed_dim', y='speedup', ax=axes[1, 1], marker='o')
        axes[1, 1].set_title('Speedup vs Embedding Dimension')
        axes[1, 1].set_ylabel('Speedup Factor')
        
        plt.tight_layout()
        plt.savefig('benchmark_results.png')
        plt.close()

if __name__ == "__main__":
    tester = EmbeddingTester()
    
    # Run correctness tests
    tester.test_correctness()
    
    # Run benchmarks
    tester.run_benchmarks()
    
    # Plot results
    tester.plot_results()