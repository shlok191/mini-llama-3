from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Get CUDA_HOME
cuda_home = os.getenv('CUDA_HOME', '/usr/local/cuda')

# Check if CUDA is available
if not os.path.exists(cuda_home):
    raise RuntimeError(
        "CUDA_HOME environment variable is not set or CUDA is not installed. "
        "Please install CUDA and set CUDA_HOME environment variable."
    )

setup(
    # Basic package information
    name="cuda_kernels",
    version="0.1.0",
    author="Shlok Sabarwal",
    author_email="ssabarwal@wisc.edu",
    description="Custom CUDA kernels for a Mini LLama Model!",
    long_description="A fast CUDA implementation for the Mini LLama Model with aim of having 80% speed compared to CuBLAS",
    
    # Package configuration
    packages=find_packages(),
    
    # Extension module configuration
    ext_modules=[
        CUDAExtension(
            name='cuda_kernels',  
            sources=[
                './bindings.cpp',     
                './embedding/embedding.cu',           
                './linear/linear.cu',       
            ],
            
            # Include directories
            include_dirs=[
                os.path.join(cuda_home, 'include'),
                'include'  # If you have a local include directory
            ],
            
            # Extra compile arguments
            extra_compile_args={
                'cxx': [
                    '-O3',                  # High optimization level
                    '-std=c++17',          # C++ standard
                    '-Wno-deprecated',      # Suppress deprecation warnings
                    '-fopenmp'             # Enable OpenMP support
                ],
                'nvcc': [
                    '-O3',                         # High optimization level
                    '--use_fast_math',             # Use fast math operations
                    '-std=c++17',                  # C++ standard
                    '--ptxas-options=-v',          # Verbose PTXAS output
                    '-lineinfo',                   # Include line information
                    '--expt-relaxed-constexpr',
                    '--expt-extended-lambda',
                    '-Xcompiler', '-fPIC',         # Pass -fPIC to the host compiler
                    '-gencode=arch=compute_70,code=sm_70',
                    '-gencode=arch=compute_75,code=sm_75',
                    '-gencode=arch=compute_80,code=sm_80',
                    '-gencode=arch=compute_86,code=sm_86'
                ],

            }
        )
    ],
    
    # Build configuration
    cmdclass={
        'build_ext': BuildExtension
    },
    
    # Python package dependencies
    install_requires=[
        'torch>=2.4.0',
        'numpy==1.26.0'
    ],
    
    # Additional package data
    package_data={
        'cuda_kernels': ['*.h']
    },
    
    # Python version requirement
    python_requires='>=3.7',
)
