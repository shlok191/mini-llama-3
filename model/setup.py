from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

def get_cuda_home():
    """
    Retrieves and validates CUDA home directory.
    Raises informative error if CUDA is not properly configured.
    """
    cuda_home = os.getenv('CUDA_HOME', '/usr/local/cuda')
    
    if not os.path.exists(cuda_home):
        
        raise RuntimeError(
            "CUDA_HOME environment variable is not set or CUDA is not installed. "
            "Please install CUDA and set CUDA_HOME environment variable."
        )
        
    return cuda_home


def get_cuda_extension():
    """
    Configures CUDA extension with appropriate compiler flags and source files.
    Returns configured CUDAExtension object.
    """
    
    cuda_home = get_cuda_home()
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    return CUDAExtension(
        name='mini_llama.cuda',  # Note: Changed to submodule for better organization
        sources=[
            './cuda/bindings.cpp',
            './cuda/embedding/embedding.cu',
            './cuda/linear/linear.cu',
            './cuda/attention/attention.cu'
        ],
        include_dirs=[
            os.path.join(cuda_home, 'include'),
            os.path.join(project_root, 'cuda'),  # Add cuda directory for header discovery
            os.path.join(project_root, 'cuda/embedding'),  # Add specific module directories
            os.path.join(project_root, 'cuda/linear'),
            os.path.join(project_root, 'cuda/attention'),
        ],
        extra_compile_args={
            'cxx': [
                '-O3',
                '-std=c++17',
                '-Wno-deprecated',
                '-fopenmp'
            ],
            'nvcc': [
                '-O3',
                '--use_fast_math',
                '-std=c++17',
                '--ptxas-options=-v',
                '-lineinfo',
                '--expt-relaxed-constexpr',
                '--expt-extended-lambda',
                '-Xcompiler', '-fPIC',
                '-gencode=arch=compute_86,code=sm_86'
            ],
        }
    )
    
setup(
    # Package metadata
    name="mini_llama",
    version="0.1.0",
    author="Shlok Sabarwal",
    author_email="ssabarwal@wisc.edu",
    description="Custom CUDA kernels for a Mini LLama Model!",
    long_description="A fast CUDA implementation for the Mini LLama Model with aim of having 80% speed compared to CuBLAS",
    
    # Package structure configuration
    packages=find_packages(where="src"),  # Look for packages in src directory
    package_dir={"": "src"},  # Tell setuptools packages are under src
    
    # Extension modules (CUDA)
    ext_modules=[get_cuda_extension()],
    
    # Build configuration
    cmdclass={
        'build_ext': BuildExtension
    },
    
    # Dependencies
    install_requires=[
        'torch>=2.4.0',
        'numpy==1.26.0',
        "lightning>=1.9", # For checkpointing and connecting to W&B  
        "wandb>=0.19.0",  # For visualizing training statistics :)
        "tqdm>=4.6.0"     # Adds fancy progress bars
    ],
    
    # Additional package data
    package_data={
        'mini_llama': [
            './*/*.json',
        ]
    },
    include_package_data=True,
    # Python version requirement
    python_requires='>=3.10',
    
    # Additional classifiers for PyPI
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: C++',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)