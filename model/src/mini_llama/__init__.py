# Import the CUDA extension
from . import cuda

# You can also expose specific functions if you want
from .cuda import attention_forward, attention_backward
