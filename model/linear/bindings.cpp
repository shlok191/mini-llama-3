#include <torch/extension.h>
#include "linear.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    
    m.def("linear_forward", &linear_forward_cuda, 
        "CUDA Linear Forward",
        py::arg("X"),
        py::arg("weights"));
}