#include <torch/extension.h>
#include "linear.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    
    m.def("linear_forward", &linear_forward_cuda, 
        "CUDA Linear Forward",
        py::arg("X"),
        py::arg("weights"));

    m.def("linear_backward", &linear_backward_cuda,
        "CUDA Linear Backward",
        py::arg("grad_output"),
        py::arg("input"),
        py::arg("weights"));
}
