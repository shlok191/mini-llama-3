#include <torch/extension.h>
#include "linear/linear.h"
#include "embedding/embedding.h"

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

    m.def("embedding_forward", &embedding_forward_cuda,
        "CUDA Embedding Forward",
        py::arg("indices"),
        py::arg("table"));

    m.def("embedding_backward", &embedding_backward_cuda,
        "CUDA Embedding Backward",
        py::arg("grad_output"),
        py::arg("indices"),
        py::arg("table"));
}
