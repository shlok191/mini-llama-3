#include <torch/extension.h>
#include "linear/linear.h"
#include "embedding/embedding.h"
#include "attention/attention.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    
    m.def("linear_forward", &linear_forward_cuda, 
        "CUDA Linear Forward",
        py::arg("X"),
        py::arg("weights"));

    m.def("linear_backward_inputs", &linear_backward_inputs_cuda,
        "CUDA Linear Backward Inputs",
        py::arg("grad_output"),
        py::arg("weights_T"));

    m.def("linear_backward_weights", &linear_backward_weights_cuda,
        "CUDA Linear Backward Weights",
        py::arg("grad_output"),
        py::arg("input_T"));

    m.def("embedding_forward", &embedding_forward_cuda,
        "CUDA Embedding Forward",
        py::arg("indices"),
        py::arg("table"));

    m.def("embedding_backward", &embedding_backward_cuda,
        "CUDA Embedding Backward",
        py::arg("grad_output"),
        py::arg("indices"),
        py::arg("table"));

    m.def("calculate_attention_scores", &calculate_attention_scores_cuda, 
        "Calculate attention scores (CUDA)",
        py::arg("query"),
        py::arg("key"));
}
