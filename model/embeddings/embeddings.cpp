#include <torch/extension.h>
#include "embedding.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("embedding_forward", &embedding_forward_cuda, 
          "CUDA Embedding Forward",
          py::arg("indices"),
          py::arg("weights"),
          py::arg("padding_token_index"));
    
    m.def("embedding_backward", &embedding_backward_cuda,
          "CUDA Embedding Backward",
          py::arg("grad_output"),
          py::arg("indices"),
          py::arg("weights"),
          py::arg("padding_idx"));
}
