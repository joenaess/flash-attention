#include <torch/extension.h>
#include "flashreduce_kernels.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Flashreduce Affine Scan for Mamba Sequence Models";
    m.def("mamba_forward", &mamba_fwd_cuda, "Mamba Affine Scan Forward Pass");
    m.def("mamba_backward", &mamba_bwd_cuda, "Mamba Affine Scan Backward Pass");
}
