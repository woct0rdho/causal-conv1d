#pragma once

#include <cuda_runtime.h>

#include <initializer_list>
#include <optional>

#include <torch/csrc/inductor/aoti_torch/generated/c_shim_aten.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/version.h>

namespace causal_conv1d::stable_compat {

using torch::stable::Tensor;
using torch::headeronly::ScalarType;

inline Tensor contiguous(const Tensor& tensor) {
#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_10_0
    return torch::stable::contiguous(tensor);
#else
    auto out = torch::stable::new_empty(tensor, tensor.sizes(), tensor.scalar_type());
    torch::stable::copy_(out, tensor);
    return out;
#endif
}

inline Tensor subtract(const Tensor& tensor, const Tensor& other, double alpha = 1.0) {
#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_10_0
    return torch::stable::subtract(tensor, other, alpha);
#else
    AtenTensorHandle ret = nullptr;
    TORCH_ERROR_CODE_CHECK(
        aoti_torch_aten_subtract_Tensor(tensor.get(), other.get(), alpha, &ret));
    return Tensor(ret);
#endif
}

#if TORCH_FEATURE_VERSION < TORCH_VERSION_2_10_0

inline Tensor new_filled_float_tensor(
    const Tensor& tensor,
    std::initializer_list<int64_t> sizes,
    double value) {
    auto sizes_ref = torch::headeronly::IntHeaderOnlyArrayRef(sizes);
    auto out = torch::stable::new_empty(tensor, sizes_ref, ScalarType::Float);
    torch::stable::fill_(out, value);
    return out;
}

inline Tensor sum_dim0_with_matmul(const Tensor& workspace) {
    STD_TORCH_CHECK(
        workspace.dim() == 2 || workspace.dim() == 3,
        "2.9 compat reduction over dim 0 only supports rank-2 or rank-3 workspaces");

    auto ones = new_filled_float_tensor(workspace, {workspace.size(0), 1}, 1.0);
    if (workspace.dim() == 2) {
        auto transposed = contiguous(torch::stable::transpose(workspace, 0, 1));
        auto reduced = torch::stable::matmul(transposed, ones);
        return torch::stable::select(reduced, 1, 0);
    }

    auto permuted = torch::stable::transpose(workspace, 0, 1);
    permuted = torch::stable::transpose(permuted, 1, 2);
    permuted = contiguous(permuted);
    auto reduced = torch::stable::matmul(permuted, ones);
    return torch::stable::select(reduced, 2, 0);
}

inline Tensor sum_dims01_with_matmul(const Tensor& workspace) {
    STD_TORCH_CHECK(
        workspace.dim() == 3 || workspace.dim() == 4,
        "2.9 compat reduction over dims {0, 1} only supports rank-3 or rank-4 workspaces");

    if (workspace.dim() == 3) {
        auto permuted = torch::stable::transpose(workspace, 0, 2);
        auto flattened = contiguous(torch::stable::flatten(permuted, 1, 2));
        auto ones = new_filled_float_tensor(workspace, {flattened.size(1), 1}, 1.0);
        auto reduced = torch::stable::matmul(flattened, ones);
        return torch::stable::select(reduced, 1, 0);
    }

    auto permuted = torch::stable::transpose(workspace, 0, 2);
    permuted = torch::stable::transpose(permuted, 1, 3);
    auto flattened = contiguous(torch::stable::flatten(permuted, 2, 3));
    auto ones = new_filled_float_tensor(workspace, {flattened.size(2), 1}, 1.0);
    auto reduced = torch::stable::matmul(flattened, ones);
    return torch::stable::select(reduced, 2, 0);
}

inline Tensor sum_along_dims_compat(
    const Tensor& workspace,
    std::initializer_list<int64_t> dims) {
    if (dims.size() == 1 && *dims.begin() == 0) {
        return sum_dim0_with_matmul(workspace);
    }
    if (dims.size() == 2) {
        auto it = dims.begin();
        const int64_t first = *it++;
        const int64_t second = *it;
        if (first == 0 && second == 1) {
            return sum_dims01_with_matmul(workspace);
        }
    }
    STD_TORCH_CHECK(false, "Unsupported reduction dims for 2.9 compat path");
}

#endif

inline void add_workspace_sum_in_place(
    Tensor tensor,
    const Tensor& workspace,
    std::initializer_list<int64_t> dims,
    cudaStream_t stream) {
#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_10_0
    auto dims_ref = torch::headeronly::IntHeaderOnlyArrayRef(dims);
    auto reduced = torch::stable::sum(
        workspace,
        std::optional<torch::headeronly::IntHeaderOnlyArrayRef>(dims_ref),
        false,
        std::nullopt);
    auto updated = subtract(tensor, reduced, -1.0);
    torch::stable::copy_(tensor, updated);
#else
    (void)stream;
    auto reduced = sum_along_dims_compat(workspace, dims);
    auto updated = subtract(tensor, reduced, -1.0);
    torch::stable::copy_(tensor, updated);
#endif
}

} // namespace causal_conv1d::stable_compat
