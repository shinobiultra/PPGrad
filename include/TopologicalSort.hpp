
/** @file */

#pragma once
#include "Tensor/TensorBase.hpp"
#include <stack>
#include <memory>

namespace PPGrad
{

    template <int Dim, typename DT>
    std::stack<std::shared_ptr<PPGrad::TensorBase<Dim, DT>>> topologicalSort(std::shared_ptr<PPGrad::TensorBase<Dim, DT>> root);

} // namespace PPGrad