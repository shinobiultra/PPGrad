#include "Tensor/Tensor.hpp"
#include "Tensor/AddTensor.hpp"
#include "Tensor/SubTensor.hpp"
#include "Tensor/MultTensor.hpp"
#include "Tensor/AddSTensor.hpp"
#include "Tensor/SubSTensor.hpp"
#include "Tensor/MultSTensor.hpp"
#include "Tensor/DivSTensor.hpp"

#include <memory>

namespace PPGrad
{

    /// @brief Backward will simply do nothing for raw Tensors as they are leaf nodes in backpropagation.
    template <int Dim, typename DT>
    void Tensor<Dim, DT>::_backward()
    {
        return;
    }

    /// @brief Get the underlying data of the tensor (of type T).
    /// @details Will probably not be implemented outside of debugging.
    /// @return The actual data.
    template <int Dim, typename DT>
    std::shared_ptr<Eigen::Tensor<DT, Dim>> Tensor<Dim, DT>::getData()
    {
        return this->data;
    }

    /// @brief Zero out the gradient of this tensor.
    template <int Dim, typename DT>
    void Tensor<Dim, DT>::zeroGrad()
    {
        this->gradient->setZero();
    }

    /// @brief Accumulate the gradient of this tensor (intended to be called during backpropagation from nodes "upstream")
    template <int Dim, typename DT>
    void Tensor<Dim, DT>::addGrad(std::shared_ptr<Eigen::Tensor<DT, Dim>> grad)
    {
        *this->gradient += *grad;
    }

    // Explicit template instantiations
    template class Tensor<2, double>; // Used in tests
}