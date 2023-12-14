
/** @file */

#include "Tensor/ReLUTensor.hpp"
#include <unsupported/Eigen/CXX11/Tensor>
#include <memory>
#include <vector>
#include <iostream>

namespace PPGrad
{

    /// @brief Calculate the gradient of this tensor with respect to it's inputs.
    template <int Dim, typename DT>
    void ReLUTensor<Dim, DT>::_backward()
    {
        // C = ReLU(A)
        // dC/dA = 1 if A > 0, 0 otherwise
        if (this->input->getRequiresGrad())
        {
            Eigen::Tensor<DT, Dim> grad = this->input->getData()->unaryExpr([](DT x) { return x > 0 ? (DT) 1 : (DT) 0; });
            std::shared_ptr<Eigen::Tensor<DT, Dim>> gradPtr = std::make_shared<Eigen::Tensor<DT, Dim>>(*this->gradient * grad);
            this->input->addGrad(gradPtr);
        }

    }

    template <int Dim, typename DT>
    std::vector<std::shared_ptr<PPGrad::TensorBase<Dim, DT>>> ReLUTensor<Dim, DT>::getParents()
    {
        std::vector<std::shared_ptr<PPGrad::TensorBase<Dim, DT>>> parents;
        parents.push_back(this->input);
        return parents;
    }

    // Explicit instantiations
    template class ReLUTensor<2, double>; // used in tests
}