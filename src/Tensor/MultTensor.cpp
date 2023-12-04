#include "Tensor/MultTensor.hpp"

#include <eigen3/unsupported/Eigen/CXX11/Tensor>

namespace PPGrad {

    /// @brief Calculate the gradient of this tensor with respect to it's inputs.
    template <int Dim, typename DT>
    void MultTensor<Dim, DT>::backward()
    {
        // C = A * B
        // dC/dA = B (shape of A)
        // dC/dB = A (shape of B)
        if (this->inputA->getRequiresGrad())
        {
            std::shared_ptr<Eigen::Tensor<DT, Dim>> gradPtr = std::make_shared<Eigen::Tensor<DT, Dim>>(*this->gradient * *this->inputB->getData());
            this->inputA->addGrad(gradPtr);
        }

        if (this->inputB->getRequiresGrad())
        {
            std::shared_ptr<Eigen::Tensor<DT, Dim>> gradPtr = std::make_shared<Eigen::Tensor<DT, Dim>>(*this->gradient * *this->inputA->getData());
            this->inputB->addGrad(gradPtr);
        }
    }

    // Explicit instantiations
    template class MultTensor<2, double>; // used in tests
}