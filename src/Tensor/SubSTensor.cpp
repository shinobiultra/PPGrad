#include "Tensor/SubSTensor.hpp"

#include <eigen3/unsupported/Eigen/CXX11/Tensor>

namespace PPGrad
{

    /// @brief Calculate the gradient of this tensor with respect to it's inputs.
    template <int Dim, typename DT>
    void SubSTensor<Dim, DT>::_backward()
    {
        // C = A - B
        // dC/dA = 1 (shape of A)
        // dC/dB = -1 (shape of B)
        if (this->inputA->getRequiresGrad())
        {
            Eigen::Tensor<DT, Dim> grad = Eigen::Tensor<DT, Dim>(this->inputA->getData()->dimensions());
            grad.setConstant(1);
            std::shared_ptr<Eigen::Tensor<DT, Dim>> gradPtr = std::make_shared<Eigen::Tensor<DT, Dim>>(*this->gradient * grad);
            this->inputA->addGrad(gradPtr);
        }
    }

    // Explicit instantiations
    template class SubSTensor<2, double>; // used in tests
}