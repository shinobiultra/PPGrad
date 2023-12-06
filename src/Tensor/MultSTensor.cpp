#include "Tensor/MultSTensor.hpp"

#include <eigen3/unsupported/Eigen/CXX11/Tensor>

namespace PPGrad
{

    /// @brief Calculate the gradient of this tensor with respect to it's inputs.
    template <int Dim, typename DT>
    void MultSTensor<Dim, DT>::_backward()
    {
        if (this->inputA->getRequiresGrad())
        {
            Eigen::Tensor<DT, Dim> grad = Eigen::Tensor<DT, Dim>(this->inputA->getData()->dimensions());
            grad.setConstant(this->inputB);
            std::shared_ptr<Eigen::Tensor<DT, Dim>> gradPtr = std::make_shared<Eigen::Tensor<DT, Dim>>(*this->gradient * grad);

            this->inputA->addGrad(gradPtr);
        }
    }

    // Explicit instantiations
    template class MultSTensor<2, double>; // used in tests
}