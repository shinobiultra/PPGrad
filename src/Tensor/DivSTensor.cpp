
/** @file */


#include "Tensor/DivSTensor.hpp"

#include <unsupported/Eigen/CXX11/Tensor>

namespace PPGrad
{

    /// @brief Calculate the gradient of this tensor with respect to it's inputs.
    template <int Dim, typename DT>
    void DivSTensor<Dim, DT>::_backward()
    {
        if (this->inputA->getRequiresGrad())
        {
            Eigen::Tensor<DT, Dim> grad = Eigen::Tensor<DT, Dim>(this->inputA->getData()->dimensions());
            grad.setConstant(1 / this->inputB);
            std::shared_ptr<Eigen::Tensor<DT, Dim>> gradPtr = std::make_shared<Eigen::Tensor<DT, Dim>>(*this->gradient * grad);

            this->inputA->addGrad(gradPtr);
        }
    }

    template <int Dim, typename DT>
    std::vector <std::shared_ptr<PPGrad::TensorBase<Dim, DT>>> DivSTensor<Dim, DT>::getParents() {
        std::vector <std::shared_ptr<PPGrad::TensorBase<Dim, DT>>> parents;
        parents.push_back(this->inputA);
        return parents;
    }

    // Explicit instantiations
    template class DivSTensor<2, double>; // used in tests
}