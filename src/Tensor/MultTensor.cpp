#include "Tensor/MultTensor.hpp"

#include <eigen3/unsupported/Eigen/CXX11/Tensor>

namespace PPGrad
{

    /// @brief Calculate the gradient of this tensor with respect to it's inputs.
    template <int Dim, typename DT>
    void MultTensor<Dim, DT>::_backward()
    {
        if (!this->requiresGrad)
        {
            return;
        }

        // For now, let's consider only matrices, vectors and scalars
        if (Dim > 2)
        {
            throw std::runtime_error("MultTensor _backward() not implemented for tensors of dimension > 2");
        }

        if (this->inputA->getRequiresGrad())
        {
            Eigen::array<Eigen::IndexPair<int>, 1> contractionPair = {Eigen::IndexPair<int>(Dim - 1, 0)};
            Eigen::Tensor<DT, 2 * (Dim - 1)> gradA = this->gradient->contract(this->inputB->getData()->shuffle(Eigen::array<int, 2>{1, 0}), contractionPair);
            this->inputA->addGrad(std::make_shared<Eigen::Tensor<DT, Dim>>(gradA));
        }

        if (this->inputB->getRequiresGrad())
        {
            Eigen::array<Eigen::IndexPair<int>, 1> contractionPair = {Eigen::IndexPair<int>(Dim - 1, 0)};
            Eigen::Tensor<DT, 2 * (Dim - 1)> gradB = this->inputA->getData()->shuffle(Eigen::array<int, 2>{1, 0}).contract(*this->gradient, contractionPair);
            this->inputB->addGrad(std::make_shared<Eigen::Tensor<DT, Dim>>(gradB));
        }
    }

    // Explicit instantiations
    template class MultTensor<2, double>; // used in tests
}