#include "Tensor/MultTensor.hpp"

#include <eigen3/unsupported/Eigen/CXX11/Tensor>

namespace PPGrad
{

    /// @brief Calculate the gradient of this tensor with respect to it's inputs.
    template <int Dim, typename DT>
    void MultTensor<Dim, DT>::backward()
    {
        if (!this->requiresGrad)
        {
            return;
        }

        // For now, let's consider only matrices
        if (Dim != 2)
        {
            throw std::runtime_error("MultTensor backward() not implemented for tensors of dimension != 2");
        }

        // result = A * B
        const int resultRows = this->inputA->getData()->dimensions()[0];
        const int resultCols = this->inputB->getData()->dimensions()[1];
        
        if (this->inputA->getRequiresGrad())
        {
            // std::shared_ptr<Eigen::Tensor<DT, Dim>> gradPtr = std::make_shared<Eigen::Tensor<DT, Dim>>(Eigen::Tensor<DT, Dim>(this->inputA->getData()->dimensions()));
            // for (int i = 0; i < resultRows; i++)
            // {
            //     for (int j = 0; j < resultCols; j++)
            //     {
            //         gradPtr->chip(i, 0) += (*this->gradient)(i, j) * this->inputB->getData()->chip(j, 1);
            //     }
            // }

            // this->inputA->addGrad(gradPtr);

            Eigen::array<Eigen::IndexPair<int>, 1> contractionPair = {Eigen::IndexPair<int>(Dim-1, 0)};
            auto gradA = this->gradient->contract(this->inputB->getData()->shuffle(Eigen::array<int, 2>{1, 0}), contractionPair);
            this->inputA->addGrad(std::make_shared<Eigen::Tensor<DT, Dim>>(gradA));
        }

        if (this->inputB->getRequiresGrad())
        {
            // std::shared_ptr<Eigen::Tensor<DT, Dim>> gradPtr = std::make_shared<Eigen::Tensor<DT, Dim>>(Eigen::Tensor<DT, Dim>(this->inputB->getData()->dimensions()));
            // for (int i = 0; i < resultRows; i++)
            // {
            //     for (int j = 0; j < resultCols; j++)
            //     {
            //         gradPtr->chip(j, 1) += (*this->gradient)(i, j) * this->inputA->getData()->chip(i, 0);
            //     }
            // }

            // this->inputB->addGrad(gradPtr);

            Eigen::array<Eigen::IndexPair<int>, 1> contractionPair = {Eigen::IndexPair<int>(Dim-1, 0)};
            auto gradB = this->inputA->getData()->shuffle(Eigen::array<int, 2>{1, 0}).contract(*this->gradient, contractionPair);
            this->inputB->addGrad(std::make_shared<Eigen::Tensor<DT, Dim>>(gradB));
        }
    }

    // Explicit instantiations
    template class MultTensor<2, double>; // used in tests
}