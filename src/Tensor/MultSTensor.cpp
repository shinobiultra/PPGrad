#include "Tensor/MultSTensor.hpp"

#include <eigen3/unsupported/Eigen/CXX11/Tensor>

namespace PPGrad {

    /// @brief Calculate the gradient of this tensor with respect to it's inputs.
    template <int Dim, typename DT>
    void MultSTensor<Dim, DT>::backward()
    {
        // C = A * b
        // dC/dA = b (shape of A)
        // dC/db = A (shape of b)
        if (this->inputA->getRequiresGrad())
        {
            std::shared_ptr<Eigen::Tensor<DT, Dim>> gradPtr = std::make_shared<Eigen::Tensor<DT, Dim>>(*this->gradient * this->inputB);
            this->inputA->addGrad(gradPtr);
        }
    }

    // Explicit instantiations
    template class MultSTensor<2, double>; // used in tests
}