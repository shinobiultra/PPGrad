/*
    Utility file intended to implement numerical gradient checking for tensors.
    This is a very slow process and intended only for testing purposes.
*/

#include "Tensor/Tensor.hpp"
#include "Tensor/TensorBase.hpp"


#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <vector>
#include <memory>
#include <iostream>


// @brief Numerical estimate gradients of L wrt all items in inputTensors.
/// @param inputTensors Inputs of L.
/// @param L Scalar function of inputTensors to differentiate.
/// @return List of gradients of L wrt each item in inputTensors (i.e., same size as inputTensors with each item being the same size as the corresponding item in inputTensorss
std::vector<Eigen::Tensor<double, 2>> estimateGradients(
    std::vector<Eigen::Tensor<double, 2>>& inputTensors,
    std::function<double(std::vector<Eigen::Tensor<double, 2>>)> L) {

    std::vector<Eigen::Tensor<double, 2>> gradients;
    double epsilon = 1e-7;

    // Iterate over each tensor
    for (auto& tensor : inputTensors) {
        Eigen::Tensor<double, 2> gradTensor = tensor.constant(0.0);

        // Iterate over each element in the tensor
        for (int i = 0; i < tensor.dimension(0); ++i) {
            for (int j = 0; j < tensor.dimension(1); ++j) {

                // Save original value
                double originalValue = tensor(i, j);

                // Perturb the current element
                tensor(i, j) = originalValue + epsilon;
                
                // Calculate function value with perturbed input
                double L_perturbed = L(inputTensors);

                // Restore original value
                tensor(i, j) = originalValue;

                // Calculate function value with original input
                double L_original = L(inputTensors);

                // Estimate gradient
                double gradient = (L_perturbed - L_original) / epsilon;

                // Store in gradient tensor
                gradTensor(i, j) = gradient;
            }
        }

        // Store the gradient tensor
        gradients.push_back(gradTensor);
    }

    return gradients;
}


double LSum(std::vector<Eigen::Tensor<double, 2>> inputTensors) {
    Eigen::Tensor<double, 2> T = inputTensors[0].constant(0.0);
    const int numTensors = inputTensors.size();
    for (int i = 0; i < numTensors; i++) {
        if (i % 2 == 0){
            T = T + inputTensors[i];
        } else {
            // contratc T with inputTensors[i]
            Eigen::array<Eigen::IndexPair<int>, 1> contractionPair = {Eigen::IndexPair<int>(1, 0)};
            Eigen::Tensor<double, 2> res = T.contract(inputTensors[i], contractionPair);
            T = res;
        }
    }
    Eigen::Tensor<double, 0> tensorSum = T.sum();
    return tensorSum();
}

std::vector<std::shared_ptr<PPGrad::TensorBase<2, double>>> LSum(std::vector<std::shared_ptr<PPGrad::TensorBase<2, double>>> inputTensors) {
    std::vector<std::shared_ptr<PPGrad::TensorBase<2, double>>> resultTensors;
    
    Eigen::Tensor<double, 2> eT = inputTensors[0]->getData()->constant(0.0);
    std::shared_ptr<PPGrad::TensorBase<2, double>> T = std::make_shared<PPGrad::Tensor<2, double>>(std::make_shared<Eigen::Tensor<double, 2>>(eT));
    const int numTensors = inputTensors.size();
    for (int i = 0; i < numTensors; i++) {
        if (i % 2 == 0){
            std::shared_ptr<PPGrad::TensorBase<2, double>> Q = T + inputTensors[i];
            resultTensors.push_back(Q);
            T = Q;
        } else {
            std::shared_ptr<PPGrad::TensorBase<2, double>> Q = T * inputTensors[i];
            resultTensors.push_back(Q);
            T = Q;
        }
    }
    
    // initialize gradient of the last tensor to 1
    resultTensors[resultTensors.size() - 1]->getGrad()->setConstant(1.0);

    // call backward on all tensors from the end
    for (int i = resultTensors.size() - 1; i >= 0; i--) {
        resultTensors[i]->backward();
        inputTensors[i]->backward();
    }

    return resultTensors;
}