/**
 * @file NumericalGradientTests.cpp
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2023-12-06
 *
 * @copyright Copyright (c) 2023
 *
 * Utility file intended to implement numerical gradient checking for tensors.
 * This is a very slow process and intended only for testing purposes.
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
    std::vector<Eigen::Tensor<double, 2>> &inputTensors,
    std::function<double(std::vector<Eigen::Tensor<double, 2>>)> L)
{

    std::vector<Eigen::Tensor<double, 2>> gradients;
    double epsilon = 1e-7;

    // Iterate over each tensor
    for (auto &tensor : inputTensors)
    {
        Eigen::Tensor<double, 2> gradTensor = tensor.constant(0.0);

        // Iterate over each element in the tensor
        for (int i = 0; i < tensor.dimension(0); ++i)
        {
            for (int j = 0; j < tensor.dimension(1); ++j)
            {

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

// ----- LSumMult -----

double LSumMult(std::vector<Eigen::Tensor<double, 2>> inputTensors)
{
    Eigen::Tensor<double, 2> T = inputTensors[0].constant(0.0);
    const int numTensors = inputTensors.size();
    for (int i = 0; i < numTensors; i++)
    {
        if (i % 2 == 0)
        {
            T = T + inputTensors[i];
        }
        else
        {
            // contratc T with inputTensors[i]
            Eigen::array<Eigen::IndexPair<int>, 1> contractionPair = {Eigen::IndexPair<int>(1, 0)};
            Eigen::Tensor<double, 2> res = T.contract(inputTensors[i], contractionPair);
            T = res;
        }
    }
    Eigen::Tensor<double, 0> tensorSum = T.sum();
    return tensorSum();
}

std::vector<std::shared_ptr<PPGrad::TensorBase<2, double>>> LSumMult(std::vector<std::shared_ptr<PPGrad::TensorBase<2, double>>> inputTensors)
{
    std::vector<std::shared_ptr<PPGrad::TensorBase<2, double>>> resultTensors;

    Eigen::Tensor<double, 2> eT = inputTensors[0]->getData()->constant(0.0);
    std::shared_ptr<PPGrad::TensorBase<2, double>> T = std::make_shared<PPGrad::Tensor<2, double>>(std::make_shared<Eigen::Tensor<double, 2>>(eT));
    const int numTensors = inputTensors.size();
    for (int i = 0; i < numTensors; i++)
    {
        if (i % 2 == 0)
        {
            std::shared_ptr<PPGrad::TensorBase<2, double>> Q = T + inputTensors[i];
            resultTensors.push_back(Q);
            T = Q;
        }
        else
        {
            std::shared_ptr<PPGrad::TensorBase<2, double>> Q = T * inputTensors[i];
            resultTensors.push_back(Q);
            T = Q;
        }
    }

    // initialize gradient of the last tensor to 1
    resultTensors[resultTensors.size() - 1]->getGrad()->setConstant(1.0);

    // call _backward on all tensors from the end
    for (int i = resultTensors.size() - 1; i >= 0; i--)
    {
        resultTensors[i]->_backward();
        inputTensors[i]->_backward();
    }

    return resultTensors;
}

// ----- LSumMixed -----

double LSumMixed(std::vector<Eigen::Tensor<double, 2>> inputTensors)
{
    Eigen::Tensor<double, 2> T = inputTensors[0].constant(0.0);
    const int numTensors = inputTensors.size();
    for (int i = 0; i < numTensors; i++)
    {
        if (i % 4 == 0)
        {
            T = T + inputTensors[i];
        }
        else if (i % 4 == 1)
        {
            // contratc T with inputTensors[i]
            Eigen::array<Eigen::IndexPair<int>, 1> contractionPair = {Eigen::IndexPair<int>(1, 0)};
            Eigen::Tensor<double, 2> res = T.contract(inputTensors[i], contractionPair);
            T = res;
        }
        else if (i % 4 == 2)
        {
            T = T - inputTensors[i] * (double)i;
        }
        else
        {
            // contratc T with inputTensors[i]
            Eigen::array<Eigen::IndexPair<int>, 1> contractionPair = {Eigen::IndexPair<int>(1, 0)};
            Eigen::Tensor<double, 2> res = T.contract(inputTensors[i] / (double)i, contractionPair);
            T = res;
        }
    }
    Eigen::Tensor<double, 0> tensorSum = T.sum();
    return tensorSum();
}

std::vector<std::shared_ptr<PPGrad::TensorBase<2, double>>> LSumMixed(std::vector<std::shared_ptr<PPGrad::TensorBase<2, double>>> inputTensors)
{
    std::vector<std::shared_ptr<PPGrad::TensorBase<2, double>>> resultTensors;

    Eigen::Tensor<double, 2> eT = inputTensors[0]->getData()->constant(0.0);
    std::shared_ptr<PPGrad::TensorBase<2, double>> T = std::make_shared<PPGrad::Tensor<2, double>>(std::make_shared<Eigen::Tensor<double, 2>>(eT));
    const int numTensors = inputTensors.size();
    for (int i = 0; i < numTensors; i++)
    {
        if (i % 4 == 0)
        {
            std::shared_ptr<PPGrad::TensorBase<2, double>> Q = T + inputTensors[i];
            resultTensors.push_back(Q);
            resultTensors.push_back(Q); // To keep the indexing resultTensors vs inputTensors consistent
            T = Q;
        }
        else if (i % 4 == 1)
        {
            std::shared_ptr<PPGrad::TensorBase<2, double>> Q = T * inputTensors[i];
            resultTensors.push_back(Q);
            resultTensors.push_back(Q); // To keep the indexing resultTensors vs inputTensors consistent
            T = Q;
        }
        else if (i % 4 == 2)
        {
            std::shared_ptr<PPGrad::TensorBase<2, double>> Ii = (inputTensors[i] * (double)i);
            std::shared_ptr<PPGrad::TensorBase<2, double>> Q = T - Ii;
            resultTensors.push_back(Ii);
            resultTensors.push_back(Q);
            T = Q;
        }
        else
        {
            std::shared_ptr<PPGrad::TensorBase<2, double>> Ii = (inputTensors[i] / (double)i);
            std::shared_ptr<PPGrad::TensorBase<2, double>> Q = T * Ii;
            resultTensors.push_back(Ii);
            resultTensors.push_back(Q);
            T = Q;
        }
    }

    // initialize gradient of the last tensor to 1
    resultTensors[resultTensors.size() - 1]->getGrad()->setConstant(1.0);

    // call _backward on all tensors from the end
    for (int i = inputTensors.size() - 1; i >= 0; i--)
    {
        int resultTensorIndex = i * 2 + 1;
        resultTensors[resultTensorIndex]->_backward();
        if (i % 4 == 3 || i % 4 == 2)
        {
            resultTensors[resultTensorIndex - 1]->_backward();
        }

        inputTensors[i]->_backward();
    }

    return resultTensors;
}