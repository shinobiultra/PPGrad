
/** @file */


#pragma once
#include "Tensor/TensorBase.hpp"
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>
#include <memory>

/// @brief Numerical estimate gradients of L wrt all items in inputTensors.
/// @param inputTensors Inputs of L.
/// @param L Scalar function of inputTensors to differentiate.
/// @return List of gradients of L wrt each item in inputTensors (i.e., same size as inputTensors with each item being the same size as the corresponding item in inputTensors
std::vector<Eigen::Tensor<double, 2>> estimateGradients(
    std::vector<Eigen::Tensor<double, 2>> &inputTensors,
    std::function<double(std::vector<Eigen::Tensor<double, 2>>)> L);

/// @brief Function that interleaves addition & multiplication of tensors, intended for testing gradients.
/// @param inputTensors Tensors to add and multiply.
/// @return Sum of the final tensor.
double LSumMult(std::vector<Eigen::Tensor<double, 2>> inputTensors);

/// @brief Function that interleaves addition & multiplication of tensors, intended for testing gradients. Calls _bakcward() on each tensor manually.
/// @param inputTensors Tensors to add and multiply.
/// @param autoBackward Whether or not to call _backward() on each tensor automatically.
/// @return Vector of intermediate tensors.
std::vector<std::shared_ptr<PPGrad::TensorBase<2, double>>> LSumMult(std::vector<std::shared_ptr<PPGrad::TensorBase<2, double>>> inputTensors, bool autoBackward = false);

/// @brief Function for thoroughly testing gradients.
/// @param inputTensors Tensors to add, multiply and scalar multiply/divide.
/// @return Sum of the final tensor.
double LSumMixed(std::vector<Eigen::Tensor<double, 2>> inputTensors);

/// @brief Function for thoroughly testing gradients.
/// @param inputTensors Tensors to add, multiply and scalar multiply/divide.
/// @param autoBackward Whether or not to call _backward() on each tensor automatically.
/// @return Vector of intermediate tensors.
std::vector<std::shared_ptr<PPGrad::TensorBase<2, double>>> LSumMixed(std::vector<std::shared_ptr<PPGrad::TensorBase<2, double>>> inputTensors, bool autoBackward = false);


/// @brief Multiply Tensors (T[i] * T[i+1]) with some scalar addition sparkled in and sum the result.
/// @param inputTensors Tensors to multiply.
/// @return Sum of the final tensor.
double LMultSumShapes(std::vector<Eigen::Tensor<double, 2>> inputTensors);

/// @brief Multiply Tensors (T[i] * T[i+1]) with some scalar addition sparkled in and sum the result.
/// @param inputTensors Tensors to multiply.
/// @param autoBackward Whether or not to call _backward() on each tensor automatically.
/// @return Vector of intermediate tensors.
std::vector<std::shared_ptr<PPGrad::TensorBase<2, double>>> LMultSumShapes(std::vector<std::shared_ptr<PPGrad::TensorBase<2, double>>> inputTensors, bool autoBackward = false);