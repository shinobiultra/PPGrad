#pragma once
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <vector>


/// @brief Numerical estimate gradients of L wrt all items in inputTensors.
/// @param inputTensors Inputs of L.
/// @param L Scalar function of inputTensors to differentiate.
/// @return List of gradients of L wrt each item in inputTensors (i.e., same size as inputTensors with each item being the same size as the corresponding item in inputTensors
std::vector<Eigen::Tensor<double, 2>> estimateGradients(
    std::vector<Eigen::Tensor<double, 2>>& inputTensors,
    std::function<double(std::vector<Eigen::Tensor<double, 2>>)> L
);



double LSum(std::vector<Eigen::Tensor<double, 2>> inputTensors);


std::vector<std::shared_ptr<PPGrad::TensorBase<2, double>>> LSum(std::vector<std::shared_ptr<PPGrad::TensorBase<2, double>>> inputTensors);