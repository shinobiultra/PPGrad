
/** @file */

#include <gtest/gtest.h>

#include "Tensor/TensorBase.hpp"
#include "Tensor/Tensor.hpp"
#include "Tensor/AddTensor.hpp"
#include "Tensor/SubTensor.hpp"
#include "Tensor/MultTensor.hpp"
#include "Tensor/AddSTensor.hpp"
#include "Tensor/SubSTensor.hpp"
#include "Tensor/MultSTensor.hpp"
#include "Tensor/DivSTensor.hpp"

#include "NumericalGradientTests.hpp"

// -------- AddTensor Tests --------

// Tests initialization of AddTensors.
TEST(AddTensorTest, ProperInitialization)
{
    Eigen::Tensor<double, 2> aT(2, 2);
    Eigen::Tensor<double, 2> bT(2, 2);
    aT.setConstant(1);
    bT.setConstant(2);
    std::shared_ptr<PPGrad::TensorBase<2, double>> a = std::make_shared<PPGrad::Tensor<2, double>>(std::make_shared<Eigen::Tensor<double, 2>>(aT));
    std::shared_ptr<PPGrad::TensorBase<2, double>> b = std::make_shared<PPGrad::Tensor<2, double>>(std::make_shared<Eigen::Tensor<double, 2>>(bT));

    std::shared_ptr<PPGrad::TensorBase<2, double>> add = a + b;
    Eigen::Tensor<double, 2> c = *add->getData();
    EXPECT_EQ(c(0, 0), 3);
    EXPECT_EQ(c(0, 1), 3);
    EXPECT_EQ(c(1, 0), 3);
    EXPECT_EQ(c(1, 1), 3);
}

// Tests Adding multiple tensors (of various types)
TEST(AddTensorTest, AdditionToVariousTypesManual)
{
    Eigen::Tensor<double, 2> aT(2, 2);
    Eigen::Tensor<double, 2> bT(2, 2);
    aT.setConstant(1);
    bT.setConstant(2);

    std::shared_ptr<PPGrad::TensorBase<2, double>> a = std::make_shared<PPGrad::Tensor<2, double>>(std::make_shared<Eigen::Tensor<double, 2>>(aT));
    std::shared_ptr<PPGrad::TensorBase<2, double>> b = std::make_shared<PPGrad::Tensor<2, double>>(std::make_shared<Eigen::Tensor<double, 2>>(bT));

    std::shared_ptr<PPGrad::TensorBase<2, double>> add_1 = a + b;

    Eigen::Tensor<double, 2> cT(2, 2);
    cT.setConstant(3);

    std::shared_ptr<PPGrad::TensorBase<2, double>> c = std::make_shared<PPGrad::Tensor<2, double>>(std::make_shared<Eigen::Tensor<double, 2>>(cT));

    std::shared_ptr<PPGrad::TensorBase<2, double>> add_2 = add_1 + c;

    std::shared_ptr<PPGrad::TensorBase<2, double>> add_3 = add_2 + c;

    Eigen::Tensor<double, 2> d = *add_3->getData();
    EXPECT_EQ(d(0, 0), 9);
    EXPECT_EQ(d(0, 1), 9);
    EXPECT_EQ(d(1, 0), 9);
    EXPECT_EQ(d(1, 1), 9);

    Eigen::Tensor<double, 2> e = *add_2->getData();
    EXPECT_EQ(e(0, 0), 6);
    EXPECT_EQ(e(0, 1), 6);
    EXPECT_EQ(e(1, 0), 6);
    EXPECT_EQ(e(1, 1), 6);

    Eigen::Tensor<double, 2> f = *add_1->getData();
    EXPECT_EQ(f(0, 0), 3);
    EXPECT_EQ(f(0, 1), 3);
    EXPECT_EQ(f(1, 0), 3);
    EXPECT_EQ(f(1, 1), 3);
}

TEST(AddTensorTest, AdditionToVariousTypesOverload)
{
    Eigen::Tensor<double, 2> aT(2, 2);
    Eigen::Tensor<double, 2> bT(2, 2);
    aT.setConstant(1);
    bT.setConstant(2);

    std::shared_ptr<PPGrad::TensorBase<2, double>> a = std::make_shared<PPGrad::Tensor<2, double>>(std::make_shared<Eigen::Tensor<double, 2>>(aT));
    std::shared_ptr<PPGrad::TensorBase<2, double>> b = std::make_shared<PPGrad::Tensor<2, double>>(std::make_shared<Eigen::Tensor<double, 2>>(bT));
    std::shared_ptr<PPGrad::TensorBase<2, double>> add_1 = a + b;

    Eigen::Tensor<double, 2> cT(2, 2);
    cT.setConstant(3);

    std::shared_ptr<PPGrad::TensorBase<2, double>> c = std::make_shared<PPGrad::Tensor<2, double>>(std::make_shared<Eigen::Tensor<double, 2>>(cT));

    std::shared_ptr<PPGrad::TensorBase<2, double>> add_2 = add_1 + c;

    std::shared_ptr<PPGrad::TensorBase<2, double>> add_3 = add_2 + c;

    Eigen::Tensor<double, 2> d = *add_3->getData();
    EXPECT_EQ(d(0, 0), 9);
    EXPECT_EQ(d(0, 1), 9);
    EXPECT_EQ(d(1, 0), 9);
    EXPECT_EQ(d(1, 1), 9);

    Eigen::Tensor<double, 2> e = *add_2->getData();
    EXPECT_EQ(e(0, 0), 6);
    EXPECT_EQ(e(0, 1), 6);
    EXPECT_EQ(e(1, 0), 6);
    EXPECT_EQ(e(1, 1), 6);

    Eigen::Tensor<double, 2> f = *add_1->getData();
    EXPECT_EQ(f(0, 0), 3);
    EXPECT_EQ(f(0, 1), 3);
    EXPECT_EQ(f(1, 0), 3);
    EXPECT_EQ(f(1, 1), 3);
}

// // Tests Gradient propagation through addition
TEST(AddTensorTest, GradientPropagationManual)
{
    Eigen::Tensor<double, 2> aT(2, 2);
    Eigen::Tensor<double, 2> bT(2, 2);
    aT.setConstant(1);
    bT.setConstant(2);

    std::shared_ptr<PPGrad::TensorBase<2, double>> a = std::make_shared<PPGrad::Tensor<2, double>>(std::make_shared<Eigen::Tensor<double, 2>>(aT), true);
    std::shared_ptr<PPGrad::TensorBase<2, double>> b = std::make_shared<PPGrad::Tensor<2, double>>(std::make_shared<Eigen::Tensor<double, 2>>(bT), true);

    std::shared_ptr<PPGrad::TensorBase<2, double>> add = a + b;

    Eigen::Tensor<double, 2> cT(2, 2);
    cT.setConstant(3);

    std::shared_ptr<PPGrad::TensorBase<2, double>> c = std::make_shared<PPGrad::Tensor<2, double>>(std::make_shared<Eigen::Tensor<double, 2>>(cT), true);

    std::shared_ptr<PPGrad::TensorBase<2, double>> add_2 = add + c;

    std::shared_ptr<PPGrad::TensorBase<2, double>> add_3 = add_2 + c;

    Eigen::Tensor<double, 2> dT(2, 2);
    dT.setConstant(1);

    std::shared_ptr<Eigen::Tensor<double, 2>> dPtr = std::make_shared<Eigen::Tensor<double, 2>>(dT);
    add_3->addGrad(dPtr);
    add_3->_backward();
    add_2->_backward();
    add->_backward();
    c->_backward();
    b->_backward();
    a->_backward();

    Eigen::Tensor<double, 2> d = *a->getGrad();
    EXPECT_EQ(d(0, 0), 1);
    EXPECT_EQ(d(0, 1), 1);
    EXPECT_EQ(d(1, 0), 1);
    EXPECT_EQ(d(1, 1), 1);

    Eigen::Tensor<double, 2> e = *b->getGrad();
    EXPECT_EQ(e(0, 0), 1);
    EXPECT_EQ(e(0, 1), 1);
    EXPECT_EQ(e(1, 0), 1);
    EXPECT_EQ(e(1, 1), 1);

    Eigen::Tensor<double, 2> f = *c->getGrad();
    EXPECT_EQ(f(0, 0), 2);
    EXPECT_EQ(f(0, 1), 2);
    EXPECT_EQ(f(1, 0), 2);
    EXPECT_EQ(f(1, 1), 2);

    Eigen::Tensor<double, 2> g = *add->getGrad();
    EXPECT_EQ(g(0, 0), 1);
    EXPECT_EQ(g(0, 1), 1);
    EXPECT_EQ(g(1, 0), 1);
    EXPECT_EQ(g(1, 1), 1);

    Eigen::Tensor<double, 2> h = *add_2->getGrad();
    EXPECT_EQ(h(0, 0), 1);
    EXPECT_EQ(h(0, 1), 1);
    EXPECT_EQ(h(1, 0), 1);
    EXPECT_EQ(h(1, 1), 1);

    Eigen::Tensor<double, 2> i = *add_3->getGrad();
    EXPECT_EQ(i(0, 0), 1);
    EXPECT_EQ(i(0, 1), 1);
    EXPECT_EQ(i(1, 0), 1);
    EXPECT_EQ(i(1, 1), 1);
}

// // -------- SubTensor Tests --------

TEST(SubTensorTest, ProperInitialization)
{
    Eigen::Tensor<double, 2> aT(2, 2);
    Eigen::Tensor<double, 2> bT(2, 2);
    aT.setConstant(1);
    bT.setConstant(2);
    std::shared_ptr<PPGrad::TensorBase<2, double>> a = std::make_shared<PPGrad::Tensor<2, double>>(std::make_shared<Eigen::Tensor<double, 2>>(aT));
    std::shared_ptr<PPGrad::TensorBase<2, double>> b = std::make_shared<PPGrad::Tensor<2, double>>(std::make_shared<Eigen::Tensor<double, 2>>(bT));

    std::shared_ptr<PPGrad::TensorBase<2, double>> sub = a - b;
    Eigen::Tensor<double, 2> c = *sub->getData();
    EXPECT_EQ(c(0, 0), -1);
    EXPECT_EQ(c(0, 1), -1);
    EXPECT_EQ(c(1, 0), -1);
    EXPECT_EQ(c(1, 1), -1);
}

TEST(SubTensorTest, SubtractionToVariousTypes)
{
    Eigen::Tensor<double, 2> aT(2, 2);
    Eigen::Tensor<double, 2> bT(2, 2);
    aT.setConstant(1);
    bT.setConstant(2);

    std::shared_ptr<PPGrad::TensorBase<2, double>> a = std::make_shared<PPGrad::Tensor<2, double>>(std::make_shared<Eigen::Tensor<double, 2>>(aT));
    std::shared_ptr<PPGrad::TensorBase<2, double>> b = std::make_shared<PPGrad::Tensor<2, double>>(std::make_shared<Eigen::Tensor<double, 2>>(bT));

    std::shared_ptr<PPGrad::TensorBase<2, double>> sub_1 = a - b;

    Eigen::Tensor<double, 2> cT(2, 2);
    cT.setConstant(3);

    std::shared_ptr<PPGrad::TensorBase<2, double>> c = std::make_shared<PPGrad::Tensor<2, double>>(std::make_shared<Eigen::Tensor<double, 2>>(cT));

    std::shared_ptr<PPGrad::TensorBase<2, double>> sub_2 = sub_1 - c;

    std::shared_ptr<PPGrad::TensorBase<2, double>> sub_3 = sub_2 - c;

    Eigen::Tensor<double, 2> d = *sub_3->getData();
    EXPECT_EQ(d(0, 0), -7);
    EXPECT_EQ(d(0, 1), -7);
    EXPECT_EQ(d(1, 0), -7);
    EXPECT_EQ(d(1, 1), -7);

    Eigen::Tensor<double, 2> e = *sub_2->getData();
    EXPECT_EQ(e(0, 0), -4);
    EXPECT_EQ(e(0, 1), -4);
    EXPECT_EQ(e(1, 0), -4);
    EXPECT_EQ(e(1, 1), -4);

    Eigen::Tensor<double, 2> f = *sub_1->getData();
    EXPECT_EQ(f(0, 0), -1);
    EXPECT_EQ(f(0, 1), -1);
    EXPECT_EQ(f(1, 0), -1);
    EXPECT_EQ(f(1, 1), -1);
}

// // Tests Gradient propagation through subtraction
TEST(SubTensorTest, GradientPropagationManual)
{
    Eigen::Tensor<double, 2> aT(2, 2);
    Eigen::Tensor<double, 2> bT(2, 2);
    aT.setConstant(1);
    bT.setConstant(2);

    std::shared_ptr<PPGrad::TensorBase<2, double>> a = std::make_shared<PPGrad::Tensor<2, double>>(std::make_shared<Eigen::Tensor<double, 2>>(aT), true); // a = 1
    std::shared_ptr<PPGrad::TensorBase<2, double>> b = std::make_shared<PPGrad::Tensor<2, double>>(std::make_shared<Eigen::Tensor<double, 2>>(bT), true); // b = 2

    std::shared_ptr<PPGrad::TensorBase<2, double>> sub = a - b; // sub = a - b = -1

    Eigen::Tensor<double, 2> cT(2, 2);
    cT.setConstant(3);

    std::shared_ptr<PPGrad::TensorBase<2, double>> c = std::make_shared<PPGrad::Tensor<2, double>>(std::make_shared<Eigen::Tensor<double, 2>>(cT), true); // c = 3

    std::shared_ptr<PPGrad::TensorBase<2, double>> sub_2 = sub - c; // sub_2 = sub - c = -4

    std::shared_ptr<PPGrad::TensorBase<2, double>> sub_3 = sub_2 - c; // sub_3 = sub_2 - c = -7

    Eigen::Tensor<double, 2> dT(2, 2);
    dT.setConstant(1);

    std::shared_ptr<Eigen::Tensor<double, 2>> dPtr = std::make_shared<Eigen::Tensor<double, 2>>(dT);
    sub_3->addGrad(dPtr);
    sub_3->_backward();
    sub_2->_backward();
    sub->_backward();
    c->_backward();
    b->_backward();
    a->_backward();

    Eigen::Tensor<double, 2> d = *a->getGrad();
    EXPECT_EQ(d(0, 0), 1);
    EXPECT_EQ(d(0, 1), 1);
    EXPECT_EQ(d(1, 0), 1);
    EXPECT_EQ(d(1, 1), 1);

    Eigen::Tensor<double, 2> e = *b->getGrad();
    EXPECT_EQ(e(0, 0), -1);
    EXPECT_EQ(e(0, 1), -1);
    EXPECT_EQ(e(1, 0), -1);
    EXPECT_EQ(e(1, 1), -1);

    Eigen::Tensor<double, 2> f = *c->getGrad();
    EXPECT_EQ(f(0, 0), -2);
    EXPECT_EQ(f(0, 1), -2);
    EXPECT_EQ(f(1, 0), -2);
    EXPECT_EQ(f(1, 1), -2);

    Eigen::Tensor<double, 2> g = *sub->getGrad();
    EXPECT_EQ(g(0, 0), 1);
    EXPECT_EQ(g(0, 1), 1);
    EXPECT_EQ(g(1, 0), 1);
    EXPECT_EQ(g(1, 1), 1);

    Eigen::Tensor<double, 2> h = *sub_2->getGrad();
    EXPECT_EQ(h(0, 0), 1);
    EXPECT_EQ(h(0, 1), 1);
    EXPECT_EQ(h(1, 0), 1);
    EXPECT_EQ(h(1, 1), 1);

    Eigen::Tensor<double, 2> i = *sub_3->getGrad();
    EXPECT_EQ(i(0, 0), 1);
    EXPECT_EQ(i(0, 1), 1);
    EXPECT_EQ(i(1, 0), 1);
    EXPECT_EQ(i(1, 1), 1);
}

// // -------- MultTensor Tests --------

TEST(MultTensorTest, ProperInitialization)
{
    Eigen::Tensor<double, 2> aT(2, 3);
    Eigen::Tensor<double, 2> bT(3, 2);
    aT.setConstant(2);
    bT.setConstant(3);
    std::shared_ptr<PPGrad::TensorBase<2, double>> a = std::make_shared<PPGrad::Tensor<2, double>>(std::make_shared<Eigen::Tensor<double, 2>>(aT));
    std::shared_ptr<PPGrad::TensorBase<2, double>> b = std::make_shared<PPGrad::Tensor<2, double>>(std::make_shared<Eigen::Tensor<double, 2>>(bT));

    std::shared_ptr<PPGrad::TensorBase<2, double>> mult = a * b;
    Eigen::Tensor<double, 2> c = *mult->getData();
    EXPECT_EQ(c(0, 0), 18);
    EXPECT_EQ(c(0, 1), 18);
    EXPECT_EQ(c(1, 0), 18);
    EXPECT_EQ(c(1, 1), 18);
}

TEST(MultTensorTest, MultiplicationToVariousTypes)
{
    Eigen::Tensor<double, 2> aT(2, 3);
    Eigen::Tensor<double, 2> bT(3, 2);
    aT.setConstant(2);
    bT.setConstant(3);

    std::shared_ptr<PPGrad::TensorBase<2, double>> a = std::make_shared<PPGrad::Tensor<2, double>>(std::make_shared<Eigen::Tensor<double, 2>>(aT)); // a = 2 (2x3)

    std::shared_ptr<PPGrad::TensorBase<2, double>> b = std::make_shared<PPGrad::Tensor<2, double>>(std::make_shared<Eigen::Tensor<double, 2>>(bT)); // b = 3 (3x2)

    std::shared_ptr<PPGrad::TensorBase<2, double>> mult_1 = a * b; // mult_1 = a * b = 18 (2x2)
    Eigen::Tensor<double, 2> cT(2, 2);
    cT.setConstant(3);

    std::shared_ptr<PPGrad::TensorBase<2, double>> c = std::make_shared<PPGrad::Tensor<2, double>>(std::make_shared<Eigen::Tensor<double, 2>>(cT)); // c = 3 (2x2)

    std::shared_ptr<PPGrad::TensorBase<2, double>> mult_2 = mult_1 * c; // mult_2 = mult_1 * c = 108 (2x2)

    std::shared_ptr<PPGrad::TensorBase<2, double>> mult_3 = mult_2 * c; // mult_3 = mult_2 * c = 648 (2x2)

    std::shared_ptr<PPGrad::TensorBase<2, double>> mult_4 = mult_3 + mult_3;

    std::shared_ptr<PPGrad::TensorBase<2, double>> mult_5 = mult_3 * 2.0;

    Eigen::Tensor<double, 2> d = *mult_5->getData();
    EXPECT_EQ(d(0, 0), 1296);
    EXPECT_EQ(d(0, 1), 1296);
    EXPECT_EQ(d(1, 0), 1296);
    EXPECT_EQ(d(1, 1), 1296);

    Eigen::Tensor<double, 2> e = *mult_4->getData();
    EXPECT_EQ(e(0, 0), 1296);
    EXPECT_EQ(e(0, 1), 1296);
    EXPECT_EQ(e(1, 0), 1296);
    EXPECT_EQ(e(1, 1), 1296);

    Eigen::Tensor<double, 2> f = *mult_3->getData();
    EXPECT_EQ(f(0, 0), 648);
    EXPECT_EQ(f(0, 1), 648);
    EXPECT_EQ(f(1, 0), 648);
    EXPECT_EQ(f(1, 1), 648);

    Eigen::Tensor<double, 2> g = *mult_2->getData();
    EXPECT_EQ(g(0, 0), 108);
    EXPECT_EQ(g(0, 1), 108);
    EXPECT_EQ(g(1, 0), 108);
    EXPECT_EQ(g(1, 1), 108);

    Eigen::Tensor<double, 2> h = *mult_1->getData();
    EXPECT_EQ(h(0, 0), 18);
    EXPECT_EQ(h(0, 1), 18);
    EXPECT_EQ(h(1, 0), 18);
    EXPECT_EQ(h(1, 1), 18);
}

// // Tests Gradient propagation through multiplication
TEST(MultTensorTest, GradientPropagationManual)
{
    // Instantiate N random tensors
    const int N = 10;
    std::vector<std::shared_ptr<PPGrad::TensorBase<2, double>>> tensors;
    std::vector<Eigen::Tensor<double, 2>> eigenTensors;
    for (int i = 0; i < N; i++)
    {
        Eigen::Tensor<double, 2> aT(5, 5);
        aT.setRandom();
        std::shared_ptr<PPGrad::TensorBase<2, double>> a = std::make_shared<PPGrad::Tensor<2, double>>(std::make_shared<Eigen::Tensor<double, 2>>(aT), true);
        tensors.push_back(a);
        eigenTensors.push_back(aT);
    }

    // Run PPGrad Tensors through our test function `LSum`
    std::vector<std::shared_ptr<PPGrad::TensorBase<2, double>>> resultTensors = LSumMult(tensors);

    // Run numerical gradient estimation on the same function
    std::vector<Eigen::Tensor<double, 2>> numericalGradients = estimateGradients(eigenTensors, [](const std::vector<Eigen::Tensor<double, 2>> &eTensors)
                                                                                 { return LSumMult(eTensors); });

    // make sure the result (sum of tensors) is correct
    Eigen::Tensor<double, 0> resultOur = resultTensors[resultTensors.size() - 1]->getData()->sum();
    double resultNumerical = LSumMult(eigenTensors);
    EXPECT_NEAR(resultOur(), resultNumerical, 1e-5);

    // Compare the gradients
    for (int i = 0; i < N; i++)
    {
        Eigen::Tensor<double, 2> grad = *tensors[i]->getGrad();
        Eigen::Tensor<double, 2> numericalGrad = numericalGradients[i];

        for (int j = 0; j < grad.dimensions()[0]; j++)
        {
            for (int k = 0; k < grad.dimensions()[1]; k++)
            {
                EXPECT_NEAR(grad(j, k), numericalGrad(j, k), 1e-5);
            }
        }
    }
}

// // -------- Mixed Tensor Tests --------

TEST(MixedTensorTest, GradientPropagationManual)
{
    // Instantiate N random tensors
    const int N = 32;
    std::vector<std::shared_ptr<PPGrad::TensorBase<2, double>>> tensors;
    std::vector<Eigen::Tensor<double, 2>> eigenTensors;
    for (int i = 0; i < N; i++)
    {
        Eigen::Tensor<double, 2> aT(5, 5);
        aT.setRandom();
        std::shared_ptr<PPGrad::TensorBase<2, double>> a = std::make_shared<PPGrad::Tensor<2, double>>(std::make_shared<Eigen::Tensor<double, 2>>(aT), true);
        tensors.push_back(a);
        eigenTensors.push_back(aT);
    }

    // Run PPGrad Tensors through our test function `LSum`
    std::vector<std::shared_ptr<PPGrad::TensorBase<2, double>>> resultTensors = LSumMixed(tensors);

    // Run numerical gradient estimation on the same function
    std::vector<Eigen::Tensor<double, 2>> numericalGradients = estimateGradients(eigenTensors, [](const std::vector<Eigen::Tensor<double, 2>> &eTensors)
                                                                                 { return LSumMixed(eTensors); });

    // make sure the result (sum of tensors) is correct
    Eigen::Tensor<double, 0> resultOur = resultTensors[resultTensors.size() - 1]->getData()->sum();
    double resultNumerical = LSumMixed(eigenTensors);
    EXPECT_NEAR(resultOur(), resultNumerical, 1e-5);

    // Compare the gradients
    for (int i = 0; i < N; i++)
    {
        Eigen::Tensor<double, 2> grad = *tensors[i]->getGrad();
        Eigen::Tensor<double, 2> numericalGrad = numericalGradients[i];

        for (int j = 0; j < grad.dimensions()[0]; j++)
        {
            for (int k = 0; k < grad.dimensions()[1]; k++)
            {
                EXPECT_NEAR(grad(j, k), numericalGrad(j, k), 1e-5);
            }
        }
    }
}

TEST(MixedTensorTest, GradientPropagationAutomatic)
{
    // Instantiate N random tensors
    const int N = 32;
    std::vector<std::shared_ptr<PPGrad::TensorBase<2, double>>> tensors;
    std::vector<Eigen::Tensor<double, 2>> eigenTensors;
    for (int i = 0; i < N; i++)
    {
        Eigen::Tensor<double, 2> aT(5, 5);
        aT.setRandom();
        std::shared_ptr<PPGrad::TensorBase<2, double>> a = std::make_shared<PPGrad::Tensor<2, double>>(std::make_shared<Eigen::Tensor<double, 2>>(aT), true);
        tensors.push_back(a);
        eigenTensors.push_back(aT);
    }

    // Run PPGrad Tensors through our test function `LSum`
    std::vector<std::shared_ptr<PPGrad::TensorBase<2, double>>> resultTensors = LSumMixed(tensors, true);

    // Run numerical gradient estimation on the same function
    std::vector<Eigen::Tensor<double, 2>> numericalGradients = estimateGradients(eigenTensors, [](const std::vector<Eigen::Tensor<double, 2>> &eTensors)
                                                                                 { return LSumMixed(eTensors); });

    // make sure the result (sum of tensors) is correct
    Eigen::Tensor<double, 0> resultOur = resultTensors[resultTensors.size() - 1]->getData()->sum();
    double resultNumerical = LSumMixed(eigenTensors);
    EXPECT_NEAR(resultOur(), resultNumerical, 1e-5);

    // Compare the gradients
    for (int i = 0; i < N; i++)
    {
        Eigen::Tensor<double, 2> grad = *tensors[i]->getGrad();
        Eigen::Tensor<double, 2> numericalGrad = numericalGradients[i];

        for (int j = 0; j < grad.dimensions()[0]; j++)
        {
            for (int k = 0; k < grad.dimensions()[1]; k++)
            {
                EXPECT_NEAR(grad(j, k), numericalGrad(j, k), 1e-5);
            }
        }
    }
}

TEST(MixedTensorTest, GradientPropagationAutomaticNonSquare)
{
    // Instantiate N random tensors
    const int N = 10;
    std::vector<std::shared_ptr<PPGrad::TensorBase<2, double>>> tensors;
    std::vector<Eigen::Tensor<double, 2>> eigenTensors;
    int initialSize = 2;
    for (int i = 0; i < N; i++)
    {
        Eigen::Tensor<double, 2> aT(initialSize + i, initialSize + i + 1);
        aT.setRandom();
        std::shared_ptr<PPGrad::TensorBase<2, double>> a = std::make_shared<PPGrad::Tensor<2, double>>(std::make_shared<Eigen::Tensor<double, 2>>(aT), true);
        tensors.push_back(a);
        eigenTensors.push_back(aT);
    }

    // Run PPGrad Tensors through our test function `LMultSumShapes`
    std::vector<std::shared_ptr<PPGrad::TensorBase<2, double>>> resultTensors = LMultSumShapes(tensors, true);

    // Run numerical gradient estimation on the same function
    std::vector<Eigen::Tensor<double, 2>> numericalGradients = estimateGradients(eigenTensors, [](const std::vector<Eigen::Tensor<double, 2>> &eTensors)
                                                                                 { return LMultSumShapes(eTensors); });

    // make sure the result (sum of tensors) is correct
    Eigen::Tensor<double, 0> resultOur = resultTensors[resultTensors.size() - 1]->getData()->sum();
    double resultNumerical = LMultSumShapes(eigenTensors);
    EXPECT_NEAR(resultOur(), resultNumerical, 1e-5);

    // Compare the gradients
    for (int i = 0; i < N; i++)
    {
        Eigen::Tensor<double, 2> grad = *tensors[i]->getGrad();
        Eigen::Tensor<double, 2> numericalGrad = numericalGradients[i];

        for (int j = 0; j < grad.dimensions()[0]; j++)
        {
            for (int k = 0; k < grad.dimensions()[1]; k++)
            {
                EXPECT_NEAR(grad(j, k), numericalGrad(j, k), 1); // we have to use a larger tolerance here because the numerical gradient is not very accurate (hopefully lol)
            }
        }
    }
}