#include <gtest/gtest.h>

#include "Tensor/TensorBase.hpp"
#include "Tensor/Tensor.hpp"
#include "Tensor/AddTensor.hpp"


// Tests factorial of 0.
TEST(AddTensorTest, ProperInitialization) {
    Eigen::Tensor<double, 2> aT(2, 2);
    Eigen::Tensor<double, 2> bT(2, 2);
    aT.setConstant(1);
    bT.setConstant(2);
    PPGrad::Tensor<2, double> a(std::make_shared<Eigen::Tensor<double, 2>>(aT));
    PPGrad::Tensor<2, double> b(std::make_shared<Eigen::Tensor<double, 2>>(bT));

    PPGrad::AddTensor<2, double> add(std::make_shared<PPGrad::Tensor<2, double>>(a),
                                     std::make_shared<PPGrad::Tensor<2, double>>(b));
    Eigen::Tensor<double, 2> c = *add.getData();
    EXPECT_EQ(c(0, 0), 3);
    EXPECT_EQ(c(0, 1), 3);
    EXPECT_EQ(c(1, 0), 3);
    EXPECT_EQ(c(1, 1), 3);
}

// Tests Adding multiple tensors (of various types)
TEST(AddTensorTest, AdditionToVariousTypes) {
    Eigen::Tensor<double, 2> aT(2, 2);
    Eigen::Tensor<double, 2> bT(2, 2);
    aT.setConstant(1);
    bT.setConstant(2);

    PPGrad::Tensor<2, double> a(std::make_shared<Eigen::Tensor<double, 2>>(aT));
    PPGrad::Tensor<2, double> b(std::make_shared<Eigen::Tensor<double, 2>>(bT));

    PPGrad::AddTensor<2, double> add_1(std::make_shared<PPGrad::Tensor<2, double>>(a),
                                       std::make_shared<PPGrad::Tensor<2, double>>(b));
    
    Eigen::Tensor<double, 2> cT(2, 2);
    cT.setConstant(3);

    PPGrad::Tensor<2, double> c(std::make_shared<Eigen::Tensor<double, 2>>(cT));
    
    PPGrad::AddTensor<2, double> add_2(std::make_shared<PPGrad::Tensor<2, double>>(add_1),
                                       std::make_shared<PPGrad::Tensor<2, double>>(c));

    PPGrad::AddTensor<2, double> add_3(std::make_shared<PPGrad::Tensor<2, double>>(add_2),
                                       std::make_shared<PPGrad::Tensor<2, double>>(c));

    Eigen::Tensor<double, 2> d = *add_3.getData();
    EXPECT_EQ(d(0, 0), 9);
    EXPECT_EQ(d(0, 1), 9);
    EXPECT_EQ(d(1, 0), 9);
    EXPECT_EQ(d(1, 1), 9);

    Eigen::Tensor<double, 2> e = *add_2.getData();
    EXPECT_EQ(e(0, 0), 6);
    EXPECT_EQ(e(0, 1), 6);
    EXPECT_EQ(e(1, 0), 6);
    EXPECT_EQ(e(1, 1), 6);

    Eigen::Tensor<double, 2> f = *add_1.getData();
    EXPECT_EQ(f(0, 0), 3);
    EXPECT_EQ(f(0, 1), 3);
    EXPECT_EQ(f(1, 0), 3);
    EXPECT_EQ(f(1, 1), 3);        
}


// Tests Gradient propagation through addition
TEST(AddTensorTest, GradientPropagationManual) {
    Eigen::Tensor<double, 2> aT(2, 2);
    Eigen::Tensor<double, 2> bT(2, 2);
    aT.setConstant(1);
    bT.setConstant(2);

    PPGrad::Tensor<2, double> a(std::make_shared<Eigen::Tensor<double, 2>>(aT), true);
    PPGrad::Tensor<2, double> b(std::make_shared<Eigen::Tensor<double, 2>>(bT), true);

    PPGrad::AddTensor<2, double> add(std::make_shared<PPGrad::Tensor<2, double>>(a),
                                     std::make_shared<PPGrad::Tensor<2, double>>(b), true);
    
    Eigen::Tensor<double, 2> cT(2, 2);
    cT.setConstant(3);

    PPGrad::Tensor<2, double> c(std::make_shared<Eigen::Tensor<double, 2>>(cT), true);
    
    PPGrad::AddTensor<2, double> add_2(std::make_shared<PPGrad::Tensor<2, double>>(add),
                                       std::make_shared<PPGrad::Tensor<2, double>>(c), true);

    PPGrad::AddTensor<2, double> add_3(std::make_shared<PPGrad::Tensor<2, double>>(add_2),
                                       std::make_shared<PPGrad::Tensor<2, double>>(c), true);

    Eigen::Tensor<double, 2> dT(2, 2);
    dT.setConstant(1);

    std::shared_ptr<Eigen::Tensor<double, 2>> dPtr = std::make_shared<Eigen::Tensor<double, 2>>(dT);
    add_3.addGrad(dPtr);
    add_3.backward();
    add_2.backward();
    add.backward();
    c.backward();
    b.backward();
    a.backward();

    Eigen::Tensor<double, 2> d = *a.getGrad();
    EXPECT_EQ(d(0, 0), 1);
    EXPECT_EQ(d(0, 1), 1);
    EXPECT_EQ(d(1, 0), 1);
    EXPECT_EQ(d(1, 1), 1);

    Eigen::Tensor<double, 2> e = *b.getGrad();
    EXPECT_EQ(e(0, 0), 1);
    EXPECT_EQ(e(0, 1), 1);
    EXPECT_EQ(e(1, 0), 1);
    EXPECT_EQ(e(1, 1), 1);

    Eigen::Tensor<double, 2> f = *c.getGrad();
    EXPECT_EQ(f(0, 0), 2);
    EXPECT_EQ(f(0, 1), 2);
    EXPECT_EQ(f(1, 0), 2);
    EXPECT_EQ(f(1, 1), 2);

    Eigen::Tensor<double, 2> g = *add.getGrad();
    EXPECT_EQ(g(0, 0), 1);
    EXPECT_EQ(g(0, 1), 1);
    EXPECT_EQ(g(1, 0), 1);
    EXPECT_EQ(g(1, 1), 1);

    Eigen::Tensor<double, 2> h = *add_2.getGrad();
    EXPECT_EQ(h(0, 0), 1);
    EXPECT_EQ(h(0, 1), 1);
    EXPECT_EQ(h(1, 0), 1);
    EXPECT_EQ(h(1, 1), 1);

    Eigen::Tensor<double, 2> i = *add_3.getGrad();
    EXPECT_EQ(i(0, 0), 1);
    EXPECT_EQ(i(0, 1), 1);
    EXPECT_EQ(i(1, 0), 1);
    EXPECT_EQ(i(1, 1), 1);

    Eigen::Tensor<double, 2> j = *dPtr;
    EXPECT_EQ(j(0, 0), 1);
    EXPECT_EQ(j(0, 1), 1);
    EXPECT_EQ(j(1, 0), 1);
    EXPECT_EQ(j(1, 1), 1);
}

