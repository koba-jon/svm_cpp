#include <iostream>
#include <string>
#include <vector>
#include <functional>
#include <cmath>
// Original
#include "svm.hpp"


// ---------------------------------------
// namespace{kernel} -> function{linear}
// ---------------------------------------
double kernel::linear(const std::vector<double> x1, const std::vector<double> x2, const std::vector<double> params){

    size_t i;
    double ans;

    if (x1.size() != x2.size()){
        std::cerr << "Error : Don't match the number of elements for inner product." << std::endl;
        std::exit(-1);
    }

    ans = 0.0;
    for (i = 0; i < x1.size(); i++){
        ans += x1[i] * x2[i];
    }

    return ans;

}


// -------------------------------------------
// namespace{kernel} -> function{polynomial}
// -------------------------------------------
double kernel::polynomial(const std::vector<double> x1, const std::vector<double> x2, const std::vector<double> params){

    size_t i;
    double ans;

    if (x1.size() != x2.size()){
        std::cerr << "Error : Don't match the number of elements for inner product." << std::endl;
        std::exit(-1);
    }
    else if (params.size() != 2){
        std::cerr << "Error : Don't match the number of hyper-parameters." << std::endl;
        std::exit(-1);
    }

    ans = 0.0;
    for (i = 0; i < x1.size(); i++){
        ans += x1[i] * x2[i];
    }
    ans += params[0];
    ans = std::pow(ans, params[1]);

    return ans;

}


// ------------------------------------
// namespace{kernel} -> function{rbf}
// ------------------------------------
double kernel::rbf(const std::vector<double> x1, const std::vector<double> x2, const std::vector<double> params){

    size_t i;
    double ans;

    if (x1.size() != x2.size()){
        std::cerr << "Error : Don't match the number of elements for inner product." << std::endl;
        std::exit(-1);
    }
    else if (params.size() != 1){
        std::cerr << "Error : Don't match the number of hyper-parameters." << std::endl;
        std::exit(-1);
    }

    ans = 0.0;
    for (i = 0; i < x1.size(); i++){
        ans += (x1[i] - x2[i]) * (x1[i] - x2[i]);
    }
    ans = std::exp(-params[0] * ans);

    return ans;

}


// ----------------------------------
// class{Kernel_SVM} -> constructor
// ----------------------------------
Kernel_SVM::Kernel_SVM(const std::function<double(const std::vector<double>, const std::vector<double>, const std::vector<double>)> K_, const std::vector<double> params_, const bool verbose_){
    this->K = K_;
    this->params = params_;
    this->verbose = verbose_;
}


// ------------------------------------
// class{Kernel_SVM} -> function{log}
// ------------------------------------
void Kernel_SVM::log(const std::string str){
    if (this->verbose){
        std::cout << str << std::flush;
    }
    return;
}


// --------------------------------------
// class{Kernel_SVM} -> function{train}
// --------------------------------------
void Kernel_SVM::train(const std::vector<std::vector<double>> class1_data, const std::vector<std::vector<double>> class2_data, const size_t D, const double C, const double lr, const double limit){

    constexpr double eps = 0.0000001;

    size_t i, j;
    size_t N, Ns, Ns_in;
    bool judge;
    double item1, item2, item3;
    double delta;
    double beta;
    double error;
    std::vector<std::vector<double>> x;
    std::vector<int> y;
    std::vector<double> alpha;

    // (1.1) Set class 1 data
    for (i = 0; i < class1_data.size(); i++){
        x.push_back(class1_data[i]);
        y.push_back(1);
    }

    // (1.2) Set class 2 data
    for (i = 0; i < class2_data.size(); i++){
        x.push_back(class2_data[i]);
        y.push_back(-1);
    }

    // (2) Set Lagrange Multiplier and Parameters
    N = x.size();
    alpha = std::vector<double>(N, 0.0);
    beta = 1.0;

    // (3) Training
    this->log("\n");
    this->log("/////////////////////// Training ///////////////////////\n");
    do {

        judge = false;
        error = 0.0;

        // (3.1) Update Alpha
        for (i = 0; i < N; i++){

            // Set item 1
            item1 = 0.0;
            for (j = 0; j < N; j++){
                item1 += alpha[j] * (double)y[i] * (double)y[j] * this->K(x[i], x[j], this->params);
            }

            // Set item 2
            item2 = 0.0;
            for (j = 0; j < N; j++){
                item2 += alpha[j] * (double)y[i] * (double)y[j];
            }
            
            // Set Delta
            delta = 1.0 - item1 - beta * item2;

            // Update
            alpha[i] += lr * delta;
            if (alpha[i] < 0.0){
                alpha[i] = 0.0;
            }
            else if (alpha[i] > C){
                alpha[i] = C;
            }
            else if (std::abs(delta) > limit){
                judge = true;
                error += std::abs(delta) - limit;
            }

        }

        // (3.2) Update Beta
        item3 = 0.0;
        for (i = 0; i < N; i++){
            item3 += alpha[i] * (double)y[i];
        }
        beta += item3 * item3 / 2.0;

        // (3.3) Output Residual Error
        this->log("\rerror: " + std::to_string(error));

    }while (judge);
    this->log("\n");
    this->log("////////////////////////////////////////////////////////\n");

    // (4.1) Description for support vectors
    Ns = 0;
    Ns_in = 0;
    this->xs = std::vector<std::vector<double>>();
    this->ys = std::vector<int>();
    this->alpha_s = std::vector<double>();
    this->xs_in = std::vector<std::vector<double>>();
    this->ys_in = std::vector<int>();
    this->alpha_s_in = std::vector<double>();
    for (i = 0; i < N; i++){
        if ((eps < alpha[i]) && (alpha[i] < C - eps)){
            this->xs.push_back(x[i]);
            this->ys.push_back(y[i]);
            this->alpha_s.push_back(alpha[i]);
            Ns++;
        }
        else if (alpha[i] >= C - eps){
            this->xs_in.push_back(x[i]);
            this->ys_in.push_back(y[i]);
            this->alpha_s_in.push_back(alpha[i]);
            Ns_in++;
        }
    }
    this->log("Ns (number of support vectors on margin) = " + std::to_string(Ns) + "\n");
    this->log("Ns_in (number of support vectors inside margin) = " + std::to_string(Ns_in) + "\n");

    // (4.2) Description for b
    this->b = 0.0;
    for (i = 0; i < Ns; i++){
        this->b += (double)this->ys[i];
        for (j = 0; j < Ns; j++){
            this->b -= this->alpha_s[j] * (double)this->ys[j] * this->K(this->xs[j], this->xs[i], this->params);
        }
        for (j = 0; j < Ns_in; j++){
            this->b -= this->alpha_s_in[j] * (double)this->ys_in[j] * this->K(this->xs_in[j], this->xs[i], this->params);
        }
    }
    this->b /= (double)Ns;
    this->log("bias = " + std::to_string(this->b) + "\n");
    this->log("////////////////////////////////////////////////////////\n\n");

    return;
}


// -------------------------------------
// class{Kernel_SVM} -> function{test}
// -------------------------------------
void Kernel_SVM::test(const std::vector<std::vector<double>> class1_data, const std::vector<std::vector<double>> class2_data){

    size_t i;

    this->correct_c1 = 0;
    for (i = 0; i < class1_data.size(); i++){
        if (this->g(class1_data[i]) == 1){
            this->correct_c1++;
        }
    }

    this->correct_c2 = 0;
    for (i = 0; i < class2_data.size(); i++){
        if (this->g(class2_data[i]) == -1){
            this->correct_c2++;
        }
    }

    this->accuracy = (double)(this->correct_c1 + this->correct_c2) / (double)(class1_data.size() + class2_data.size());
    this->accuracy_c1 = (double)this->correct_c1 / (double)class1_data.size();
    this->accuracy_c2 = (double)this->correct_c2 / (double)class2_data.size();

    return;
}


// ----------------------------------
// class{Kernel_SVM} -> function{f}
// ----------------------------------
double Kernel_SVM::f(const std::vector<double> x){

    size_t i;
    double ans;

    ans = 0.0;
    for (i = 0; i < this->xs.size(); i++){
        ans += this->alpha_s[i] * this->ys[i] * this->K(this->xs[i], x, this->params);
    }
    for (i = 0; i < this->xs_in.size(); i++){
        ans += this->alpha_s_in[i] * this->ys_in[i] * this->K(this->xs_in[i], x, this->params);
    }
    ans += this->b;
    
    return ans;    
}


// ----------------------------------
// class{Kernel_SVM} -> function{g}
// ----------------------------------
double Kernel_SVM::g(const std::vector<double> x){

    double fx;
    int gx;

    fx = this->f(x);
    if (fx >= 0.0){
        gx = 1;
    }
    else{
        gx = -1;
    }
    
    return gx;    
}