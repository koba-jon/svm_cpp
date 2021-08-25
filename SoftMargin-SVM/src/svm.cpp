#include <iostream>
#include <string>
#include <vector>
#include <cmath>
// Original
#include "svm.hpp"


// --------------------------------------
// class{SoftMargin_SVM} -> constructor
// --------------------------------------
SoftMargin_SVM::SoftMargin_SVM(const bool verbose_){
    this->verbose = verbose_;
}


// ----------------------------------------
// class{SoftMargin_SVM} -> function{dot}
// ----------------------------------------
double SoftMargin_SVM::dot(const std::vector<double> x1, const std::vector<double> x2){
    
    size_t i;
    double ans;

    if (x1.size() != x2.size()){
        std::cerr << "Error : Couldn't match the number of elements for inner product." << std::endl;
        std::exit(-1);
    }

    ans = 0.0;
    for (i = 0; i < x1.size(); i++){
        ans += x1[i] * x2[i];
    }

    return ans;
    
}


// ----------------------------------------
// class{SoftMargin_SVM} -> function{log}
// ----------------------------------------
void SoftMargin_SVM::log(const std::string str){
    if (this->verbose){
        std::cout << str << std::flush;
    }
    return;
}


// ------------------------------------------
// class{SoftMargin_SVM} -> function{train}
// ------------------------------------------
void SoftMargin_SVM::train(const std::vector<std::vector<double>> class1_data, const std::vector<std::vector<double>> class2_data, const size_t D, const double C, const double lr, const double limit){

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
                item1 += alpha[j] * (double)y[i] * (double)y[j] * this->dot(x[i], x[j]);
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

    // (4.2) Description for w
    this->log("weight = [ ");
    this->w = std::vector<double>(D, 0.0);
    for (j = 0; j < D; j++){
        for (i = 0; i < Ns; i++){
            this->w[j] += alpha_s[i] * (double)ys[i] * xs[i][j];
        }
        for (i = 0; i < Ns_in; i++){
            this->w[j] += alpha_s_in[i] * (double)ys_in[i] * xs_in[i][j];
        }
        this->log(std::to_string(this->w[j]) + " ");
    }
    this->log("]\n");

    // (4.3) Description for b
    this->b = 0.0;
    for (i = 0; i < Ns; i++){
        this->b += (double)this->ys[i] - this->dot(this->w, this->xs[i]);
    }
    this->b /= (double)Ns;
    this->log("bias = " + std::to_string(this->b) + "\n");
    this->log("////////////////////////////////////////////////////////\n\n");

    return;
}


// -----------------------------------------
// class{SoftMargin_SVM} -> function{test}
// -----------------------------------------
void SoftMargin_SVM::test(const std::vector<std::vector<double>> class1_data, const std::vector<std::vector<double>> class2_data){

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


// --------------------------------------
// class{SoftMargin_SVM} -> function{f}
// --------------------------------------
double SoftMargin_SVM::f(const std::vector<double> x){
    return this->dot(this->w, x) + this->b;    
}


// --------------------------------------
// class{SoftMargin_SVM} -> function{g}
// --------------------------------------
double SoftMargin_SVM::g(const std::vector<double> x){

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