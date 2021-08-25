#include <iostream>
#include <string>
#include <vector>
#include <utility>
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
        std::cerr << "Error : Couldn't match the number of elements for inner product." << std::endl;
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
        std::cerr << "Error : Couldn't match the number of elements for inner product." << std::endl;
        std::exit(-1);
    }
    else if (params.size() != 2){
        std::cerr << "Error : Couldn't match the number of hyper-parameters." << std::endl;
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
        std::cerr << "Error : Couldn't match the number of elements for inner product." << std::endl;
        std::exit(-1);
    }
    else if (params.size() != 1){
        std::cerr << "Error : Couldn't match the number of hyper-parameters." << std::endl;
        std::exit(-1);
    }

    ans = 0.0;
    for (i = 0; i < x1.size(); i++){
        ans += (x1[i] - x2[i]) * (x1[i] - x2[i]);
    }
    ans = std::exp(-params[0] * ans);

    return ans;

}


// ------------------------------
// class{OC_SVM} -> constructor
// ------------------------------
OC_SVM::OC_SVM(const std::function<double(const std::vector<double>, const std::vector<double>, const std::vector<double>)> K_, const std::vector<double> params_, const bool verbose_){
    this->K = K_;
    this->params = params_;
    this->verbose = verbose_;
}


// --------------------------------
// class{OC_SVM} -> function{log}
// --------------------------------
void OC_SVM::log(const std::string str){
    if (this->verbose){
        std::cout << str << std::flush;
    }
    return;
}


// ---------------------------------
// class{OC_SVM} -> function{sort}
// ---------------------------------
void OC_SVM::sort(std::vector<std::pair<double, int>> &data){

    size_t i, j;
    std::pair<double, int> tmp;
    
    for (i = 1; i < data.size(); i++){
        j = i;
        while ((j > 0) && (data[j - 1].first > data[j].first)){
            tmp = data[j - 1];
            data[j - 1] = data[j];
            data[j] = tmp;
            j--;
        }
    }

    return;
}


// ----------------------------------
// class{OC_SVM} -> function{train}
// ----------------------------------
void OC_SVM::train(const std::vector<std::vector<double>> x, const size_t D, const double nu, const double lr, const double limit){

    constexpr double eps = 0.0000001;

    size_t i, j;
    size_t N, Ns, Ns_in;
    bool judge;
    double C;
    double item1, item2, item3;
    double delta;
    double beta;
    double error;
    std::vector<double> alpha;

    // (1) Set Lagrange Multiplier and Parameters
    N = x.size();
    C = 1.0 / ((double)N * nu);
    alpha = std::vector<double>(N, 0.0);
    beta = 1.0;

    // (2) Training
    this->log("\n");
    this->log("/////////////////////// Training ///////////////////////\n");
    do {

        judge = false;
        error = 0.0;

        // (2.1) Update Alpha
        for (i = 0; i < N; i++){

            // Set item 1
            item1 = 0.0;
            for (j = 0; j < N; j++){
                item1 += alpha[j] * this->K(x[i], x[j], this->params);
            }

            // Set item 2
            item2 = 0.0;
            for (j = 0; j < N; j++){
                item2 += alpha[j];
            }
            item2 -= 1.0;
            
            // Set Delta
            delta = item1 + beta * item2;

            // Update
            alpha[i] -= lr * delta;
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

        // (2.2) Update Beta
        item3 = 0.0;
        for (i = 0; i < N; i++){
            item3 += alpha[i];
        }
        item3 -= 1.0;
        beta += item3 * item3 / 2.0;

        // (2.3) Output Residual Error
        this->log("\rerror: " + std::to_string(error));

    }while (judge);
    this->log("\n");
    this->log("////////////////////////////////////////////////////////\n");

    // (3.1) Description for support vectors
    Ns = 0;
    Ns_in = 0;
    this->xs = std::vector<std::vector<double>>();
    this->alpha_s = std::vector<double>();
    this->xs_in = std::vector<std::vector<double>>();
    this->alpha_s_in = std::vector<double>();
    for (i = 0; i < N; i++){
        if ((eps < alpha[i]) && (alpha[i] < C - eps)){
            this->xs.push_back(x[i]);
            this->alpha_s.push_back(alpha[i]);
            Ns++;
        }
        else if (alpha[i] >= C - eps){
            this->xs_in.push_back(x[i]);
            this->alpha_s_in.push_back(alpha[i]);
            Ns_in++;
        }
    }
    this->log("Ns (number of support vectors on margin) = " + std::to_string(Ns) + "\n");
    this->log("Ns_in (number of support vectors inside margin) = " + std::to_string(Ns_in) + "\n");

    // (3.2) Description for b
    this->b = 0.0;
    for (i = 0; i < Ns; i++){
        for (j = 0; j < Ns; j++){
            this->b += this->alpha_s[j] * this->K(this->xs[j], this->xs[i], this->params);
        }
        for (j = 0; j < Ns_in; j++){
            this->b += this->alpha_s_in[j] * this->K(this->xs_in[j], this->xs[i], this->params);
        }
    }
    this->b /= (double)Ns;
    this->log("bias = " + std::to_string(this->b) + "\n");
    this->log("////////////////////////////////////////////////////////\n\n");

    return;
}


// ---------------------------------
// class{OC_SVM} -> function{test}
// ---------------------------------
void OC_SVM::test(const std::vector<std::vector<double>> normal_data, const std::vector<std::vector<double>> anomaly_data){

    size_t i;
    std::vector<std::pair<double, int>> score;

    this->correct_n = 0;
    for (i = 0; i < normal_data.size(); i++){
        score.push_back({this->f(normal_data[i]), 1});
        if (this->g(normal_data[i]) == 1){
            this->correct_n++;
        }
    }

    this->correct_a = 0;
    for (i = 0; i < anomaly_data.size(); i++){
        score.push_back({this->f(anomaly_data[i]), -1});
        if (this->g(anomaly_data[i]) == -1){
            this->correct_a++;
        }
    }

    this->accuracy = (double)(this->correct_n + this->correct_a) / (double)(normal_data.size() + anomaly_data.size());
    this->accuracy_n = (double)this->correct_n / (double)normal_data.size();
    this->accuracy_a = (double)this->correct_a / (double)anomaly_data.size();

    this->roc(score);

    return;
}


// --------------------------------
// class{OC_SVM} -> function{roc}
// --------------------------------
void OC_SVM::roc(const std::vector<std::pair<double, int>> score){

    size_t i;
    size_t Np, Nn;
    size_t TP, FP;
    double TP_rate, FP_rate;
    double pre_TP_rate, pre_FP_rate;
    std::vector<std::pair<double, int>> score_sorted;    

    // (1) Sort (Ascending Order)
    score_sorted = score;
    this->sort(score_sorted);

    // (2) Set Parameters
    Np = 0;
    Nn = 0;
    for (i = 0; i < score_sorted.size(); i++){
        if (score_sorted[i].second == -1){
            Np++;
        }
        else{
            Nn++;
        }
    }

    // (3) Set Parameters
    TP = 0;
    FP = 0;
    TP_rate = 0.0;
    FP_rate = 0.0;
    pre_TP_rate = 0.0;
    pre_FP_rate = 0.0;
    this->auroc = 0.0;

    // (4) Anomaly Detection
    for (i = 0; i < score_sorted.size() - 1; i++){

        if (score_sorted[i].second == -1){
            TP++;
        }
        else{
            FP++;
        }

        if (score_sorted[i].first != score_sorted[i + 1].first){
            TP_rate = (double)TP / (double)Np;
            FP_rate = (double)FP / (double)Nn;
            this->auroc += (TP_rate + pre_TP_rate) * (FP_rate - pre_FP_rate) * 0.5;
            pre_TP_rate = TP_rate;
            pre_FP_rate = FP_rate;
        }

    }
    this->auroc += (1.0 + pre_TP_rate) * (1.0 - pre_FP_rate) * 0.5;

    return;
}


// ------------------------------
// class{OC_SVM} -> function{f}
// ------------------------------
double OC_SVM::f(const std::vector<double> x){

    size_t i;
    double ans;

    ans = 0.0;
    for (i = 0; i < this->xs.size(); i++){
        ans += this->alpha_s[i] * this->K(this->xs[i], x, this->params);
    }
    for (i = 0; i < this->xs_in.size(); i++){
        ans += this->alpha_s_in[i] * this->K(this->xs_in[i], x, this->params);
    }
    ans -= this->b;
    
    return ans;
}


// ------------------------------
// class{OC_SVM} -> function{g}
// ------------------------------
double OC_SVM::g(const std::vector<double> x){

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