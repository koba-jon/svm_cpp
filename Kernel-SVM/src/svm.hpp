#ifndef SVM_HPP
#define SVM_HPP

#include <string>
#include <vector>
#include <functional>


// -------------------
// namespace{kernel}
// -------------------
namespace kernel{
    double linear(const std::vector<double> x1, const std::vector<double> x2, const std::vector<double> params);
    double polynomial(const std::vector<double> x1, const std::vector<double> x2, const std::vector<double> params);
    double rbf(const std::vector<double> x1, const std::vector<double> x2, const std::vector<double> params);
}


// -------------------
// class{Kernel_SVM}
// -------------------
class Kernel_SVM{
private:

    // member variable
    bool verbose;
    double b;
    std::vector<std::vector<double>> xs;
    std::vector<int> ys;
    std::vector<double> alpha_s;
    std::vector<std::vector<double>> xs_in;
    std::vector<int> ys_in;
    std::vector<double> alpha_s_in;
    /*****  kernel  *****/
    std::function<double(const std::vector<double>, const std::vector<double>, const std::vector<double>)> K;
    std::vector<double> params;
    /*****  kernel  *****/

    // fuction
    void log(const std::string str);

public:

    // member variable
    double accuracy;
    double accuracy_c1, accuracy_c2;
    size_t correct_c1, correct_c2;

    // constructor
    Kernel_SVM() = delete;
    Kernel_SVM(const std::function<double(const std::vector<double>, const std::vector<double>, const std::vector<double>)> K_=kernel::rbf, const std::vector<double> params_={1.0}, const bool verbose_=true);

    // function
    void train(const std::vector<std::vector<double>> class1_data, const std::vector<std::vector<double>> class2_data, const size_t D, const double C, const double lr, const double limit=0.0001);
    void test(const std::vector<std::vector<double>> class1_data, const std::vector<std::vector<double>> class2_data);
    double f(const std::vector<double> x);
    double g(const std::vector<double> x);

};


#endif