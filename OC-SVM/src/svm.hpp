#ifndef SVM_HPP
#define SVM_HPP

#include <string>
#include <vector>
#include <utility>
#include <functional>


// -------------------
// namespace{kernel}
// -------------------
namespace kernel{
    double linear(const std::vector<double> x1, const std::vector<double> x2, const std::vector<double> params);
    double polynomial(const std::vector<double> x1, const std::vector<double> x2, const std::vector<double> params);
    double rbf(const std::vector<double> x1, const std::vector<double> x2, const std::vector<double> params);
}
typedef std::function<double(const std::vector<double>, const std::vector<double>, const std::vector<double>)> KernelFunc;


// ---------------
// class{OC_SVM}
// ---------------
class OC_SVM{
private:

    // member variable
    bool verbose;
    double b;
    std::vector<std::vector<double>> xs;
    std::vector<double> alpha_s;
    std::vector<std::vector<double>> xs_in;
    std::vector<double> alpha_s_in;
    /*****  kernel  *****/
    KernelFunc K;
    std::vector<double> params;
    /*****  kernel  *****/

    // fuction
    void log(const std::string str);
    void sort(std::vector<std::pair<double, int>> &data);

public:

    // member variable
    double accuracy;
    double accuracy_n, accuracy_a;
    double auroc;
    size_t correct_n, correct_a;

    // constructor
    OC_SVM() = delete;
    OC_SVM(const KernelFunc K_=kernel::rbf, const std::vector<double> params_={1.0}, const bool verbose_=true);

    // function
    void train(const std::vector<std::vector<double>> x, const size_t D, const double nu, const double lr, const double limit=0.0001);
    void test(const std::vector<std::vector<double>> normal_data, const std::vector<std::vector<double>> anomaly_data);
    void roc(const std::vector<std::pair<double, int>> score);
    double f(const std::vector<double> x);
    double g(const std::vector<double> x);

};


#endif
