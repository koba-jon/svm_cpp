#ifndef SVM_HPP
#define SVM_HPP

#include <string>
#include <vector>


// ----------------------
// class{HardMargin_SVM}
// ----------------------
class HardMargin_SVM{
private:

    // member variable
    bool verbose;
    std::vector<double> w;
    double b;
    std::vector<std::vector<double>> xs;
    std::vector<int> ys;
    std::vector<double> alpha_s;

    // fuction
    double dot(const std::vector<double> x1, const std::vector<double> x2);
    void log(const std::string str);

public:

    // member variable
    double accuracy;
    double accuracy_c1, accuracy_c2;
    size_t correct_c1, correct_c2;

    // constructor
    HardMargin_SVM() = delete;
    HardMargin_SVM(const bool verbose_=true);

    // function
    void train(const std::vector<std::vector<double>> class1_data, const std::vector<std::vector<double>> class2_data, const size_t D, const double lr, const double limit=0.0001);
    void test(const std::vector<std::vector<double>> class1_data, const std::vector<std::vector<double>> class2_data);
    double f(const std::vector<double> x);
    double g(const std::vector<double> x);

};


#endif