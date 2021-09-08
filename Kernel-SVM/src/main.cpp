#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <algorithm>
#include <filesystem>
// 3rd-Party Libraries
#include <boost/program_options.hpp>
// Original
#include "svm.hpp"

// Define Namespace
namespace fs = std::filesystem;
namespace po = boost::program_options;

// Function Prototype
void Collect_Paths(const std::string root, const std::string sub, std::vector<std::string> &paths);
std::vector<std::string> Get_Paths(const std::string root);
std::vector<std::vector<double>> Get_Data(const std::vector<std::string> paths, const size_t D);
void Set_Kernel(po::variables_map &vm, KernelFunc &K, std::vector<double> &params);


// ----------------------
// 0. Argument Function
// ----------------------
po::options_description parse_arguments(){

    po::options_description args("Options", 200, 30);

    args.add_options()

        // (1) Define for General Parameter
        ("help", "produce help message")
        ("dataset", po::value<std::string>()->default_value("toy"), "dataset name")
        ("nd", po::value<size_t>()->default_value(2), "number of dimensions")
        ("verbose", po::value<bool>()->default_value(true), "verbose")

        // (2) Define for Training
        ("train_class1_dir", po::value<std::string>()->default_value("train/class1"), "training class 1 directory : ./datasets/<dataset>/<train_class1_dir>/<all data>")
        ("train_class2_dir", po::value<std::string>()->default_value("train/class2"), "training class 2 directory : ./datasets/<dataset>/<train_class2_dir>/<all data>")
        ("C", po::value<double>()->default_value(1.0), "regularization parameter (C > 0.0)")
        ("lr", po::value<double>()->default_value(0.0001), "learning rate of alpha")

        // (3) Define for Test
        ("test_class1_dir", po::value<std::string>()->default_value("test/class1"), "test class 1 directory : ./datasets/<dataset>/<test_class1_dir>/<all data>")
        ("test_class2_dir", po::value<std::string>()->default_value("test/class2"), "test class 2 directory : ./datasets/<dataset>/<test_class2_dir>/<all data>")

        // (4) Define for Kernel
        ("kernel", po::value<std::string>()->default_value("rbf"), "kernel : linear / polynomial / rbf")
        ("c", po::value<double>()->default_value(1.0), "addition parameter for polynomial kernel (c > 0.0)")
        ("d", po::value<double>()->default_value(2.0), "exponent parameter for polynomial kernel (natural number)")
        ("gamma", po::value<double>()->default_value(1.0), "precision (inverse of variance) parameter for rbf kernel (gamma > 0.0)")

    ;

    return args;

}


// ------------------
// 1. Main Function
// ------------------
int main(int argc, const char *argv[]){

    // (1) Extract Arguments
    po::options_description args = parse_arguments();
    po::variables_map vm{};
    po::store(po::parse_command_line(argc, argv, args), vm);
    po::notify(vm);
    if (vm.empty() || vm.count("help")){
        std::cout << args << std::endl;
        return 1;
    }

    // (2) Set Kernel
    KernelFunc K;
    std::vector<double> params;
    Set_Kernel(vm, K, params);

    // (3.1) Get Training Data for class 1
    std::string train_class1_dir;
    std::vector<std::string> train_class1_paths;
    std::vector<std::vector<double>> train_class1_data;
    /*****************************************************/
    train_class1_dir = "datasets/" + vm["dataset"].as<std::string>() + "/" + vm["train_class1_dir"].as<std::string>();
    train_class1_paths = Get_Paths(train_class1_dir);
    train_class1_data = Get_Data(train_class1_paths, vm["nd"].as<size_t>());

    // (3.2) Get Training Data for class 2
    std::string train_class2_dir;
    std::vector<std::string> train_class2_paths;
    std::vector<std::vector<double>> train_class2_data;
    /*****************************************************/
    train_class2_dir = "datasets/" + vm["dataset"].as<std::string>() + "/" + vm["train_class2_dir"].as<std::string>();
    train_class2_paths = Get_Paths(train_class2_dir);
    train_class2_data = Get_Data(train_class2_paths, vm["nd"].as<size_t>());

    // (3.3) Training for SVM
    Kernel_SVM svm(K, params, vm["verbose"].as<bool>());
    svm.train(train_class1_data, train_class2_data, vm["nd"].as<size_t>(), vm["C"].as<double>(), vm["lr"].as<double>());

    // (4.1) Get Test Data for class 1
    std::string test_class1_dir;
    std::vector<std::string> test_class1_paths;
    std::vector<std::vector<double>> test_class1_data;
    /*****************************************************/
    test_class1_dir = "datasets/" + vm["dataset"].as<std::string>() + "/" + vm["test_class1_dir"].as<std::string>();
    test_class1_paths = Get_Paths(test_class1_dir);
    test_class1_data = Get_Data(test_class1_paths, vm["nd"].as<size_t>());

    // (4.2) Get Test Data for class 2
    std::string test_class2_dir;
    std::vector<std::string> test_class2_paths;
    std::vector<std::vector<double>> test_class2_data;
    /*****************************************************/
    test_class2_dir = "datasets/" + vm["dataset"].as<std::string>() + "/" + vm["test_class2_dir"].as<std::string>();
    test_class2_paths = Get_Paths(test_class2_dir);
    test_class2_data = Get_Data(test_class2_paths, vm["nd"].as<size_t>());

    // (4.3) Test for SVM
    svm.test(test_class1_data, test_class2_data);
    std::cout << "///////////////////////// Test /////////////////////////" << std::endl;
    std::cout << "accuracy-all: " << svm.accuracy << " (" << svm.correct_c1 + svm.correct_c2 << "/" << test_class1_data.size() + test_class2_data.size() << ")" << std::endl;
    std::cout << "accuracy-class1: " << svm.accuracy_c1 << " (" << svm.correct_c1 << "/" << test_class1_data.size() << ")" << std::endl;
    std::cout << "accuracy-class2: " << svm.accuracy_c2 << " (" << svm.correct_c2 << "/" << test_class2_data.size() << ")" << std::endl;
    std::cout << "////////////////////////////////////////////////////////" << std::endl;

    return 0;
}


// ------------------------------
// 2. Collecting Paths Function
// ------------------------------
void Collect_Paths(const std::string root, const std::string sub, std::vector<std::string> &paths){
    
    fs::path ROOT(root);
    
    for (auto &p : fs::directory_iterator(ROOT)){
        if (!fs::is_directory(p)){
            std::stringstream rpath;
            rpath << p.path().string();
            paths.push_back(rpath.str());
        }
        else{
            std::stringstream subsub;
            subsub << p.path().filename().string();
            Collect_Paths(root + '/' + subsub.str(), sub + subsub.str() + '/', paths);
        }
    }
    
    return;
}


// ---------------------------
// 3. Getting Paths Function
// ---------------------------
std::vector<std::string> Get_Paths(const std::string root){

    std::vector<std::string> paths;

    Collect_Paths(root, "", paths);
    std::sort(paths.begin(), paths.end());

    return paths;

}


// ---------------------------
// 4. Getting Data Function
// ---------------------------
std::vector<std::vector<double>> Get_Data(const std::vector<std::string> paths, const size_t D){

    size_t i;
    double element;
    std::ifstream ifs;
    std::vector<double> data_one;
    std::vector<std::vector<double>> data;

    for (std::string path : paths){
        
        ifs.open(path);

        data_one = std::vector<double>(D);
        for (i = 0; i < D; i++){
            ifs >> element;
            data_one[i] = element;
        }
        data.push_back(data_one);
        
        ifs.close();
    
    }

    return data;
}


// ----------------------------
// 5. Setting Kernel Function
// ----------------------------
void Set_Kernel(po::variables_map &vm, KernelFunc &K, std::vector<double> &params){

    if (vm["kernel"].as<std::string>() == "linear"){
        K = kernel::linear;
    }
    else if (vm["kernel"].as<std::string>() == "polynomial"){
        K = kernel::polynomial;
        params = {vm["c"].as<double>(), vm["d"].as<double>()};
    }
    else if (vm["kernel"].as<std::string>() == "rbf"){
        K = kernel::rbf;
        params = {vm["gamma"].as<double>()};
    }
    else{
        std::cerr << "Error : Couldn't match the name of kernel." << std::endl;
        std::exit(-1);
    }

    return;

}