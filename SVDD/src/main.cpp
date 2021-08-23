#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <algorithm>
#include <functional>
#include <filesystem>
// 3rd-Party Libraries
#include <boost/program_options.hpp>
// Original
#include "svdd.hpp"

// Define Namespace
namespace fs = std::filesystem;
namespace po = boost::program_options;

// Function Prototype
void Collect_Paths(const std::string root, const std::string sub, std::vector<std::string> &paths);
std::vector<std::string> Get_Paths(const std::string root);
std::vector<std::vector<double>> Get_Data(const std::vector<std::string> paths, const size_t D);
void Set_Kernel(po::variables_map &vm, std::function<double(const std::vector<double>, const std::vector<double>, const std::vector<double>)> &K, std::vector<double> &params);


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
        ("train_dir", po::value<std::string>()->default_value("train"), "training directory : ./datasets/<dataset>/<train_dir>/<all data>")
        ("nu", po::value<double>()->default_value(0.003), "regularization parameter (0.0 < nu <= 1.0)")
        ("lr", po::value<double>()->default_value(0.0001), "learning rate of alpha")

        // (3) Define for Test
        ("test_normal_dir", po::value<std::string>()->default_value("test/normal"), "test normal directory : ./datasets/<dataset>/<test_normal_dir>/<all data>")
        ("test_anomaly_dir", po::value<std::string>()->default_value("test/anomaly"), "test anomaly directory : ./datasets/<dataset>/<test_anomaly_dir>/<all data>")

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
    std::function<double(const std::vector<double>, const std::vector<double>, const std::vector<double>)> K;
    std::vector<double> params;
    Set_Kernel(vm, K, params);

    // (3.1) Get Training Data
    std::string train_dir;
    std::vector<std::string> train_paths;
    std::vector<std::vector<double>> train_data;
    /*****************************************************/
    train_dir = "datasets/" + vm["dataset"].as<std::string>() + "/" + vm["train_dir"].as<std::string>();
    train_paths = Get_Paths(train_dir);
    train_data = Get_Data(train_paths, vm["nd"].as<size_t>());

    // (3.2) Training for SVDD
    SVDD svdd(K, params, vm["verbose"].as<bool>());
    svdd.train(train_data, vm["nd"].as<size_t>(), vm["nu"].as<double>(), vm["lr"].as<double>());

    // (4.1) Get Test Normal Data
    std::string test_normal_dir;
    std::vector<std::string> test_normal_paths;
    std::vector<std::vector<double>> test_normal_data;
    /*****************************************************/
    test_normal_dir = "datasets/" + vm["dataset"].as<std::string>() + "/" + vm["test_normal_dir"].as<std::string>();
    test_normal_paths = Get_Paths(test_normal_dir);
    test_normal_data = Get_Data(test_normal_paths, vm["nd"].as<size_t>());

    // (4.2) Get Test Anomaly Data
    std::string test_anomaly_dir;
    std::vector<std::string> test_anomaly_paths;
    std::vector<std::vector<double>> test_anomaly_data;
    /*****************************************************/
    test_anomaly_dir = "datasets/" + vm["dataset"].as<std::string>() + "/" + vm["test_anomaly_dir"].as<std::string>();
    test_anomaly_paths = Get_Paths(test_anomaly_dir);
    test_anomaly_data = Get_Data(test_anomaly_paths, vm["nd"].as<size_t>());

    // (4.3) Test for SVDD
    svdd.test(test_normal_data, test_anomaly_data);
    std::cout << "//////////////////////////// Test ////////////////////////////" << std::endl;
    std::cout << "accuracy-all: " << svdd.accuracy << " (" << svdd.correct_n + svdd.correct_a << "/" << test_normal_data.size() + test_anomaly_data.size() << ")" << std::endl;
    std::cout << "accuracy-normal: " << svdd.accuracy_n << " (" << svdd.correct_n << "/" << test_normal_data.size() << ")" << std::endl;
    std::cout << "accuracy-anomaly: " << svdd.accuracy_a << " (" << svdd.correct_a << "/" << test_anomaly_data.size() << ")" << std::endl;
    std::cout << "AUROC: " << svdd.auroc << std::endl;
    std::cout << "//////////////////////////////////////////////////////////////" << std::endl;

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
void Set_Kernel(po::variables_map &vm, std::function<double(const std::vector<double>, const std::vector<double>, const std::vector<double>)> &K, std::vector<double> &params){

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
        std::cerr << "Error : Don't match the name of kernel." << std::endl;
        std::exit(-1);
    }

    return;

}