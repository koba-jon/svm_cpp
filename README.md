# SVM C++ Implementation
These are implementations of Support Vector Machines from scratch in C++.

## 1. Implementation

<table>
  <tr>
    <th>Model</th>
    <th>Class</th>
    <th>Problem</th>
    <th>Decision boundary</th>
    <th>Code</th>
  </tr>
  <tr>
    <td>Hard Margin SVM<br>(Hard Margin Support Vector Machine)</td>
    <td>2</td>
    <td>Linearly Separable</td>
    <td>Linear</td>
    <td><a href="HardMargin-SVM">HardMargin-SVM</a></td>
  </tr>
  <tr>
    <td>Soft Margin SVM<br>(Soft Margin Support Vector Machine)</td>
    <td>2</td>
    <td>Linearly Non-separable</td>
    <td>Linear</td>
    <td><a href="SoftMargin-SVM">SoftMargin-SVM</a></td>
  </tr>
  <tr>
    <td>Kernel SVM<br>(Non-linear Support Vector Machine)</td>
    <td>2</td>
    <td>Linearly Non-separable</td>
    <td>Non-linear</td>
    <td><a href="Kernel-SVM">Kernel-SVM</a></td>
  </tr>
  <tr>
    <td>OC-SVM<br>(One Class Support Vector Machine)</td>
    <td>1</td>
    <td>Linearly Non-separable</td>
    <td>Non-linear</td>
    <td><a href="OC-SVM">OC-SVM</a></td>
  </tr>
</table>

## 2. Requirement

### Boost

This is used for command line arguments, etc. <br>
~~~
$ sudo apt install libboost-dev libboost-all-dev
~~~

## 3. Preparation

### Git Clone
~~~
$ git clone https://github.com/koba-jon/svm_cpp.git
$ cd svm_cpp
~~~

## 4. Execution
- [Hard Margin SVM (Hard Margin Support Vector Machine)](HardMargin-SVM)
- [Soft Margin SVM (Soft Margin Support Vector Machine)](SoftMargin-SVM)
- [Kernel SVM (Non-linear Support Vector Machine)](Kernel-SVM)
- [OC-SVM (One Class Support Vector Machine)](OC-SVM)

## 5. License

This repository: [MIT License](LICENSE)

### 3rd-Party Libraries
- Boost <br>
Official : https://www.boost.org/ <br>
License : https://www.boost.org/users/license.html <br>

## References
- https://qiita.com/ta-ka/items/e6fd0b6fc46dbab4a651 (Japanese)
- https://www.slideshare.net/ssuser186f56/svm-146231602 (Japanese)
