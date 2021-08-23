# Kernel SVM

This is the implementation of Kernel SVM (Non-linear Support Vector Machine).

- Class: 2
- Problem: Linearly Non-separable
- Decision Boundary: Hyperplane + Kernel

## Usage

### 1. Build
Please build the source file according to the procedure.
~~~
$ mkdir build
$ cd build
$ cmake ..
$ make
$ cd ..
~~~

### 2. Dataset Setting

The following hierarchical relationships are recommended.

![Kernel-SVM_dataset](datasets/dataset.png)

### 3. Execution

The following is an example for Toy Dataset.

#### Setting
Please set the shell for executable file.
~~~
$ vi scripts/toy.sh
~~~
If you want to view specific examples of command line arguments, please view "src/main.cpp" or add "--help" to the argument.
~~~
#!/bin/bash

DATA='toy'

./Kernel-SVM \
    --dataset ${DATA} \
    --nd 2 \
    --C 10.0 \
    --lr 0.0001 \
    --kernel "rbf" \
    --gamma 5.0
~~~

#### Run
Please execute the following to start the program.
~~~
$ sh scripts/toy.sh
~~~


## Formula

![Kernel-SVM_dual](https://user-images.githubusercontent.com/56967584/130267966-98d98cb9-bfb0-4d85-aba6-606ba7b04568.png)
![Kernel-SVM_obj](https://user-images.githubusercontent.com/56967584/130268001-7c4fd249-c8a3-4b8c-bfbc-bdb9a1b9ec78.png)
![Kernel-SVM_delta](https://user-images.githubusercontent.com/56967584/130268014-4a2ac9fa-9491-407c-a925-5eecbb665039.png)
![Kernel-SVM_update](https://user-images.githubusercontent.com/56967584/130268022-7659c8ec-795d-465c-8e6d-b8ec923e7cfa.png)
![Kernel-SVM_class](https://user-images.githubusercontent.com/56967584/130281897-452729b3-c200-4fa7-8f57-20a2a1b6f296.png)


## Algorithm
![train](https://user-images.githubusercontent.com/56967584/130327320-5855fdfc-eb53-4407-a8a3-ad885805786e.png)
![test](https://user-images.githubusercontent.com/56967584/130327324-6c89897b-123f-4b0a-9360-375679932033.png)
