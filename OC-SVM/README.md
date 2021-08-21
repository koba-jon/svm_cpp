# OC-SVM

This is the implementation of OC-SVM (One Class Support Vector Machine).

- Class: 1
- Problem: Linearly Non-separable
- Decision Boundary: Non-linear

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

![OC-SVM_dataset](datasets/dataset.png)

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

./OC-SVM \
    --dataset ${DATA} \
    --nd 2 \
    --nu 0.003 \
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

![OC-SVM_dual](https://user-images.githubusercontent.com/56967584/130268146-fd64d0e5-b781-4608-90a1-189ce9ed5173.png)
![OC-SVM_obj](https://user-images.githubusercontent.com/56967584/130268158-5ac93d71-f411-4aaf-9cb2-3a1dc7946fcd.png)
![OC-SVM_delta](https://user-images.githubusercontent.com/56967584/130329353-ee2b7cf0-0bad-4082-944d-44140ff4adf8.png)
![OC-SVM_update](https://user-images.githubusercontent.com/56967584/130329356-d196bad5-16eb-4eab-84dc-690092ca3db6.png)
![OC-SVM_class](https://user-images.githubusercontent.com/56967584/130329448-f5fc295a-8a82-499f-af50-d62dd133ba84.png)


## Algorithm
![train](https://user-images.githubusercontent.com/56967584/130329071-a78d24f5-aea1-4866-865d-6e03bdce15f6.png)
![test](https://user-images.githubusercontent.com/56967584/130328123-beac0ce3-50ed-4ca0-8a44-234c35ce5af9.png)
