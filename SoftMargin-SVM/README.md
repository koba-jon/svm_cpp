# Soft Margin SVM

This is the implementation of Soft Margin SVM (Soft Margin Support Vector Machine).

- Class: 2
- Problem: Linearly Non-separable
- Decision Boundary: Hyperplane

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

![SoftMargin-SVM_dataset](datasets/dataset.png)

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

./SoftMargin-SVM \
    --dataset ${DATA} \
    --nd 2 \
    --C 10.0 \
    --lr 0.0001
~~~

#### Run
Please execute the following to start the program.
~~~
$ sh scripts/toy.sh
~~~

## Formula

![SoftMargin-SVM_dual](https://user-images.githubusercontent.com/56967584/130267821-a46ce22f-1acd-4e37-9e14-0269e30b1e00.png)
![SoftMargin-SVM_obj](https://user-images.githubusercontent.com/56967584/130267828-c346f820-7c96-4b7d-afc9-127d7539b0d5.png)
![SoftMargin-SVM_delta](https://user-images.githubusercontent.com/56967584/130267833-48e2d2a8-e54f-407d-92e1-db31f85f13f1.png)
![SoftMargin-SVM_update](https://user-images.githubusercontent.com/56967584/130267837-854ac1fa-3f09-46a8-a030-d406dd96752c.png)
![SoftMargin-SVM_class](https://user-images.githubusercontent.com/56967584/130281866-5e8209af-89c0-4ff1-b686-47256a5461fe.png)


## Algorithm
![train](https://user-images.githubusercontent.com/56967584/130327310-f7a7d992-970e-4026-9ac0-6ac61129ca5c.png)
![test](https://user-images.githubusercontent.com/56967584/130327313-baf9e738-8918-49ea-857a-65f72a23f491.png)

