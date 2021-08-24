# SVDD

This is the implementation of SVDD (Support Vector Data Description).

- Class: 1
- Problem: Linearly Non-separable
- Decision Boundary: Hypersphere + Kernel

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

![SVDD_dataset](datasets/dataset.png)

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

./SVDD \
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
![SVDD_dual](https://user-images.githubusercontent.com/56967584/130436790-62bf1b4a-10d1-4419-b578-341e28392374.png)
![SVDD_obj](https://user-images.githubusercontent.com/56967584/130436797-a82d8950-2ecd-48a5-8cde-deb5a63eaaf5.png)
![SVDD_delta](https://user-images.githubusercontent.com/56967584/130436807-99953c32-8acc-498d-9182-08706ee5819b.png)
![SVDD_update](https://user-images.githubusercontent.com/56967584/130436813-3cabea09-0d5f-4065-ab0f-2f8d99964b86.png)
![SVDD_class](https://user-images.githubusercontent.com/56967584/130538309-9d5e691a-2223-4ef3-b2e8-f228de053e99.png)



## Algorithm
![train](https://user-images.githubusercontent.com/56967584/130538323-a9219a72-5895-4973-80f1-ed207a254085.png)
![test](https://user-images.githubusercontent.com/56967584/130538333-77f5cfd4-01b1-4cf2-9535-80c5d7be8796.png)



