# SVDD

This is the implementation of SVDD (Support Vector Data Description).

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
![SVDD_dual](https://user-images.githubusercontent.com/56967584/130434531-85b48ae0-0226-44fa-b8ea-4ce1d9c8b8da.png)
![SVDD_obj](https://user-images.githubusercontent.com/56967584/130434532-2aff815d-ae25-4b22-bfee-aae64d9f8e96.png)
![SVDD_delta](https://user-images.githubusercontent.com/56967584/130434551-55238a4a-38a2-4972-a12c-64c24b2de713.png)
![SVDD_update](https://user-images.githubusercontent.com/56967584/130434565-ab6b7a7f-f1ef-4a12-9428-2c63467f5918.png)
![SVDD_class](https://user-images.githubusercontent.com/56967584/130434570-1e489b23-ddc0-4777-abe4-ca46fc24c320.png)


## Algorithm
![train](https://user-images.githubusercontent.com/56967584/130434491-fec5aa39-cab4-40e7-8110-259e6e865f21.png)
![test](https://user-images.githubusercontent.com/56967584/130434497-5db59507-8a71-44ed-9b9d-1f6e1a4d6cbc.png)


