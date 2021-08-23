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
![SVDD_class](https://user-images.githubusercontent.com/56967584/130436824-d9c4c91d-32e1-4a87-af0f-40d160a87993.png)


## Algorithm
![train](https://user-images.githubusercontent.com/56967584/130436758-fe822b42-a2cb-4b8b-984b-c9d1f7bdda90.png)
![test](https://user-images.githubusercontent.com/56967584/130436762-de392801-9183-4cfe-9bb8-f287079467bf.png)


