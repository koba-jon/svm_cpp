# Hard Margin SVM

This is the implementation of Hard Margin SVM (Hard Margin Support Vector Machine).

- Class: 2
- Problem: Linearly Separable
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

![HardMargin-SVM_dataset](datasets/dataset.png)

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

./HardMargin-SVM \
    --dataset ${DATA} \
    --nd 2 \
    --lr 0.0001
~~~

#### Run
Please execute the following to start the program.
~~~
$ sh scripts/toy.sh
~~~

## Formula

![HardMargin-SVM_dual](https://user-images.githubusercontent.com/56967584/130267566-f6f7e656-2c39-4db2-8ba4-51fc3cf0354a.png)
![HardMargin-SVM_obj](https://user-images.githubusercontent.com/56967584/130267582-afff3278-3204-4d4b-aae1-1c3703822838.png)
![HardMargin-SVM_delta](https://user-images.githubusercontent.com/56967584/130267591-637f03d0-7ee1-4585-8ada-078b918e66e5.png)
![HardMargin-SVM_update](https://user-images.githubusercontent.com/56967584/130267594-8cff7a82-6645-4d4a-a4aa-1b87b966df2d.png)
![HardMargin-SVM_class](https://user-images.githubusercontent.com/56967584/130281675-ffc26b7c-bfd6-445b-8dd1-7cd7858b1843.png)


## Algorithm
![train](https://user-images.githubusercontent.com/56967584/130327291-57894837-dba8-4083-baaf-5854e0cd181c.png)
![test](https://user-images.githubusercontent.com/56967584/130327294-fa563cd0-5378-498f-817d-6dc13dfa38ab.png)


