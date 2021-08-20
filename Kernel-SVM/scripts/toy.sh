#!/bin/bash

DATA='toy'

./Kernel-SVM \
    --dataset ${DATA} \
    --nd 2 \
    --C 10.0 \
    --lr 0.0001 \
    --kernel "rbf" \
    --gamma 5.0