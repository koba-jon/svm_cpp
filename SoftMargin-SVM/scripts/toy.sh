#!/bin/bash

DATA='toy'

./SoftMargin-SVM \
    --dataset ${DATA} \
    --nd 2 \
    --C 10.0 \
    --lr 0.0001