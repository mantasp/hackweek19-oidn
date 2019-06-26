#!/bin/bash

for layer in conv1b conv2 conv3 conv4 conv5 conv6 conv7b conv8b conv9b conv10 conv10b
do
    echo "making mosaic $layer"
    montage ../kernels/${layer}_kernel* -geometry +2+2 -border 1 -background "#444444" -bordercolor "#000000" ${layer}_montage.png
done