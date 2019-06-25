#!/usr/bin/env python

import numpy as np
import json

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("input", help = "input filename")
parser.add_argument("output", help = "output filename")
args = parser.parse_args()

with open(args.input, "r") as json_file:
    results = json.load(json_file)
    tensors = results['inputs']
    for tensor in tensors:
        dims = tensor['shape']
        # format from dump is oihw: Kernels(0), Channels(1), Height(2), Width(3)
        # expected format         : Height,   Width,   Channels, Kernels 
        #  reorder data = np.tensor([2,3,1,0])
        if len(dims) == 4:
            order = [2,3,1,0]
            data = tensor['data']
            # dims=[2,2,2,2]
            # a1 = np.array([x for x in range(16)])
            a1 = np.array(data)
            a2 = np.reshape(a1, dims)
            # print ("a2 " + str(a2))
            a3 = np.transpose(a2, order)
            # print ("a3 " + str(a3))
            tensor['shape'] = [dims[o] for o in order]
            tensor['data'] = a3.flatten().tolist()

with open(args.output, 'w') as outfile:  
    json.dump(results, outfile, indent=1)


