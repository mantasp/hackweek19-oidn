#!/usr/bin/env python

import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("input", help = "input filename")
parser.add_argument("output", help = "output filename")
args = parser.parse_args()

f = open(args.input, "r")
numTensors = -1
tensors = []
tensor = {}
ndims = 0
dims = []
dataCount = 0
data = []
for line in f:
    if line.startswith('numTensors:'):
        numTensors = int(line.split()[1])
        ndims = 0
    if line.startswith('ndims:'):
        ndims = int(line.split()[1])
        dims = []
    if line.startswith('tensor.dims'):
        dims.append(int(line.split()[1]))
    if line.startswith('name'):
        tensor['name'] = line.split()[1]
    if dataCount > 0:
        data.append(float(line))
        dataCount = dataCount - 1
        if dataCount == 0:
            # build an entry into JSON file
            tensor["type"] = "vector"
            tensor["shape"] = dims
            tensor["data"] = data
            tensors.append(tensor)
            tensor = {}

    if line.startswith('data:'):
        dataCount = 1
        for dim in dims:
            dataCount *= dim
        data = []
        print('dataCount: %f' % dataCount)

results = {}
results['inputs'] = tensors
with open(args.output, 'w') as outfile:  
    json.dump(results, outfile, indent=1)

