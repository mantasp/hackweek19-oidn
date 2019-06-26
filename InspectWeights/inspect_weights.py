#!/usr/bin/env python

import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
# import png
from PIL import Image
import matplotlib.pyplot as plt
import visdom

parser = argparse.ArgumentParser()
parser.add_argument("weights", help = "input json weights")
# parser.add_argument("--res", help = "resolution of output image")
parser.add_argument("--input-image", help = "input image")
args = parser.parse_args()

vis = visdom.Visdom()

# read the model weights
inputs = []
with open(args.weights, "r") as json_file:
    results = json.load(json_file)
    inputs = results['inputs']
def getInputByName(name):
    for input in inputs:
        if input['name'] == name:
            return input
    return None

# create the model
class OIDNModel(torch.nn.Module):
    def __init__(self):
        super(OIDNModel, self).__init__()
        # self.layers = []
        # AddConvolution(layers, "conv1", false, true, testSet, res.inputs[0].name);
        self.conv1 = self.create_conv('conv1')
        self.conv1b = self.create_conv('conv1b')
        # self.pool = torch.nn.MaxPool2d(2, stride=2)
        self.conv2 = self.create_conv('conv2')
        self.conv3 = self.create_conv('conv3')
        self.conv4 = self.create_conv('conv4')
        self.conv5 = self.create_conv('conv5')
        # self.upsample = torch.nn.Upsample(scale_factor=2)
        self.conv6 = self.create_conv('conv6')
        self.conv6b = self.create_conv('conv6b')
        self.conv7 = self.create_conv('conv7')
        self.conv7b = self.create_conv('conv7b')
        self.conv8 = self.create_conv('conv8')
        self.conv8b = self.create_conv('conv8b')
        self.conv9 = self.create_conv('conv9')
        self.conv9b = self.create_conv('conv9b')
        self.conv10 = self.create_conv('conv10')
        self.conv10b = self.create_conv('conv10b')
        self.conv11 = self.create_conv('conv11')
    def create_conv(self, name):
        inputWeights = getInputByName(name + "/W")
        inputBiases = getInputByName(name + "/b")
        shape = inputWeights['shape']

        channelCount = shape[0]
        weightTensor = torch.tensor(inputWeights['data'], dtype=torch.float32, requires_grad=False)
        weightTensor.resize_(shape)
        global vis
        # vis.images(weightTensor)

        # weightTensor = weightTensor.transpose(0, 1)
        biasTensor = torch.tensor(inputBiases['data'], dtype=torch.float32, requires_grad=False)
        biasTensor.resize_(channelCount)
        # shape kernel weights are: height, width, input, output
        # Kernels(0), Channels(1), Height(2), Width(3)
        # torch expects: N, C_{in}, H_{in}, W_{in}: 
        conv = torch.nn.Conv2d(in_channels=shape[1], out_channels=shape[0], kernel_size=shape[2], padding=1)
        conv.weight = torch.nn.Parameter(weightTensor)
        conv.bias = torch.nn.Parameter(biasTensor)
        return conv
    def forward(self, x):
        start = x
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv1b(x))
        x = F.max_pool2d(x, 2, stride=2)
        pool1 = x
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, stride=2)
        pool2 = x
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, stride=2)
        pool3 = x
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, stride=2)
        pool4 = x
        x = F.relu(self.conv5(x))
        x = F.max_pool2d(x, 2, stride=2)
        # pool5 = x

        x = F.interpolate(x, mode='nearest', scale_factor=2)
        x = torch.cat((x, pool4), 1)
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv6b(x))

        x = F.interpolate(x, mode='nearest', scale_factor=2)
        x = torch.cat((x, pool3), 1)
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv7b(x))
        
        x = F.interpolate(x, mode='nearest', scale_factor=2)
        x = torch.cat((x, pool2), 1)
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv8b(x))
        
        x = F.interpolate(x, mode='nearest', scale_factor=2)
        x = torch.cat((x, pool1), 1)
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv9b(x))
        
        x = F.interpolate(x, mode='nearest', scale_factor=2)
        x = torch.cat((x, start), 1)
        x = F.relu(self.conv10(x))
        x = F.relu(self.conv10b(x))

        x = F.relu(self.conv11(x))
        return x

class SaveFeatures():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.features = torch.tensor(output, requires_grad=True)
        # self.features = F.relu(self.features)
    def close(self):
        self.hook.remove()

class FilterVisualizer():
    def __init__(self, size=56, upscaling_steps=12, upscaling_factor=1.2):
        self.size, self.upscaling_steps, self.upscaling_factor = size, upscaling_steps, upscaling_factor
        # self.model = vgg16(pre=True).cuda().eval()
        self.model = OIDNModel()
        self.model.eval()
        # set_trainable(self.model, False)
        for p in self.model.parameters(): p.requires_grad=False

    def visualize(self, layer, filter, lr=0.02, opt_steps=20, blur=None):
        sz = self.size
        np_img = np.random.uniform(0.45, 0.55, (1, sz, sz)).astype(np.float32)  # generate random image
        # global vis
        # vis.image(np_img)
        img = torch.tensor(np_img)
        img = img.expand(3, img.size()[1], img.size()[2])
        img = img.unsqueeze(0)
        modelLayer = getattr(self.model, layer)
        if modelLayer is None:
            print("Error: could not find layer %s" % layer)
            return
        # activations = SaveFeatures(list(self.model.children())[layer])  # register hook
        activations = SaveFeatures(modelLayer)
        self.outputs = []

        for sizeStep in range(self.upscaling_steps):  # scale the image up upscaling_steps times
            # train_tfms, val_tfms = tfms_from_model(vgg16, sz)
            # img_var = V(val_tfms(img)[None], requires_grad=True)  # convert image to Variable that requires grad
            img_var = torch.autograd.Variable(img, requires_grad=True)  # convert image to Variable that requires grad
            optimizer = torch.optim.Adam([img_var], lr=lr) # , weight_decay=1e-6
            for n in range(opt_steps):  # optimize pixel values for opt_steps times
                optimizer.zero_grad()
                self.model(img_var)
                loss = -activations.features[0, filter].mean()
                loss.backward()
                optimizer.step()
            # img = val_tfms.denorm(img_var.data.numpy()[0].transpose(1,2,0))
            # img = img_var[0].transpose(1, 2, 0).clamp(0, 1)
            img = img_var
            self.output = img[0].detach()
            self.outputs.append(self.output)
            if sizeStep < self.upscaling_steps - 1:
                sz = int(self.upscaling_factor * sz)  # calculate new image size
                # img = cv2.resize(img, (sz, sz), interpolation = cv2.INTER_CUBIC)  # scale image up
                img = F.interpolate(img, size=(sz, sz), mode='bilinear', align_corners=True)
                if blur is not None:
                    kernel = (1.0 / 5) * torch.tensor([[[[0,1,0], [1,1,1], [0,1,0]]]], dtype=torch.float32)
                    kernel = kernel.expand(3, 1, 3, 3)
                    for blurIndex in range(blur):
                        # img = cv2.blur(img,(blur,blur))  # blur image to reduce high frequency patterns
                        img = F.conv2d(img, kernel, stride=1, padding=1, groups=3)
                img = img.detach()
        self.save(layer, filter)
        activations.close()
        
    def save(self, layer, filter):
        name = "%s_kernel_%s" % (layer, filter)
        opts = {'title' : name, 'caption': name}
        norm = self.output
        min = self.output.min()
        max = self.output.max()
        print ("(min, max): (%g, %g)" % (min, max))
        norm = norm - min
        if max - min > 1:
            norm = self.output / (max - min)
        norm = norm.clamp(0, 1)
        # vis.image(norm, opts)
        np_img = np.array(norm).transpose((1,2,0))
        plt.imsave("layers/" + name + ".png", np_img)

# use FilterVisualizer like this
FV = FilterVisualizer(size=32, upscaling_steps=3, upscaling_factor=2)
# layer = 'conv2'
# filter = 1
layers = [
    {'name' : 'conv1b', 'kernels' : 32},
    {'name' : 'conv2', 'kernels' : 48},
    {'name' : 'conv3', 'kernels' : 64},
    {'name' : 'conv4', 'kernels' : 80},
    {'name' : 'conv5', 'kernels' : 112},
    {'name' : 'conv6', 'kernels' : 160},
    {'name' : 'conv7b', 'kernels' : 112},
    {'name' : 'conv8b', 'kernels' : 96},
    {'name' : 'conv9b', 'kernels' : 64},
    {'name' : 'conv10', 'kernels' : 64},
    {'name' : 'conv10b', 'kernels' : 32},
]
for layer in layers:
    for kernel in range(layer['kernels']):
        print("layer %s, kernel %s" % (layer['name'], kernel))
        FV.visualize(layer['name'], kernel, lr=0.03, opt_steps=20, blur=3)


def main():
    im = Image.open(args.input_image)
    im = im.convert(mode='RGB')
    imageTensor = torchvision.transforms.ToTensor()(im)
    
    # imageTensor *= 64
    # im = np.random.rand(1, 256, 256).astype(np.float32)
    # imageTensor = torch.tensor(im)
    # imageTensor = imageTensor.expand(3, imageTensor.size()[1], imageTensor.size()[2])

    vis.image(imageTensor)

    imageTensor = imageTensor.unsqueeze(0)
    oidnModel = OIDNModel()
    modelOut = oidnModel(imageTensor)
    imagesToShow = modelOut.transpose(0, 1) # shape: 32, 1, w, h

    imageToShow = modelOut.squeeze()
    clamped = imageToShow.clamp(0, 1)
    vis.image(clamped)

# main()
