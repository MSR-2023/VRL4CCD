import torch
import torch.nn as nn

from nets.vgg import VGG16


def get_img_output_length(width, height):
    def get_output_length(input_length):
        # input_length += 6
        filter_sizes = [2, 2, 2, 2, 2]
        padding = [0, 0, 0, 0, 0]
        stride = 2
        for i in range(5):
            input_length = (input_length + 2 * padding[i] - filter_sizes[i]) // stride + 1
        return input_length
    return get_output_length(width) * get_output_length(height) 
    
class Siamese(nn.Module):
    def __init__(self, input_shape, pretrained=False):
        super(Siamese, self).__init__()
        self.vgg = VGG16(pretrained, 3)
        self.vgg.features = self.vgg.main
        del self.vgg.avgpool
        del self.vgg.classfication  # classifier
        
        flat_shape = 512 * get_img_output_length(input_shape[1], input_shape[0])
        self.fully_connect1 = torch.nn.Linear(flat_shape, 512)
        self.fully_connect2 = torch.nn.Linear(512, 1)

    def forward(self, x):
        x1, x2 = x

        #   Pass two inputs to the backbone feature extraction network
        x1 = self.vgg.features(x1)
        x2 = self.vgg.features(x2)

        #   Subtract to take the absolute value, take l1 distance
        x1 = torch.flatten(x1, 1)
        x2 = torch.flatten(x2, 1)
        x = torch.abs(x1 - x2)

        #   two full connections
        x = self.fully_connect1(x)
        x = self.fully_connect2(x)
        return x
