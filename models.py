## Define the convolutional neural network architecture
# This file contains multiple architecture, which were used while 
# building up to the final architecture
# The one on top is the final architecture used

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


def init_weights_biases (layer):
    ''' Initializes the weights and biases of each layers

    Using He (Kaimer) Initialization instead of Xavier as it works better
    while using ReLU (https://towardsdatascience.com/weight-initialization-in-neural-networks-a-journey-from-the-basics-to-kaiming-954fb9b47c79)
    
    Parameter: `net` - The architecture defined
   
    '''
    
    if isinstance (layer, (nn.Conv2d, nn.Linear)):
        # Using He Initialization
        nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
        
        # The bias is initialized to 0
        layer.bias.data.fill_ (0.0)
        

class NaimishNet (nn.Module):
    ''' This is based on the paper https://arxiv.org/pdf/1710.00977.pdf
        which is based on LeNet's Architecture
    '''
    
    def __init__ (self, drop_prob=None, use_batch_norm=False):
        ''' Define the layers that make up the architecture of the network 
        
        Parameters:
            `drop_prob` - Probability of the nodes being dropped
 
        '''
        
        # Initialize the parent class
        super(NaimishNet, self).__init__()
        
        # Check the call to see if bn is to be used
        self.use_batch_norm = use_batch_norm

        # The architecture is inspired by LeNet, which uses a
        # sequence of conv layer followed by pooling layers
        # the same is done here
        
        # Define the layers to use
        # The input image size is 96x96 grayscale image as in the paper
        
        # Block 1
        self.conv1 = nn.Conv2d (1, 32, 4)
        self.bn1 = nn.BatchNorm2d (32)
        self.mp1 = nn.MaxPool2d (2, 2)
        self.drop1 = nn.Dropout2d (0.1)
        
        # Block 2
        self.conv2 = nn.Conv2d (32, 64, 3)
        self.bn2 = nn.BatchNorm2d (64)
        self.mp2 = nn.MaxPool2d (2, 2)
        self.drop2 = nn.Dropout2d (0.2)
        
        # Block 3
        self.conv3 = nn.Conv2d (64, 128, 2)
        self.bn3 = nn.BatchNorm2d (128)
        self.mp3 = nn.MaxPool2d (2, 2)
        self.drop3 = nn.Dropout2d (0.3)
        
        # Block 4
        self.conv4 = nn.Conv2d (128, 256, 1)
        self.bn4 = nn.BatchNorm2d (256)
        self.mp4 = nn.MaxPool2d (2, 2)
        self.drop4 = nn.Dropout2d (0.4)
        
        # FC layers
        self.fc1 = nn.Linear (6400, 1000)
        self.fc_drop1 = nn.Dropout (0.4)
        
        self.fc2 = nn.Linear (1000, 1000)
        self.fc_drop2 = nn.Dropout (0.5)
        
        self.fc3 = nn.Linear (1000, 1000)
        self.fc_drop3 = nn.Dropout (0.6)
        
        # Final layer
        self.final = nn.Linear (1000, 68*2)
        
            
    def forward(self, x):
        ''' Defines the forward pass 
        
        Parameters:
            `x` - Input image
        '''
        
        # Block 1
        x = F.relu (self.bn1 (self.conv1 (x)) if self.use_batch_norm else self.conv1 (x))
        x = self.mp1 (x)
        x = self.drop1 (x)
        
        # Block 2
        x = F.relu (self.bn2 (self.conv2 (x)) if self.use_batch_norm else self.conv2 (x))
        x = self.mp2 (x)
        x = self.drop2 (x)
        
        # Block 3
        x = F.relu (self.bn3 (self.conv3 (x)) if self.use_batch_norm else self.conv3 (x))
        x = self.mp3 (x)
        x = self.drop3 (x)
        
        # Block 4
        x = F.relu (self.bn4 (self.conv4 (x)) if self.use_batch_norm else self.conv4 (x))
        x = self.mp4 (x)
        x = self.drop4 (x)
        
        # Flatten before passing to FC layers
        x = x.view (x.size (0), -1)
        
        # FC Layers
        x = F.relu (self.fc1 (x))
        x = self.fc_drop1 (x)

        x = F.relu (self.fc2 (x))
        x = self.fc_drop2 (x)
        
        # Final prediction layer
        x = self.final (x)
        
        return x
        

        
class SimpleNet (nn.Module):
    ''' This is a simple conv net, with a few conv layers followed by pooling
    finally leading to the fully connected layer
    '''
    
    def __init__ (self, drop_prob=0.2):
        ''' Define the layers that make up the architecture of the network 
        
        Parameters:
            `drop_prob` - Probability of the nodes being dropped
 
        '''
        
        # Initialize the parent class
        super(SimpleNet, self).__init__()
       
        # Set the dropout probability
        self.drop_prob = drop_prob
        
        # Define the layers to use
        # Block 1
        self.conv1_1 = nn.Conv2d (1, 16, 5)
        self.conv1_2 = nn.Conv2d (16, 32, 5)
        self.mp1 = nn.MaxPool2d (2, 2)
        
        # Block 2
        self.conv2_1 = nn.Conv2d (32, 64, 5)
        self.conv2_2 = nn.Conv2d (64, 64, 5)
        self.mp2 = nn.MaxPool2d (2, 2)
        
        # Block 3
        self.conv3_1 = nn.Conv2d (64, 128, 5)
        self.conv3_2 = nn.Conv2d (128, 256, 5)
        
        # FC layers
        self.avg = nn.AvgPool2d (42)
        self.fc = nn.Linear (256, 68*2)
        
            
    def forward(self, x):
        ''' Defines the forward pass 
        
        Parameters:
            `x` - Input image
        '''
        
        # Block 1
        x = F.relu (self.conv1_1 (x))
        x = F.relu (self.conv1_2 (x))
        x = self.mp1 (x)
        
        # Block 2
        x = F.relu (self.conv2_1 (x))
        x = F.relu (self.conv2_2 (x))
        x = self.mp2 (x)
        
        # Block 3
        x = F.relu (self.conv3_1 (x))
        x = F.relu (self.conv3_2 (x))
        
        x = self.avg (x)
        
        x = x.view (x.size (0), -1)
        x = self.fc (x)
        
        return x
        
        

class ResNet(nn.Module):
    ''' Not being used, was just trying to implement '''
    ''' Inspired by ResNet '''

    def __init__(self, drop_prob=0.2):
        ''' Define the layers that make up the architecture of the network 
        
        Parameters:
            `drop_prob` - Probability of the nodes being dropped
 
        '''
        
        # Initialize the parent class
        super(ResNet, self).__init__()
        
        # Using 3x3 kernels everywhere as it uses less parameters (faster to train)
        # and stacking them together would give the same results as a higher sized kernels
        
        # The architecture is similar to ResNet (cutdown version) which uses multiple 
        # convolution layers followed by maxpooling and these blocks are repeated
        # The ideas is to capture some of the low-level features such as edges and gradients
        # in the first block, more complex pattern in the next block and so on
        
        # After each convolution (without padding) the resulting size 
        # of the images/feature map reduces by 2 (along x and y) 
        # (if using a 3x3 kernel with a stride of 1)
        
        # For facial recognision, the only a size of ~56x56 should be sufficient
        # to get accurate results (don't have the source to cite this, but had read somewhere)
        # So, drop the resolution of the image down quickly during 
        # the initial convolution using  higher kernel size
        
        # Taken from 
        # https://www.researchgate.net/profile/Paolo_Napoletano/publication/322476121/figure/tbl1/AS:668726449946625@1536448218498/ResNet-18-Architecture.png
        
        # Removes MaxPooling as its not used in the original architecture (except for the first layer)
        # https://www.quora.com/Why-is-there-no-max-pooling-in-residual-neural-networks
        # Using 1x1 conv instead
        
        self.init_conv = nn.Conv2d(1, 64, 7, stride=2, padding=3)
        self.init_bn = nn.BatchNorm2d (64)
        self.init_mp = nn.MaxPool2d (3, 2, padding=1)
        
        # Block 1
        self.conv1_1 = nn.Conv2d (64, 64, 3, padding=1, stride=1)
        self.bn1_1 = nn.BatchNorm2d (64)
        self.conv1_2 = nn.Conv2d (64, 64, 3, padding=1)
        self.bn1_2 = nn.BatchNorm2d (64)
        
        ### Skip connection
        ### passing through 1x1 conv
        #self.one1 = nn.Conv2d (64, 64, 1, stride=2)
        
        # Block 2
        self.conv2_1 = nn.Conv2d (64, 128, 3, padding=1, stride=2)
        self.bn2_1 = nn.BatchNorm2d (128)
        self.conv2_2 = nn.Conv2d (128, 128, 3, padding=1)
        self.bn2_2 = nn.BatchNorm2d (128)
        
        ### Skip connection
        ### passing through 1x1 conv
        self.one2 = nn.Conv2d (128, 128, 1, stride=2)
        
        # Block 3
        self.conv3_1 = nn.Conv2d (128, 256, 3, padding=1, stride=2)
        self.bn3_1 = nn.BatchNorm2d (256)
        self.conv3_2 = nn.Conv2d (256, 256, 3, padding=1)
        self.bn3_2 = nn.BatchNorm2d (256)
        
        ### Skip connection
        ### passing through 1x1 conv
        self.one3 = nn.Conv2d (256, 256, 1, stride=2)
        
        # Block 4
        self.conv4_1 = nn.Conv2d (256, 512, 3, padding=1, stride=2)
        self.bn4_1 = nn.BatchNorm2d (512)
        self.conv4_2 = nn.Conv2d (512, 512, 3, padding=1)
        self.bn4_2 = nn.BatchNorm2d (512)
        
        ### Skip connection
        ### passing through 1x1 conv
        self.one4 = nn.Conv2d (512, 512, 1, stride=2)

        # Final fully connected layer(s)        
        ### Pass it through average pooling might be better 
        ### rather than using FC layer(s)???
        ### https://stackoverflow.com/questions/58689997/why-does-the-global-average-pooling-work-in-resnet

        # Using Global Average Pooling instead
        self.avg = nn.AdaptiveAvgPool2d ((1, 1))
        self.fc = nn.Linear (512, 68*2)

        
    def forward(self, x):
        ''' Defines the forward pass 
        
        Parameters:
            `x` - Input image
        '''
    
        # Using ReLU as the non-linear activation after each layer
        x = F.relu (self.init_bn (self.init_conv (x)))
        x = self.init_mp (x)
        print (x.shape)
        
        # Block 1
        x_res = F.relu (self.bn1_1 (self.conv1_1 (x)))
        x_res = self.bn1_2 (self.conv1_2 (x_res))
        x = F.relu (x_res + x)
        print (x.shape)
        
        # Block 2
        x_res = F.relu (self.bn2_1 (self.conv2_1 (x)))
        x_res = self.bn2_2 (self.conv2_2 (x_res))
        x = F.relu (x_res + self.one2 (x))
        
        # Block 3
        x_res = F.relu (self.bn3_1 (self.conv3_1 (x)))
        x_res = self.bn3_2 (self.conv3_2 (x_res))
        x = F.relu (x_res + self.one3 (x))
        
        # Block 4
        x_res = F.relu (self.bn4_1 (self.conv4_1 (x)))
        x_res = self.bn4_2 (self.conv4_2 (x_res))
        x = F.relu (x_res + self.one4 (x))
        
        # Final layers
        x = self.avg (x)
        x = x.view (x.size (0), -1)
        x = self.fc (x)
        
        return x
    
### END ###