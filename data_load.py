import os
import cv2
import glob
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
from torch.utils.data import Dataset, DataLoader


class FacialKeypointsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.key_pts_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.key_pts_frame)

    def __getitem__(self, idx):
        image_name = os.path.join(self.root_dir,
                                self.key_pts_frame.iloc[idx, 0])
        
        image = mpimg.imread(image_name)
        
        # if image has an alpha color channel, get rid of it
        if(image.shape[2] == 4):
            image = image[:,:,0:3]
        
        key_pts = self.key_pts_frame.iloc[idx, 1:].values
        key_pts = key_pts.astype('float').reshape(-1, 2)
        sample = {'image': image, 'keypoints': key_pts}

        if self.transform:
            sample = self.transform(sample)

        return sample
    

    
# tranforms

class Normalize(object):
    """Convert a color image to grayscale and normalize the color range to [0,1]."""        

    def __call__(self, sample, call_type='train'):
        ''' Call type is used to distinguish between `train` and `test`;
            `test` implies on new images without keypoints 
        '''
        if call_type == 'train':
            image, key_pts = sample['image'], sample['keypoints']
        else:
            image = sample
        
        image_copy = np.copy(image)
        if call_type == 'train':
            key_pts_copy = np.copy(key_pts)

        # convert image to grayscale
        image_copy = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # scale color range from [0, 255] to [0, 1]
        image_copy = image_copy/255.0
            
        
        # scale keypoints to be centered around 0 with a range of [-1, 1]
        # mean = 100, sqrt = 50, so, pts should be (pts - 100)/50
        if call_type == 'train':
            key_pts_copy = (key_pts_copy - 100)/50.0

            return {'image': image_copy, 'keypoints': key_pts_copy}
        
        return image_copy


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample, call_type='train'):
        ''' Call type is used to distinguish between `train` and `test`;
            `test` implies on new images without keypoints 
        '''
        
        if call_type == 'train':
            image, key_pts = sample['image'], sample['keypoints']
        else:
            image = sample

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            new_h, new_w = self.output_size, self.output_size
            '''
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
            '''
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # scale the pts, too
        if call_type == 'train':
            key_pts = key_pts * [new_w / w, new_h / h]

            return {'image': img, 'keypoints': key_pts}
        
        return img


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
        crop_size (tuple or int): The amount by which to crop the image
    """

    def __init__(self, output_size, crop_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
            
        # This crops of the portion of the image
        assert isinstance(crop_size, (int, tuple, list))
        if isinstance(crop_size, int):
            self.crop_size = (crop_size, crop_size)
        else:
            assert len(crop_size) == 2
            self.crop_size = crop_size
            

    def __call__(self, sample, call_type='train'):
        ''' Call type is used to distinguish between `train` and `test`;
            `test` implies on new images without keypoints 
        '''
        # Reshape the image to the required size before proceeding
        # This is done to retain a consistant image size
        sample = Rescale (self.output_size) (sample, call_type)        
        
        if call_type == 'train':
            image, key_pts = sample['image'], sample['keypoints']
        else:
            image = sample

        # Choose whether or not to crop
        if np.random.randint (0, 2): # Either 0 or 1
            h, w = image.shape[:2]
            new_h, new_w = self.crop_size

            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)

            replace_img = np.zeros_like (image)

            img = image[top: top + new_h,
                          left: left + new_w]
            replace_img [top: top+new_h, left:left+new_w] = img
            image = replace_img.copy ()

        if call_type == 'train':
            #key_pts = key_pts - [left, top]
    
            return {'image': image, 'keypoints': key_pts}
       
        return image
    

class RandomShear (object):
    ''' Shear the image, so as the help the model generalize
    Args: 
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    '''
    
    def __init__ (self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
            
    def __call__ (self, sample, call_type='train'):
        ''' Call type is used to distinguish between `train` and `test`;
            `test` implies on new images without keypoints 
        '''
        # Resize the image before apply shear
        sample = Rescale (self.output_size) (sample, call_type)
        
        if call_type == 'train':
            image, key_pts = sample ['image'], sample ['keypoints']
        else:
            image = sample
            
        # Get a random integer to decide whether or not to perform shear on the image
        if np.random.randint (0, 2): # Returns either 0 or 1
            # Perform shear
            # Get random shear values for x and y
            shear_x, shear_y = np.random.ranf (2)
            
            # Check if either of it is present
            # if not, nothing to  perform
            if shear_x + shear_y:
                # Scale it down
                shear_x, shear_y = shear_x/2, shear_y/2
                
                # Define the affine matrix
                rand_M = np.array (
                    [[1, shear_x, 0],
                     [shear_y, 1, 0]]
                )
                
                # Shear the image
                image = cv2.warpAffine (image, rand_M, self.output_size)
                
                # Shift the keypoints too
                if call_type == 'train':
                    key_pts = rand_M [:,:-1].dot (key_pts.T).T


        # Return the results
        if call_type == 'train':
            return {'image': image, 'keypoints': key_pts}
        return image
    
    
class RandomScale (object):
    ''' Scale/zoom the image in or out, so that the network trains
        to recognize faces that are at different scales 
        
        It first scales the image by a randomly chosen scaling factor 
        in the range [1, 2) and then the center region with the required
        size is cropped (assuming the face to be generally centered in the image)
        
    '''
    
    def __init__ (self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
            
    def __call__ (self, sample, call_type='train'):
        ''' Call type is used to distinguish between `train` and `test`;
            `test` implies on new images without keypoints 
        '''
            
        if call_type == 'test':
            # Resize the image if its for testing
            image = Rescale (self.output_size) (sample, call_type)
            
            return image
        
        
        # Get the image and the keypoints
        image, key_pts = sample ['image'], sample ['keypoints']
            
        # Get a random integer to decide whether or not to perform scaling on the image
        if np.random.randint (0, 2): # Returns either 0 or 1
            # Get a random scale factor
            # Using the same scaling for both x and y
            # to keep the symmetry
            scale_factor = 1 + np.random.ranf (1)
            
            # Scale the image based on the scaling factor
            image = cv2.resize (
                image, None, 
                fx=scale_factor, fy=scale_factor, 
                interpolation=cv2.INTER_CUBIC
            )

            # Find the center of the image
            # and extract an image from the center 
            # to match the required output size
            # The assumption is that the face is generally in the center of the imag3
            resized_img = image [
                image.shape [0]//2-self.output_size [0]//2: image.shape [0]//2+self.output_size [1]//2,
                image.shape [1]//2-self.output_size [0]//2: image.shape [1]//2+self.output_size [1]//2
            ]

            # Update the keypoints to reflect the same
            resized_kpts = key_pts * scale_factor - [image.shape [1]//2-48, image.shape [0]//2-48]
            
        else:
            sample = Rescale (self.output_size) (sample, call_type)
            resized_img, resized_kpts = sample ['image'], sample ['keypoints']
            
        # Return the updated image and keypoints
        return {'image': resized_img, 'keypoints': resized_kpts}


class RandomRotate (object):
    ''' Rotate the image, so as the help the model generalize
    Args: 
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    '''
    
    def __init__ (self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
            
            
    def __call__ (self, sample, call_type='train'):
        ''' Call type is used to distinguish between `train` and `test`;
            `test` implies on new images without keypoints 
        '''
        # Resize the image before apply shear
        sample = Rescale (self.output_size) (sample, call_type)
            
        if call_type == 'train':
            image, key_pts = sample ['image'], sample ['keypoints']
        else:
            image = sample
            
        # Get a random integer to decide whether or not to perform shear on the image
        if np.random.randint (0, 2): # Returns either 0 or 1
            # Perform rotation
            
            # Get random theta value in degrees
            # Its restricted to 25 degrees (on both sides)
            # to avoid aggressive rotations
            theta = random.uniform (-25, 25)

            # Convert to radians
            theta *= np.pi / 180

            # Define the affine matrix for rotation
            rand_M = np.array ([
                [np.cos (theta), -np.sin (theta), 0],
                [np.sin (theta), np.cos (theta), 0]
            ])

            # Perform the rotation on the image
            image = cv2.warpAffine (image, rand_M, self.output_size)
               
            # Rotate the keypoints too
            if call_type == 'train':
                key_pts = rand_M [:,:-1].dot (key_pts.T).T

        # Return the results
        if call_type == 'train':
            return {'image': image, 'keypoints': key_pts}
        return image


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, call_type='train'):
        ''' Call type is used to distinguish between `train` and `test`;
            `test` implies on new images without keypoints 
        '''
        
        if call_type == 'train':
            image, key_pts = sample['image'], sample['keypoints']
        else:
            image = sample
         
        # if image has no grayscale color channel, add one
        if(len(image.shape) == 2):
            # add that third color dim
            image = image.reshape(image.shape[0], image.shape[1], 1)
            
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        
        if call_type == 'train':
            return {'image': torch.from_numpy(image),
                'keypoints': torch.from_numpy(key_pts)}
        
        return torch.from_numpy (image)
    
    
### END ###    
    
    