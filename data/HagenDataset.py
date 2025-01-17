from os import listdir
from os.path import join

from PIL import Image

from natsort import natsorted
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class HagenDataset(Dataset) :
    def __init__(self, opt) :
        # Inheritance
        super(HagenDataset, self).__init__()
        
        # Initialize Variables
        self.opt = opt
        self.noisyDataset = self.getPathList()
    
    def __getitem__(self, index) :
        # Load Data
        noisyImage = Image.open(join(self.noisyDataset[0], self.noisyDataset[1][index]))
        
        # Transform Data
        noisyImage = self.transforms(noisyImage)
        
        return {"noisyImage" : noisyImage, "name" : self.noisyDataset[1][index]}
    
    def __len__(self) :
        return len(self.noisyDataset[1])

    def getPathList(self) :
        # Get Absolute Parent Path of Dataset
        noisyImagePath = join(self.opt.dataRoot, "Hagen")
        
        # Create List Instance for Adding Dataset Path
        noisyImagePathList = listdir(noisyImagePath)
        
        # Sort List
        noisyImagePathList = natsorted(noisyImagePathList)
        
        # Create List Instance for Adding File Name
        noisyImageNameList = [imageName for imageName in noisyImagePathList if ".tiff" in imageName]
        print("checking...........", noisyImageNameList)
        
        return (noisyImagePath, noisyImageNameList)
    
    def transforms(self, noisyImage) :
        # Convert into PyTorch Tensor
        noisyImage = TF.to_tensor(noisyImage)
        noisyImage = noisyImage.to(torch.float32)
        noisyImage = noisyImage / torch.iinfo(torch.int16).max
        # copy channels to make it 3 inputDim
        noisyImage = noisyImage.repeat(3, 1, 1)
        print("checking................", noisyImage.shape, torch.max(noisyImage), torch.min(noisyImage))
        return noisyImage