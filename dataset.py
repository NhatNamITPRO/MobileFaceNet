import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from utils import get_subdirectories_paths,get_file_paths

class FaceRecognitionDataset(Dataset):
    def __init__(self, dataset):
        """
        Dataset class for face recognition.
        
        Args:
            dataset (str): Path to the dataset.
        """
        self.dataset = dataset
        self.file_paths = []
        self.labels = []
        self.dict_label = {}
        self.transform = transforms.ToTensor()  
        
        self._prepare_dataset()

    def _prepare_dataset(self):
        """Internal method to prepare file paths and labels."""
        subs = get_subdirectories_paths(self.dataset)
        for index, sub in enumerate(subs):
            self.dict_label[index] = os.path.basename(sub)
            file_paths = get_file_paths(sub)
            self.file_paths.extend(file_paths)
            self.labels.extend([index] * len(file_paths))

    def __len__(self):
        """Return the total number of samples."""
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        Get an image and its label by index.
        
        Args:
            idx (int): Index of the image and label to return.
            
        Returns:
            tuple: (image, label)
        """
        img_path = self.file_paths[idx]
        label = self.labels[idx]
        
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None, None
        img = np.array(img)
        img = (img - 127.5)/128
        img = self.transform(img)
        return img, label, os.path.basename(img_path)