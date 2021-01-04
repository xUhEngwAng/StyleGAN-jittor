from jittor.dataset.dataset import Dataset
import matplotlib.pyplot as plt
import os

class SymbolDataset(Dataset):
    def __init__(self, root_path, transform, resolution):
        super().__init__()
        
        resolution_path = os.path.join(root_path, str(resolution))
        train_image = []

        for image_file in os.listdir(resolution_path):
            image_path = os.path.join(resolution_path, image_file)
            if os.path.splitext(image_path)[-1] != '.jpg':
                continue
            image = plt.imread(image_path)
            train_image.append(image)
                
            setattr(self, 'train_image_' + str(resolution), train_image)
            
        self.transform  = transform
        self.resolution = resolution
        
    def __len__(self):
        return len(getattr(self, 'train_image_' + str(self.resolution)))
    
    def __getitem__(self, index):
        X = getattr(self, 'train_image_' + str(self.resolution))[index]
        return self.transform(X)