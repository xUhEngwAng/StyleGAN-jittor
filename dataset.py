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
            ext = os.path.splitext(image_path)[-1]
            
            if ext not in ['.png', '.jpg']:
                continue
                
            image = plt.imread(image_path)
            
            if ext == '.png':
                image = image * 255
                
            image = image.astype('uint8')
            train_image.append(image)
                
        self.train_image = train_image
        self.transform  = transform
        self.resolution = resolution
        
    def __len__(self):
        return len(self.train_image)
    
    def __getitem__(self, index):
        X = self.train_image[index]
        return self.transform(X)