import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize



class Your_Dataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.image_paths = []
        self.labels = []
        self.categories = [] # add your categories here
        self.transform = transform

        data_path = os.path.join(root, " ") # add your path folder

        if train:
            data_path = os.path.join(data_path, "train")
        else:
            data_path = os.path.join(data_path, "test")

        for i, category in enumerate(self.categories):
            data_files = os.path.join(data_path, category)
            for item in os.listdir(data_files):
                path = os.path.join(data_files, item)
                self.image_paths.append(path)
                self.labels.append(i)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[index]
        if self.transform:
            image = self.transform(image)
        return image, label


if __name__ == "__main__":
    transform = Compose([
        Resize((224, 224)), # you can change your image size
        ToTensor(),
    ])
    dataset = Your_Dataset(root=" ", train=True, transform=transform) # add your root path

