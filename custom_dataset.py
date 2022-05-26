from PIL import Image
import pandas as pd
import os
from torch.utils.data import Dataset
from sklearn import preprocessing
import torch.nn.functional as F

label_list = ["apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare",
            "beet_salad", "beignets", "bibimbap", "bread_pudding", "breakfast_burrito",
            "bruschetta", "caesar_salad", "cannoli", "caprese_salad", "carrot_cake",
            "ceviche", "cheese_plate"]


class CustomDataset(Dataset):
    def __init__(self, image_path: str, text_path: str, transform):
        # from path to list image path and target-label
        # self.final_data = dataset.data
        # self.final_targets = dataset.targets
        image_names = []
        targets = []
        image_paths = []
        for dirpath, dirnames, filenames in os.walk(image_path):
            # since I don't want to delete image 
            label = dirpath.split("/")[-1]
            if label not in label_list:
                continue
            image_names.extend(filenames)
            targets.extend([label] * len(filenames))
            image_paths.extend([dirpath] * len(filenames))

        self.transform = transform
        self.info = pd.read_csv(text_path, names=["index", "description", "label"], index_col=0)
        self.info = self.info.reindex(image_names)

        self.label_encoder = preprocessing.LabelEncoder()
        self.info["label"] = self.label_encoder.fit_transform(targets)

        self.info["image_path"] = image_paths
    
    def __getitem__(self, index):
        target = self.info.iloc[index, 1]
        data_path = self.info.iloc[index, 2]
        data_file = self.info.index[index]
    
        img = Image.open(os.path.join(data_path,data_file))
        
        if self.transform is not None:
            img = self.transform(img)
            
        return img, self.info.iloc[index, 0], target

    def __len__(self):
        return self.info.shape[0]
