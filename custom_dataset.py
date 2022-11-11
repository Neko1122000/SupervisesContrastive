from PIL import Image
import pandas as pd
import os
from torch.utils.data import Dataset
from sklearn import preprocessing
import torch.nn.functional as F

# label_dict = {"apple_pie":0, "baby_back_ribs":1, "baklava":2, "beef_carpaccio":3, "beef_tartare":4,
#             "beet_salad":5, "beignets":6, "bibimbap":7, "bread_pudding":8, "breakfast_burrito":9,
#             "bruschetta":10, "caesar_salad":11, "cannoli":12, "caprese_salad":13, "carrot_cake":14,
#             "ceviche":15, "cheese_plate":16}

label_dict = {"Chao long": 0, "Bun dau mam tom": 1, "Banh xeo": 2, "Com tam": 3, "Banh cuon": 4,
             "Goi cuon": 5,  "Mi quang": 6, "Bun bo Hue": 7, "Pho": 8, "Banh mi": 9}


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
            if label not in label_dict.keys():
                continue
            image_names.extend(filenames)
            targets.extend([label_dict[label]] * len(filenames))
            image_paths.extend([dirpath] * len(filenames))

        self.transform = transform
        self.info = pd.read_csv(text_path, names=["index", "description", "label"], index_col=0)
        self.info = self.info.reindex(image_names)
        self.info["label"] = targets

        self.info["description"] = self.info["description"].astype('str')

        # self.info["label"] = self.info["label"].apply(lambda x: label_dict[x])
        # print(self.info.label.unique())
        self.info["image_path"] = image_paths
    
    def __getitem__(self, index):
        target = self.info.iloc[index, 1]
        # target = int (target)
        data_path = self.info.iloc[index, 2]
        data_file = self.info.index[index]
    
        img = Image.open(os.path.join(data_path,data_file))
        
        if self.transform is not None:
            img = self.transform(img)
            
        # print(target.unique())
        return self.info.iloc[index, 0], img, target

    def __len__(self):
        return self.info.shape[0]
