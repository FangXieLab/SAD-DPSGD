import os
import pandas as pd
import torch
import torch.utils.data as data
from collections import Counter
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from glob import glob

from holoviews.plotting.bokeh.styles import font_size
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit


base_skin_dir = os.path.join('./data/HAM10000/')
imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                     for x in glob(os.path.join(base_skin_dir, '*', '*.jpg'))}





class SkinDataset(data.Dataset):
    def __init__(self, df, transform):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        X = Image.open(self.df['path'][index])
        y = torch.tensor(int(self.df['cell_type_idx'][index]))

        if self.transform:
            X = self.transform(X)

        return X, y
def plot_class_distribution(df, dataset_name):
    """
    plot the histogram of the class
    :param df: Pandas DataFrame that contains information
    :param dataset_name: the name of the dataset
    """
    class_counts = df["cell_type_idx"].value_counts().sort_index()
    labels = class_counts.index.tolist()
    counts = class_counts.values.tolist()
    #print(labels)
    plt.figure(figsize=(8,6))
    plt.bar(labels, counts, color="skyblue")


    plt.xlabel("class",fontsize=18)
    plt.ylabel("num",fontsize=18)
    #plt.title(f"{dataset_name} class", fontsize=16)
    plt.xticks(labels)
    for i, count in enumerate(counts):
        plt.text(i, count + 10, str(count), ha="center",fontsize=14)
    plt.tick_params(axis='both', labelsize=16)
    plt.show()

def getSkinDataset(batch_size):
    seed = 42
    lesion_type_dict = {
        'nv': 'Melanocytic nevi',
        'mel': 'dermatofibroma',
        'bkl': 'Benign keratosis-like lesions ',
        'bcc': 'Basal cell carcinoma',
        'akiec': 'Actinic keratoses',
        'vasc': 'Vascular lesions',
        'df': 'Dermatofibroma'
    }
    tile_df = pd.read_csv(os.path.join(base_skin_dir, 'HAM10000_metadata.csv'))
    tile_df['path'] = tile_df['image_id'].map(imageid_path_dict.get)
    tile_df['cell_type'] = tile_df['dx'].map(lesion_type_dict.get)
    tile_df['cell_type_idx'] = pd.Categorical(tile_df['cell_type']).codes

    tile_df[['cell_type_idx', 'cell_type']].sort_values('cell_type_idx').drop_duplicates()

    train_df, test_df = train_test_split(tile_df, test_size=0.2,random_state=seed)
    #validation_df, test_df = train_test_split(test_df, test_size=0.5)


    train_df = train_df.reset_index()
    test_df = test_df.reset_index()

    kwargs = {'num_workers': 10, 'pin_memory': True}
    ###### DataSet ######
    composed = transforms.Compose([transforms.RandomHorizontalFlip(),
                                   transforms.RandomVerticalFlip(),
                                   transforms.CenterCrop(256),
                                   transforms.RandomCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    training_set = SkinDataset(train_df, transform=composed)
    training_generator = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True, **kwargs)

    test_set = SkinDataset(test_df, transform=composed)
    test_generator = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, **kwargs)

    dataset = SkinDataset(tile_df, transform=composed)

    return training_generator,test_generator,training_set,test_set,dataset


def getSkinDatasetBalanced(batch_size):
    seed = 42
    lesion_type_dict = {
        'nv': 'Melanocytic nevi',
        'mel': 'Melanoma',
        'bkl': 'Benign keratosis-like lesions ',
        'bcc': 'Basal cell carcinoma',
        'akiec': 'Actinic keratoses',
        'vasc': 'Vascular lesions',
        'df': 'Dermatofibroma'
    }
    base_skin_dir = os.path.join('./data/HAM10000/')
    imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                         for x in glob(os.path.join(base_skin_dir, '*', '*.jpg'))}
    tile_df = pd.read_csv(os.path.join(base_skin_dir, 'HAM10000_metadata.csv'))
    tile_df['path'] = tile_df['image_id'].map(imageid_path_dict.get)
    tile_df['cell_type'] = tile_df['dx'].map(lesion_type_dict.get)
    tile_df['cell_type_idx'] = pd.Categorical(tile_df['cell_type']).codes

    tile_df[['cell_type_idx', 'cell_type']].sort_values('cell_type_idx').drop_duplicates()


    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    for train_index, test_index in sss.split(tile_df, tile_df['cell_type_idx']):
        train_df = tile_df.iloc[train_index].reset_index(drop=True)
        test_df = tile_df.iloc[test_index].reset_index(drop=True)

    kwargs = {'num_workers': 10, 'pin_memory': True}
    ###### DataSet ######
    composed = transforms.Compose([transforms.RandomHorizontalFlip(),
                                   transforms.RandomVerticalFlip(),
                                   transforms.CenterCrop(256),
                                   transforms.RandomCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    training_set = SkinDataset(train_df, transform=composed)
    training_generator = data.DataLoader(training_set, batch_size=batch_size, shuffle=True, **kwargs)

    test_set = SkinDataset(test_df, transform=composed)
    test_generator = data.DataLoader(test_set, batch_size=128, shuffle=True, **kwargs)

    dataset = SkinDataset(tile_df, transform=composed)

    #plot_class_distribution(train_df, "train")
    #plot_class_distribution(test_df, "test")


    return training_generator, test_generator, training_set, test_set, dataset

