import torch
import numpy as np
import cv2
import os
from torch.utils.data import Dataset

def rpt(x):
    return x.repeat(1, 1, 1)

class Dataset(Dataset):
    def __init__(self, data_dir, num_classes=36, num_chars=4, transform=None, target_transform=None, resize=(90, 25)):
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.num_chars = num_chars
        self.transform = transform
        self.target_transform = target_transform
        self.resize = resize
        self.char_set = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

        csv_file = os.path.join(data_dir, 'label.csv')
        self.images = []
        self.targets = []
        with open(csv_file, 'r') as f:
            cnt = 0
            for line in f:
                try:
                    line = line.strip()
                    if not line:
                        continue
                    img_file, label = line.split(',')
                    img_file = os.path.join(data_dir, img_file + '.png')
                    img = cv2.imread(img_file)

                    # preprocess image
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                    img = np.expand_dims(img, axis=2)

                    # in order to let the model learn the inverse of the image
                    if cnt % 2 == 0:
                        img = cv2.bitwise_not(img)

                    # resize image
                    if self.resize:
                        img = cv2.resize(img, self.resize)
                    
                    # check if lebel char is in char_set
                    for c in label:
                        if c not in self.char_set:
                            raise Exception('Invalid label')
                    
                    # convert label to output vector
                    vec = np.zeros(self.num_classes * self.num_chars)
                    for i, c in enumerate(label):
                        idx = i * self.num_classes + self.char_set.find(c)
                        vec[idx] = 1
                    
                    # save image and label
                    self.images.append(img)
                    self.targets.append(vec)
                    cnt += 1
                except Exception as e:
                    continue

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        target = self.targets[idx]

        if self.transform:
            img = self.transform(img)
        
        if self.target_transform:
            target = self.target_transform(target)

        return img, torch.tensor(target, dtype=torch.float32)

if __name__ == '__main__':
    pass