from torch.utils.data import Dataset
import glob
import os
from PIL import Image
import numpy as np
from sklearn.utils import shuffle

class DS(Dataset):
    def __init__(self, data_path):
        classes = os.listdir(data_path)
        print(f"Classes present in {classes}")

        labels = {}
        for i, c in enumerate(classes):
            print(f"{i} ----> {c}")
            labels[c] = i

        print(labels)

        self.files = []
        self.labels = []

        for c in classes:
            f = glob.glob(f"{data_path}/{c}/*.jpg")
            self.files.extend(f)
            self.labels.extend([labels[c] for _ in f])

        shuffle(self.files, self.labels, random_state=10)



    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fs = self.files[idx]
        cs = np.array([self.labels[idx]]).astype(np.float32)
        image = np.asarray(Image.open(fs).resize((128, 128))).astype(np.float32) / 255.0
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)

        image = np.transpose(image, (-1, 0, 1))

        return image, cs



if __name__ == "__main__":
    ds = DS("/home/rajat/Desktop/ops/ml/data_generator/code/Cat_Dog/Train")

    for i in range(10):
        img, cls = ds[i]
        img = np.transpose(img, (1, 2, 0))
        img = Image.fromarray(np.uint8(img * 255))
        img.show(cls)
