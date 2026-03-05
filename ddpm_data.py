import os
import cv2
import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm


def apply_ben_graham(img, sigmaX=10):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > 7
    if mask.any():
        coords = np.argwhere(mask)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        img = img[y0:y1, x0:x1]
    if img.size == 0:
        return img
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), sigmaX), -4, 128)
    return img


def preprocess_filter_and_save(csv_path, raw_img_dir, output_dir, img_size=128):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    valid_rows = []

    print(f"🔧 Preprocessing {len(df)} images...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_path = os.path.join(raw_img_dir, f"{row['id_code']}.png")
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = apply_ben_graham(img)
        if img.size == 0:
            continue
        img = cv2.resize(img, (img_size, img_size))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if gray.std() > 5.0:
            save_path = os.path.join(output_dir, f"{row['id_code']}.png")
            cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            valid_rows.append(row)

    filtered_df = pd.DataFrame(valid_rows)
    out_csv = os.path.join(output_dir, "filtered_train.csv")
    filtered_df.to_csv(out_csv, index=False)

    print(f"\n✅ Kept {len(filtered_df)}/{len(df)} images")
    for cls in range(5):
        count = (filtered_df["diagnosis"] == cls).sum()
        print(f"   Class {cls}: {count}")
    return out_csv


class DDPMDataset(Dataset):
    def __init__(self, csv_file, processed_img_dir, img_size=128):
        self.data = pd.read_csv(csv_file)
        self.img_dir = processed_img_dir
        self.img_size = img_size
        self.minority_classes = {1, 3, 4}

        self.base_transform = A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.RandomRotate90(p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ColorJitter(
                brightness=0.15, contrast=0.15,
                saturation=0.1, hue=0.03, p=0.5
            ),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2(),
        ])

        self.heavy_transform = A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.RandomRotate90(p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ColorJitter(
                brightness=0.25, contrast=0.25,
                saturation=0.2, hue=0.05, p=0.7
            ),
            A.ElasticTransform(alpha=20, sigma=5, p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.1,
                rotate_limit=180, p=0.5
            ),
            A.GaussNoise(var_limit=(5, 15), p=0.3),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]["id_code"]
        label = self.data.iloc[idx]["diagnosis"]
        img_path = os.path.join(self.img_dir, f"{img_name}.png")

        image = cv2.imread(img_path)
        if image is None:
            return self.__getitem__(np.random.randint(len(self)))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if label in self.minority_classes:
            augmented = self.heavy_transform(image=image)
        else:
            augmented = self.base_transform(image=image)

        return augmented["image"], torch.tensor(label, dtype=torch.long)


class DDPMDataModule(pl.LightningDataModule):
    def __init__(self, raw_csv_path, raw_img_dir, processed_dir,
                 batch_size=32, img_size=128):
        super().__init__()
        self.raw_csv_path = raw_csv_path
        self.raw_img_dir = raw_img_dir
        self.processed_dir = processed_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.filtered_csv = os.path.join(processed_dir, "filtered_train.csv")

    def setup(self, stage=None):
        df_exists = os.path.exists(self.filtered_csv)
        if df_exists:
            existing_df = pd.read_csv(self.filtered_csv)
            if len(existing_df) < 2000:
                df_exists = False
        if not df_exists:
            preprocess_filter_and_save(
                self.raw_csv_path, self.raw_img_dir,
                self.processed_dir, self.img_size
            )
        self.train_dataset = DDPMDataset(
            self.filtered_csv, self.processed_dir, self.img_size
        )
        labels = self.train_dataset.data["diagnosis"].values.astype(int)
        class_counts = np.bincount(labels, minlength=5).astype(float)
        class_weights = 1.0 / (class_counts + 1e-6)
        sample_weights = class_weights[labels]
        self.sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        print("\n📊 Balanced sampling active:")
        for cls in range(5):
            print(f"   Class {cls}: {int(class_counts[cls])} images")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.sampler,
            num_workers=2,
            drop_last=True,
            pin_memory=True,
        )
