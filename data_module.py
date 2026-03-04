import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pytorch_lightning as pl
from sklearn.model_selection import StratifiedKFold, KFold
import albumentations as A
from albumentations.pytorch import ToTensorV2

class GrahamPreprocessing:
    def __init__(self, alpha=4.0, beta=10.0, active=True, gamma=128):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.active = active

    def __call__(self, img_path):
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        if not self.active: return img

        img = self._crop_image_from_gray(img)
        blurred = cv2.GaussianBlur(img, (0, 0), self.beta)
        img = cv2.addWeighted(img, self.alpha, blurred, -self.alpha, self.gamma)
        return img

    def _crop_image_from_gray(self, img, tol=7):
        if img.ndim == 2:
            mask = img > tol
            return img[np.ix_(mask.any(1), mask.any(0))]
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            mask = gray > tol
            if not mask.any(): return img
            return img[np.ix_(mask.any(1), mask.any(0))]

class DRDataset(Dataset):
    def __init__(self, df, path_map, transform=None, preprocessing=None):
        self.df = df
        self.path_map = path_map
        self.transform = transform
        self.preprocessing = preprocessing or GrahamPreprocessing(active=False)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row['id_code'] # Already string cleaned in DataModule
        img_path = self.path_map.get(img_id)
        
        if img_path is None:
             raise FileNotFoundError(f"Image ID {img_id} not found in cache.")

        image = self.preprocessing(img_path)
        if self.transform:
            image = self.transform(image=image)['image']
        
        return image, torch.tensor(row['diagnosis'], dtype=torch.long)

class DRDataModule(pl.LightningDataModule):
    def __init__(self, csv_path, data_dir, batch_size=16, fold_idx=0, 
                 image_size=224, alpha=4.0, beta=10.0, use_graham=True,
                 allow_leaky_split=False):
        super().__init__()
        self.save_hyperparameters()
        self.csv_path = csv_path
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.fold_idx = fold_idx
        self.allow_leaky_split = allow_leaky_split
        self.graham = GrahamPreprocessing(alpha=alpha, beta=beta, active=use_graham)

        self.train_transform = A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, 
                               border_mode=cv2.BORDER_REFLECT_101, p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        
        self.val_transform = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def setup(self, stage=None):
        # 1. Deterministic Path Cache
        priority = {'.png': 0, '.tif': 1, '.tiff': 1, '.jpg': 2, '.jpeg': 2}
        self.path_map = {}
        all_files = sorted(
            os.listdir(self.data_dir), 
            key=lambda x: (priority.get(os.path.splitext(x)[1].lower(), 99), x)
        )

        for f in all_files:
            ext = os.path.splitext(f)[1].lower()
            if ext in priority:
                key = os.path.splitext(f)[0]
                if key not in self.path_map:
                    self.path_map[key] = os.path.join(self.data_dir, f)

        # 2. Strict Parsing & Schema Validation
        # ID Safety: Force string to prevent "123.0" float conversion bugs
        df = pd.read_csv(self.csv_path, dtype={'id_code': str})
        if len(df) == 0: raise ValueError("CSV is empty.")
        
        # Schema Check
        required = {"id_code", "diagnosis"}
        missing_cols = required - set(df.columns)
        if missing_cols:
            raise ValueError(f"CSV missing required columns: {missing_cols}")
        
        # Clean IDs and Enforce Types
        df['id_code'] = df['id_code'].str.strip()
        df["diagnosis"] = df["diagnosis"].astype(int)
        
        # Range Validation (Fail Fast)
        if not df["diagnosis"].between(0, 4).all():
            bad = df.loc[~df["diagnosis"].between(0,4), "diagnosis"].unique()[:5]
            raise ValueError(f"CRITICAL: Diagnosis out of range [0,4]. Found: {bad}")

        # Missing File Check
        csv_ids = set(df['id_code'])
        found_ids = set(self.path_map.keys())
        missing = csv_ids - found_ids
        
        if missing:
            raise ValueError(f"CRITICAL: {len(missing)} IDs in CSV are missing from disk. First 5: {list(missing)[:5]}")

        # 3. Safe Adaptive Splits
        min_class_samples = df['diagnosis'].value_counts().min()
        total_samples = len(df)
        
        n_splits = min(5, total_samples, min_class_samples) if min_class_samples > 0 else 1
        
        if n_splits < 2:
            if not self.allow_leaky_split:
                raise RuntimeError(
                    f"Dataset invalid for split (N={total_samples}, MinClass={min_class_samples}). "
                    "n_splits < 2 implies Train == Val (Data Leakage). "
                    "Set allow_leaky_split=True to override."
                )
            
            print("WARNING: Leaky split allowed (Train == Val). Metrics invalid.")
            n_splits = 1
            splitter = [(list(range(total_samples)), list(range(total_samples)))]
        else:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            splitter = skf.split(df, df['diagnosis'])
            
        target_fold = self.fold_idx % n_splits
        for fold, (train_idx, val_idx) in enumerate(splitter):
            if fold == target_fold:
                self.train_df = df.iloc[train_idx].reset_index(drop=True)
                self.val_df = df.iloc[val_idx].reset_index(drop=True)
                break

    def train_dataloader(self):
        if not hasattr(self, "train_df") or not hasattr(self, "path_map"):
            self.setup()
            
        class_counts = self.train_df['diagnosis'].value_counts().reindex(range(5), fill_value=0)
        
        if (class_counts == 0).any():
            print(f"Warning: Classes {class_counts[class_counts == 0].index.tolist()} have 0 samples in train fold!")

        class_weights = 1.0 / class_counts.replace(0, np.nan)
        sample_weights = self.train_df['diagnosis'].map(class_weights).fillna(0).to_numpy()
        
        if len(sample_weights) == 0 or np.sum(sample_weights) == 0:
             print("Fallback: Sampler weights invalid. Using uniform sampling.")
             return DataLoader(
                DRDataset(self.train_df, self.path_map, self.train_transform, self.graham),
                batch_size=self.batch_size, shuffle=True, 
                num_workers=2, persistent_workers=True, pin_memory=True, drop_last=True
             )

        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.float),
            num_samples=len(sample_weights), replacement=True
        )

        return DataLoader(
            DRDataset(self.train_df, self.path_map, self.train_transform, self.graham),
            batch_size=self.batch_size, sampler=sampler, 
            num_workers=2, persistent_workers=True, pin_memory=True, drop_last=True
        )
    
    def val_dataloader(self):
        if not hasattr(self, "val_df") or not hasattr(self, "path_map"):
            self.setup()
            
        return DataLoader(
            DRDataset(self.val_df, self.path_map, self.val_transform, self.graham),
            batch_size=self.batch_size, shuffle=False, 
            num_workers=2, persistent_workers=True, pin_memory=True
        )