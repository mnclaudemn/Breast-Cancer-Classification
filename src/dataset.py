import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
# ===================== TRANSFORMS =====================
def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])


# ===================== DATASET =====================
class BrainTumorDataset(Dataset):
    """
    Multiclass classification:
    0: glioma
    1: meningioma
    2: pituitary
    3: no_tumor
    """

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


# ===================== DATALOADER =====================
def create_dataloaders(train_data, val_data, batch_size=32, num_workers=2):

    train_dataset = BrainTumorDataset(
        train_data["images"],
        train_data["labels"],
        transform=get_transforms(train=True)
    )

    val_dataset = BrainTumorDataset(
        val_data["images"],
        val_data["labels"],
        transform=get_transforms(train=False)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader
