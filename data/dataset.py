import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from PIL import Image


class RFSignalDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform

        # Ensure consistent class ordering
        self.classes = sorted([d for d in os.listdir(
            root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i,
                             cls_name in enumerate(self.classes)}

        self.samples = []
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.samples.append(
                        (os.path.join(cls_dir, fname), self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        image = Image.open(path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, target


def get_transforms(model_name):
    """
    Returns the appropriate transforms based on the model architecture.
    Reflects the input sizes used in the paper.
    """
    # Green-Mamba uses 128x128
    if 'mamba' in model_name.lower() or 'vim' in model_name.lower():
        size = 128
    # ResNet in your legacy code used 180, but 224 is standard.
    # Sticking to 180 to match your previous experiments if preferred,
    # but MobileNet/ViT usually default to 224.
    elif 'resnet' in model_name.lower():
        size = 180
    else:
        # ViT and MobileNet standard size
        size = 224

    train_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return train_transforms, val_transforms


def create_dataloaders(data_dir, model_name, batch_size=32, num_workers=2):
    train_tf, val_tf = get_transforms(model_name)

    # We use the full dataset and split it
    # Transform applied later? No, usually applied in subset
    full_dataset = RFSignalDataset(data_dir, transform=None)

    # Better approach for splitting with transforms:
    # 1. Split indices
    # 2. Wrap in subsets with specific transforms

    dataset_size = len(full_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size

    # Fix seed for reproducibility
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size],
                                    generator=torch.Generator().manual_seed(42))

    # Apply transforms manually since random_split doesn't allow separate transforms easily
    # We override the dataset's transform attribute for the purpose of the loader
    # NOTE: This is a lightweight wrapper to apply transforms
    class TransformedSubset(Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform
            self.classes = subset.dataset.classes  # Pass through classes

        def __getitem__(self, idx):
            x, y = self.subset[idx]
            if self.transform:
                x = self.transform(x)
            return x, y

        def __len__(self):
            return len(self.subset)

    train_subset = TransformedSubset(train_ds, train_tf)
    val_subset = TransformedSubset(val_ds, val_tf)

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_subset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, full_dataset.classes
