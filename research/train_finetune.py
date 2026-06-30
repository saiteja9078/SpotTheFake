import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pillow_heif import register_heif_opener
from glob import glob
from sklearn.model_selection import StratifiedKFold
import numpy as np
import copy

register_heif_opener()

class ScreenDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.paths)
        
    def __getitem__(self, idx):
        img_path = self.paths[idx]
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception:
            img = Image.new('RGB', (256, 256), color='black')
        
        if self.transform:
            img = self.transform(img)
            
        return img, self.labels[idx]

def train_model(model, dataloaders, criterion, optimizer, num_epochs=10):
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in dataloaders[phase]:
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
    model.load_state_dict(best_model_wts)
    return model, best_acc

def main():
    real_paths = glob("Dataset/Real/*.*")
    screen_paths = glob("Dataset/Screen/*.*")
    
    # Filter hidden files
    real_paths = sorted([p for p in real_paths if not os.path.basename(p).startswith('.')])
    screen_paths = sorted([p for p in screen_paths if not os.path.basename(p).startswith('.')])
    
    np.random.seed(42)
    real_indices = np.random.permutation(len(real_paths))
    screen_indices = np.random.permutation(len(screen_paths))
    
    train_real = [real_paths[i] for i in real_indices[:50]]
    test_real = [real_paths[i] for i in real_indices[50:]]
    
    train_screen = [screen_paths[i] for i in screen_indices[:50]]
    test_screen = [screen_paths[i] for i in screen_indices[50:]]
    
    train_paths = np.array(train_real + train_screen)
    train_labels = np.array([0]*len(train_real) + [1]*len(train_screen))
    
    test_paths = np.array(test_real + test_screen)
    test_labels = np.array([0]*len(test_real) + [1]*len(test_screen))
    
    print(f"Train split: {len(train_real)} Real + {len(train_screen)} Screen = {len(train_paths)}")
    print(f"Test split:  {len(test_real)} Real + {len(test_screen)} Screen = {len(test_paths)}")
    
    # No data augmentation
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_ds = ScreenDataset(train_paths, train_labels, transform=train_transform)
    val_ds = ScreenDataset(test_paths, test_labels, transform=val_transform)
    
    dataloaders = {
        'train': DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0),
        'val': DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)
    }
    
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    print("Training model...")
    model, best_acc = train_model(model, dataloaders, criterion, optimizer, num_epochs=15)
    
    print(f"Best Test Accuracy: {best_acc:.4f}")
    # Save the model to the project root
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "best_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Saved to {model_path}")

if __name__ == '__main__':
    main()
