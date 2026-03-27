import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import Dataset, DataLoader


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


dataset_path= os.path.join(os.path.dirname(__file__),"data")
categories=["with_mask","without_mask"]


class MaskDataset(Dataset):
    def __init__(self,X, y):
        self.X=torch.tensor(X,dtype=torch.float32).permute(0, 3, 1, 2)
        self.y=torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self,idx):
        return self.X[idx],self.y[idx]


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv=nn.Sequential(
            nn.Conv2d(3,32,3),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(64,128,3),nn.ReLU(),nn.MaxPool2d(2)
        )

        self.fc=nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*6*6,128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128,2)
        )

    def forward(self,x):
        return self.fc(self.conv(x))

def load_data():
    data=[]
    labels=[]
    for category in categories:
        path=os.path.join(dataset_path, category)
        label=categories.index(category)
        print(f"Loading {category}......")

        for img in os.listdir(path):
            try:
                imgpath=os.path.join(path, img)
                image=cv2.imread(imgpath)
                image=cv2.resize(image, (64, 64))
                data.append(image)
                labels.append(label)
            except (cv2.error, OSError, AttributeError):
                continue

    data =np.array(data)/255.0
    labels= np.array(labels)

    print("Total images:", len(data))
    return data,labels


def train_model():
    print("Using:", device)
    data,labels= load_data()
    X_train,X_test,y_train,y_test= train_test_split(
        data,labels, test_size=0.2, random_state=42
    )

    train_loader= DataLoader(MaskDataset(X_train, y_train), batch_size=32, shuffle=True)
    test_loader= DataLoader(MaskDataset(X_test, y_test), batch_size=32)
    model=CNN().to(device)
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(),lr=0.001)
    history={"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}

    epochs=10
    for epoch in range(epochs):
        model.train()
        train_correct=0
        train_total=0
        train_loss=0

        for batch_images,batch_labels in train_loader:
            batch_images,batch_labels=batch_images.to(device),batch_labels.to(device)
            optimizer.zero_grad()
            outputs=model(batch_images)
            loss=criterion(outputs,batch_labels)
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()
            _, predicted=torch.max(outputs, 1)
            train_total+=batch_labels.size(0)
            train_correct+=(predicted == batch_labels).sum().item()

        train_acc=train_correct/train_total
        train_loss/=len(train_loader)

        model.eval()
        val_correct=0
        val_total=0
        val_loss=0

        with torch.no_grad():
            for batch_images,batch_labels in test_loader:
                batch_images,batch_labels=batch_images.to(device),batch_labels.to(device)
                outputs=model(batch_images)
                loss=criterion(outputs,batch_labels)
                val_loss+=loss.item()
                _, predicted=torch.max(outputs, 1)
                val_total+=batch_labels.size(0)
                val_correct+=(predicted == batch_labels).sum().item()

        val_acc=val_correct/val_total
        val_loss/=len(test_loader)

        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"Epoch {epoch+1}: "
              f"Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, "
              f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

    model.eval()
    all_preds= []
    all_labels= []

    with torch.no_grad():
        for batch_images, batch_labels in test_loader:
            batch_images=batch_images.to(device)

            outputs=model(batch_images)
            _, predicted=torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.numpy())

    accuracy=np.mean(np.array(all_preds)==np.array(all_labels))
    print(f"\nAccuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=categories))
    cm=confusion_matrix(all_labels, all_preds)
    disp=ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=categories)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.show()

    torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__),"mask_detector.pth"))
    print("Model saved!!")

    fig, (ax1, ax2)=plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history['train_acc'], label='Train')
    ax1.plot(history['val_acc'], label='Validation')
    ax1.set_title('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history['train_loss'], label='Train')
    ax2.plot(history['val_loss'], label='Validation')
    ax2.set_title('Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True,alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train_model()
