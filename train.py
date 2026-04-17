import torch
import torch.nn as nn
import torch.optim as optim

from model import PlaneClassifier
from dataset import get_dataloaders

from tqdm import tqdm

train_loader,val_loader = get_dataloaders()

device="cpu"

model=PlaneClassifier().to(device)

criterion=nn.CrossEntropyLoss()

optimizer=optim.Adam(
model.parameters(),
lr=0.0001
)

epochs=15

for epoch in range(epochs):

    model.train()

    total_loss=0

    for images,labels in tqdm(train_loader):

        images=images.to(device)

        labels=labels.to(device)

        optimizer.zero_grad()

        outputs=model(images)

        loss=criterion(outputs,labels)

        loss.backward()

        optimizer.step()

        total_loss+=loss.item()

    print("Epoch:",epoch)

    print("Loss:",total_loss)

    # validation

    model.eval()

    correct=0
    total=0

    with torch.no_grad():

        for images,labels in val_loader:

            images=images.to(device)

            labels=labels.to(device)

            outputs=model(images)

            _,pred=torch.max(outputs,1)

            total+=labels.size(0)

            correct+=(pred==labels).sum().item()

    acc=100*correct/total

    print("Validation accuracy:",acc)

torch.save(
model.state_dict(),
"plane_classifier.pth"
)

print("Training finished")