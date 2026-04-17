import torch
import torch.nn as nn
import torchvision.models as models

class PlaneClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.model = models.mobilenet_v3_small(weights="DEFAULT")

        self.model.classifier[3] = nn.Sequential(

            nn.Linear(1024,512),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(512,3)

        )

    def forward(self,x):

        return self.model(x)