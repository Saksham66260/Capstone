from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from PIL import Image

def is_valid_image(path):

    try:

        img = Image.open(path)

        img.verify()

        return True

    except:

        return False


def get_dataloaders():

    train_transform = transforms.Compose([

        transforms.Resize((224,224)),

        transforms.RandomRotation(10),

        transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),

        transforms.Normalize(
            [0.5,0.5,0.5],
            [0.5,0.5,0.5]
        )

    ])

    val_transform = transforms.Compose([

        transforms.Resize((224,224)),

        transforms.ToTensor(),

        transforms.Normalize(
            [0.5,0.5,0.5],
            [0.5,0.5,0.5]
        )

    ])

    train_data = datasets.ImageFolder(

        "plane_classifier/data/train",

        transform=train_transform,

        is_valid_file=is_valid_image
    )

    val_data = datasets.ImageFolder(

        "plane_classifier/data/val",

        transform=val_transform,

        is_valid_file=is_valid_image
    )

    train_loader = DataLoader(
        train_data,
        batch_size=32,
        shuffle=True
    )

    val_loader = DataLoader(
        val_data,
        batch_size=32
    )

    return train_loader,val_loader