from torchvision import datasets, transforms
from torch.utils.data import DataLoader
num_workers=os.cpu_count()

def create_dataloaders (
                            train_dir:str,
                            test_dir:str,
                                transform:transforms.Compose,
                                batch_size:int,
                                num_workers:int=num_workers):
    # Use ImageFolder to create datasets
    train_data=datasets.ImageFolder(train_dir, transform=transform)
    test_data=datasets.ImageFolder(test_dir,transform=transform)
    #Get class names 
    class_names=train_data.classes
    # Turn datasets into dataloaders
    train_dataloader=DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader=DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
return train_dataloader, test_dataloader, class_names
