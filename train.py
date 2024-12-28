import yaml
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from tqdm import tqdm
import os

from utils import save_checkpoint, load_checkpoint
from model import MobileFacenet
from dataset import FaceRecognitionDataset

# Đọc các hằng số từ file config.yml
with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

DATASET_ROOT = config['PROCESSED_ROOT']
BATCH_SIZE = config['BATCH_SIZE']
EPOCHS = config['EPOCHS']
CHECK_POINT = config['CHECK_POINT']
BEST_MODEL = config['BEST_MODEL']
SAVE_FREQ = config['SAVE_FREQ']
TEST_FREQ = config['TEST_FREQ']

def initialize_dataloaders():
    dataset = FaceRecognitionDataset(DATASET_ROOT)
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)
    return train_dataloader, valid_dataloader

def initialize_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = MobileFacenet().to(device)
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
    return net, device

def train_per_epoch(net, trainloader, optimizer, criterion, device):
    net.train()  
    train_total_loss = 0.0
    total = 0

    tqdm_bar = tqdm(trainloader, total=len(trainloader))
    for data in tqdm_bar:
        img, label = data[0].to(device).float(), data[1].to(device)
        batch_size = img.shape[0]

        optimizer.zero_grad()
        logits = net(img)
        total_loss = criterion(logits, label)
        total_loss.backward()
        optimizer.step()

        train_total_loss += total_loss.item() * batch_size
        total += batch_size

        avg_loss = train_total_loss / total
        tqdm_bar.set_description(desc=f"Training Loss: {avg_loss:.5f}")

    avg_train_loss = train_total_loss / total
    return avg_train_loss

def test_epoch(net, testloader, criterion, device):
    net.eval()  
    test_total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  
        tqdm_bar = tqdm(testloader, total=len(testloader))
        for data in tqdm_bar:
            img, label = data[0].to(device).float(), data[1].to(device)
            batch_size = img.size(0)

            logits = net(img)
            total_loss = criterion(logits, label)
            test_total_loss += total_loss.item() * batch_size

            _, predicted = torch.max(logits, 1)
            correct += (predicted == label).sum().item()
            total += batch_size

            tqdm_bar.set_description(desc=f"Test Loss: {total_loss:.5f}")

    test_total_loss /= total
    accuracy = correct / total
    return test_total_loss, accuracy

def save_best_model(net, epoch, accuracy, best_acc, best_epoch, save_path='best_model.pth.tar'):
    if accuracy > best_acc:
        best_acc = accuracy
        best_epoch = epoch
        print(f"New best model found at epoch {epoch} with accuracy {accuracy:.4f}")
        net_state_dict = net.module.state_dict() if torch.cuda.device_count() > 1 else net.state_dict()
        save_checkpoint({
            'epoch': epoch,
            'net_state_dict': net_state_dict,
            'accuracy': best_acc
        }, filename=save_path)
    return best_acc, best_epoch

def train():
    train_dataloader, valid_dataloader = initialize_dataloaders()
    net, device = initialize_model()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = AdamW(net.parameters(), lr=1e-4)

    best_acc = 0.0
    best_epoch = 0

    if BEST_MODEL:
        best_ckpt = load_checkpoint(BEST_MODEL, device=device)
        if best_ckpt:
            net.load_state_dict(best_ckpt['net_state_dict'])
            best_acc = best_ckpt['accuracy']
            best_epoch = best_ckpt['epoch']
            print(f"Best model loaded with accuracy {best_acc:.4f} from epoch {best_epoch}")

    start_epoch = 0
    if CHECK_POINT:
        last_ckpt = load_checkpoint(CHECK_POINT, device=device)
        if last_ckpt:
            net.load_state_dict(last_ckpt['net_state_dict'])
            start_epoch = last_ckpt['epoch'] + 1
            print(f"Last model loaded from epoch {last_ckpt['epoch']}")

    for epoch in range(start_epoch, EPOCHS + 1):
        print(f'Train Epoch: {epoch}/{EPOCHS} ...')
        train_total_loss = train_per_epoch(net, train_dataloader, optimizer, criterion, device)
        test_total_loss, accuracy = test_epoch(net, valid_dataloader, criterion, device)
        print(f"Epoch [{epoch}/{EPOCHS}], Test Loss: {test_total_loss:.4f}, Accuracy: {accuracy:.4f}")

        if epoch % SAVE_FREQ == 0:
            print(f'Saving checkpoint: {epoch}')
            net_state_dict = net.module.state_dict() if torch.cuda.device_count() > 1 else net.state_dict()
            save_checkpoint({
                'epoch': epoch,
                'net_state_dict': net_state_dict,
                'accuracy': accuracy
            }, filename=os.path.join(f'{epoch:03d}.ckpt'))

        best_acc, best_epoch = save_best_model(net, epoch, accuracy, best_acc, best_epoch)

    print('Finishing training')
    print(f"Best model found at epoch {best_epoch} with accuracy {best_acc:.4f}")
