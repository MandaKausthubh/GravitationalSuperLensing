import torch, numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def Train(model, DataLoader, ValDataLoader, criterion, optimizer, epochs, device, scheduler=None):
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(DataLoader, total=len(DataLoader), desc="Training")
        for input, target in pbar:
            input = input.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            accuracy = (torch.argmax(target, dim=1) == torch.argmax(output, dim=1)).sum().item() / target.size(0)
            # tqdm.write(f"Epoch {epoch+1}/{epochs} - Training Loss: {loss.item():.4f} - Accuracy: {accuracy:.4f}")
            pbar.set_postfix(epoch = epoch + 1, loss=loss.item(), accuracy=accuracy)
        with torch.no_grad():
            pbar = tqdm(ValDataLoader, total=len(ValDataLoader), desc="Validation")
            for input, target in pbar:
                input = input.to(device)
                target = target.to(device)
                output = model(input)
                loss = criterion(output, target)
                accuracy = (torch.argmax(target, dim=1) == torch.argmax(output, dim=1)).sum().item() / target.size(0)
                # tqdm.write(f"Epoch {epoch+1}/{epochs} - Validation Loss: {loss.item():.4f} - Accuracy: {accuracy:.4f}")
                pbar.set_postfix(epoch = epoch + 1, loss=loss.item(), accuracy=accuracy)
        if scheduler is not None:
            scheduler.step()
    print('Finished Training')


def Train_MAE(model, DataLoader, ValDataLoader, optimizer, epochs, device, masking_ratio = 0.0, scheduler=None):
    
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(DataLoader, total=len(DataLoader), desc="Training")
        for input in pbar:
            input = input.to(device)
            optimizer.zero_grad()
            loss, pred, mask = model(input, masking_ratio)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            training_loss += loss.item()
            if scheduler is not None:
                scheduler.step()
            pbar.set_postfix(epoch = epoch+1, loss=training_loss.item())

        model.eval()
        with torch.no_grad():
            pbar = tqdm(ValDataLoader, total=len(ValDataLoader), desc="Validation")
            for input in pbar:
                input = input.to(device)
                loss, pred, mask = model(input, masking_ratio)
                test_loss += loss.item()
                pbar.set_postfix(epoch = epoch+1, loss=test_loss)
        if scheduler is not None:
            scheduler.step()
    print('Finished Training')


def Train_SuperResolution(model, DataLoader, ValDataLoader, criterion, optimizer, epochs, device, scheduler=None):
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(DataLoader, total=len(DataLoader), desc="Training")
        for lr_images, hr_images in pbar:
            lr_images = lr_images.to(device)
            hr_images = hr_images.to(device)
            optimizer.zero_grad()
            sr_images = model(lr_images)  # Generate super-resolved images
            loss = criterion(sr_images, hr_images)
            loss.backward()
            optimizer.step()
            # pbar.write(f"Epoch {epoch+1}/{epochs} - Training Loss: {loss.item():.4f}")
            pbar.set_postfix(epoch = epoch+1, loss=loss.item())

        model.eval()
        with torch.no_grad():
            pbar = tqdm(ValDataLoader, total=len(ValDataLoader), desc="Validation")
            for lr_images, hr_images in pbar:
                lr_images = lr_images.to(device)
                hr_images = hr_images.to(device)
                sr_images = model(lr_images)
                loss = criterion(sr_images, hr_images)
                # pbar.write(f"Epoch {epoch+1}/{epochs} - Validation Loss: {loss.item():.4f}")
                pbar.set_postfix(epoch = epoch + 1, loss=loss.item())
        if scheduler is not None:
            scheduler.step()

    print('Finished Training')
