import torch, numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# def Train(model, DataLoader, ValDataLoader, criterion, optimizer, epochs, device, scheduler=None):
#     for epoch in range(epochs):
#         model.train()
#         pbar = tqdm(DataLoader, total=len(DataLoader), desc="Training")
#         for input, target in pbar:
#             input = input.to(device)
#             target = target.to(device)
#             optimizer.zero_grad()
#             output = model(input)
#             loss = criterion(output, target)
#             loss.backward()
#             optimizer.step()
#             accuracy = (torch.argmax(target, dim=1) == torch.argmax(output, dim=1)).sum().item() / target.size(0)
#             # tqdm.write(f"Epoch {epoch+1}/{epochs} - Training Loss: {loss.item():.4f} - Accuracy: {accuracy:.4f}")
#             pbar.set_postfix(epoch = epoch + 1, loss=loss.item(), accuracy=accuracy)
#         with torch.no_grad():
#             pbar = tqdm(ValDataLoader, total=len(ValDataLoader), desc="Validation")
#             for input, target in pbar:
#                 input = input.to(device)
#                 target = target.to(device)
#                 output = model(input)
#                 loss = criterion(output, target)
#                 accuracy = (torch.argmax(target, dim=1) == torch.argmax(output, dim=1)).sum().item() / target.size(0)
#                 # tqdm.write(f"Epoch {epoch+1}/{epochs} - Validation Loss: {loss.item():.4f} - Accuracy: {accuracy:.4f}")
#                 pbar.set_postfix(epoch = epoch + 1, loss=loss.item(), accuracy=accuracy)
#         if scheduler is not None:
#             scheduler.step()
#     print('Finished Training')

def Train(model, DataLoader, ValDataLoader, criterion, optimizer, epochs, device, scheduler=None):
    scaler = torch.cuda.amp.GradScaler()  # Mixed precision scaler

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(DataLoader, total=len(DataLoader), desc=f"Training Epoch {epoch+1}/{epochs}")

        for input, target in pbar:
            input, target = input.to(device, non_blocking=True), target.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():  # Use mixed precision
                output = model(input)
                loss = criterion(output, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            accuracy = (torch.argmax(output, dim=1) == torch.argmax(target, dim=1)).sum().detach() / target.size(0)
            pbar.set_postfix(loss=loss.item(), accuracy=accuracy.item())

        if scheduler:
            scheduler.step()

        # Validation Phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for input, target in tqdm(ValDataLoader, desc="Validating"):
                input, target = input.to(device, non_blocking=True), target.to(device, non_blocking=True)
                with torch.cuda.amp.autocast():
                    output = model(input)
                    val_loss += criterion(output, target).detach()

        print(f"Epoch {epoch+1}: Validation Loss = {val_loss / len(ValDataLoader):.4f}")



# def Train_MAE(model, DataLoader, ValDataLoader, optimizer, epochs, device, masking_ratio = 0.0, scheduler=None):
#     for epoch in range(epochs):
#         model.train()
#         pbar = tqdm(DataLoader, total=len(DataLoader), desc="Training")
#         for input in pbar:
#             input = input.to(device)
#             optimizer.zero_grad()
#             loss, pred, mask = model(input, masking_ratio)
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#             optimizer.step()
#             if scheduler is not None:
#                 scheduler.step()
#             pbar.set_postfix(epoch = epoch+1, loss=loss.item())

#         model.eval()
#         with torch.no_grad():
#             pbar = tqdm(ValDataLoader, total=len(ValDataLoader), desc="Validation")
#             for input in pbar:
#                 input = input.to(device)
#                 loss, pred, mask = model(input, masking_ratio)
#                 pbar.set_postfix(epoch = epoch+1, loss=loss.item())
#         if scheduler is not None:
#             scheduler.step()
#     print('Finished Training')


import torch
from tqdm import tqdm

def Train_MAE(model, train_loader, val_loader, criterion, optimizer, epochs, device, scheduler=None):
    scaler = torch.cuda.amp.GradScaler()  # Mixed precision scaler

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, total=len(train_loader), desc=f"Training Epoch {epoch+1}/{epochs}")

        for input, target in pbar:
            input, target = input.to(device, non_blocking=True), target.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():  # Enable mixed precision
                output = model(input)
                loss = criterion(output, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            accuracy = (torch.argmax(output, dim=1) == torch.argmax(target, dim=1)).sum().detach() / target.size(0)
            pbar.set_postfix(loss=loss.item(), accuracy=accuracy.item())

        if scheduler:
            scheduler.step()

        # Validation Phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for input, target in tqdm(val_loader, desc="Validating"):
                input, target = input.to(device, non_blocking=True), target.to(device, non_blocking=True)
                with torch.cuda.amp.autocast():
                    output = model(input)
                    val_loss += criterion(output, target).detach()

        print(f"Epoch {epoch+1}: Validation Loss = {val_loss / len(val_loader):.4f}")






# def Train_SuperResolution(model, DataLoader, ValDataLoader, criterion, optimizer, epochs, device, scheduler=None):
#     for epoch in range(epochs):
#         model.train()
#         pbar = tqdm(DataLoader, total=len(DataLoader), desc="Training")
#         for lr_images, hr_images in pbar:
#             lr_images = lr_images.to(device)
#             hr_images = hr_images.to(device)
#             optimizer.zero_grad()
#             sr_images = model(lr_images)  # Generate super-resolved images
#             loss = criterion(sr_images, hr_images)
#             loss.backward()
#             optimizer.step()
#             # pbar.write(f"Epoch {epoch+1}/{epochs} - Training Loss: {loss.item():.4f}")
#             pbar.set_postfix(epoch = epoch+1, loss=loss.item())

#         model.eval()
#         with torch.no_grad():
#             pbar = tqdm(ValDataLoader, total=len(ValDataLoader), desc="Validation")
#             for lr_images, hr_images in pbar:
#                 lr_images = lr_images.to(device)
#                 hr_images = hr_images.to(device)
#                 sr_images = model(lr_images)
#                 loss = criterion(sr_images, hr_images)
#                 # pbar.write(f"Epoch {epoch+1}/{epochs} - Validation Loss: {loss.item():.4f}")
#                 pbar.set_postfix(epoch = epoch + 1, loss=loss.item())
#         if scheduler is not None:
#             scheduler.step()

#     print('Finished Training')



import torch
from tqdm import tqdm

def Train_SuperResolution(model, train_loader, val_loader, criterion, optimizer, epochs, device, scheduler=None):
    scaler = torch.cuda.amp.GradScaler()  # Mixed precision scaler

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, total=len(train_loader), desc=f"Training Epoch {epoch+1}/{epochs}")

        for input, target in pbar:
            input, target = input.to(device, non_blocking=True), target.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():  # Enable mixed precision
                output = model(input)
                loss = criterion(output, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            psnr = 10 * torch.log10(1 / loss.detach())  # Compute Peak Signal-to-Noise Ratio (PSNR)
            pbar.set_postfix(loss=loss.item(), PSNR=psnr.item())

        if scheduler:
            scheduler.step()

        # Validation Phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for input, target in tqdm(val_loader, desc="Validating"):
                input, target = input.to(device, non_blocking=True), target.to(device, non_blocking=True)
                with torch.cuda.amp.autocast():
                    output = model(input)
                    val_loss += criterion(output, target).detach()

        print(f"Epoch {epoch+1}: Validation Loss = {val_loss / len(val_loader):.4f}")
