import torch, numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


def Train(model, DataLoader, ValDataLoader, criterion, optimizer, epochs, device, scheduler=None):
    training_losses, training_accuracies = [], []
    training_aucScores = []
    test_losses, test_accuracies = [], []
    test_aucScores = []

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    for epoch in range(epochs):

        training_loss, training_accuracy = 0.0, 0.0
        test_loss, test_accuracy = 0.0, 0.0
        test_auc_score, train_auc_score = 0.0, 0.0

        for input, target in tqdm(DataLoader, total=len(DataLoader)):
            input = input.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()
            training_accuracy += (output.argmax(1) == target.argmax(1)).sum().item()
            train_auc_score += roc_auc_score(target.to("cpu").numpy(),
                                output.detach().to("cpu").numpy(),
                                multi_class="ovr")

        with torch.no_grad():
            for input, target in tqdm(ValDataLoader, total=len(ValDataLoader)):
                input = input.to(device)
                target = target.to(device)

                output = model(input)
                loss = criterion(output, target)

                test_loss += loss.item()
                test_auc_score += roc_auc_score(target.to("cpu").numpy(),
                                                output.detach().to("cpu").numpy(),
                                                multi_class="ovr")
                test_accuracy += (output.argmax(1) == target.argmax(1)).sum().item()

        if scheduler is not None:
            scheduler.step()

        if epoch % 10 == 0:
            print(
                f"\nEpoch {epoch+1}/{epochs} -\n" +
                f"\tTraining Loss: {training_loss/len(DataLoader)} \n" + 
                f"\tAccuracy: {training_accuracy/len(DataLoader)}\n"
                f"\tAUC Score: {train_auc_score/len(DataLoader)}\n"
            )
            print(
                f"\tVal Loss: {test_loss/len(ValDataLoader)} \n" + 
                f"\tVal Accuracy: {test_accuracy/len(ValDataLoader)}\n" +
                f"\tVal AUC Score: {test_auc_score/len(ValDataLoader)}\n" 
            )

        training_losses.append(training_loss/len(DataLoader))
        training_accuracies.append(training_accuracy/len(DataLoader))
        training_aucScores.append(train_auc_score/len(DataLoader))

        test_losses.append(test_loss/len(ValDataLoader))
        test_accuracies.append(test_accuracy/len(ValDataLoader))
        test_aucScores.append(test_auc_score/len(ValDataLoader))

    print("Final Performance:")
    print(f"\tTraining Loss: {np.mean(training_losses)}")
    print(f"\tTest Loss: {np.mean(test_losses)}")
    print(f"\tTraining Accuracy : {np.mean(training_accuracies)}")
    print(f"\tTest Accuracy : {np.mean(test_accuracies)}")
    print(f"\tTraining AUC Score : {np.mean(test_auc_score)}")
    print(f"\tTest AUC Score : {np.mean(test_aucScores)}")



    ax[0].plot(training_losses, color="blue")
    ax[0].plot(test_losses, color="green")
    ax[0].set_title("Loss")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Loss")

    ax[1].plot(training_accuracies, color="blue")
    ax[1].plot(test_accuracies, color="green")
    ax[1].set_title("Accuracy")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Loss")

    ax[2].plot(training_aucScores, color="blue")
    ax[2].plot(test_aucScores, color="green")
    ax[2].set_title("AUC Score")
    ax[2].set_xlabel("Epochs")
    ax[2].set_ylabel("AUC Score")

    print('Finished Training')


def Train_MAE(model, DataLoader, ValDataLoader, optimizer, epochs, device, masking_ratio = 0.0, scheduler=None):
    training_losses = []
    test_losses = []

    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    
    for epoch in range(epochs):

        training_loss = 0.0
        test_loss = 0.0

        for input in tqdm(DataLoader, total=len(DataLoader)):
            input = input.to(device)

            optimizer.zero_grad()
            loss, pred, mask = model(input, masking_ratio)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            training_loss += loss.item()
            if scheduler is not None:
                scheduler.step()

        with torch.no_grad():
            for input in tqdm(ValDataLoader, total=len(ValDataLoader)):
                input = input.to(device)

                loss, pred, mask = model(input, masking_ratio)
                test_loss += loss.item()

        if scheduler is not None:
            scheduler.step()

        if epoch % 5 == 0:
            print(
                f"\nEpoch {epoch+1}/{epochs} -\n" +
                f"\tTraining Loss: {training_loss/len(DataLoader)}" 
            )
            print(
                f"\tVal Loss: {test_loss/len(ValDataLoader)} \n" 
            )

        training_losses.append(training_loss/len(DataLoader))
        test_losses.append(test_loss/len(ValDataLoader))


    print("Final Performance:")
    print(f"\tTraining Loss: {np.mean(training_losses)}")
    print(f"\tTest Loss: {np.mean(test_losses)}")

    ax.plot(training_losses, color="blue")
    ax.plot(test_losses, color="green")
    ax.set_title("Loss")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")

    print('Finished Training')

    
    
    
    
    
    
from skimage.metrics import structural_similarity as ssim

def calculate_psnr(img1, img2):
    """
    Calculate PSNR (Peak Signal-to-Noise Ratio) between two images.
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0  # Assuming images are normalized to [0, 1]
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item()

def calculate_ssim(img1, img2):
    """
    Calculate SSIM (Structural Similarity Index) between two images.
    """
    img1 = img1.cpu().detach().numpy().transpose((1, 2, 0))
    img2 = img2.cpu().detach().numpy().transpose((1, 2, 0))
    ssim_score = ssim(img1, img2, multichannel=True, data_range=1.0)  # Assuming images are in [0, 1]
    return ssim_score

def Train_SuperResolution(model, DataLoader, ValDataLoader, criterion, optimizer, epochs, device, scheduler=None):
    training_losses, validation_losses = [], []
    training_psnrs, validation_psnrs = [], []
    training_ssims, validation_ssims = [], []

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    for epoch in range(epochs):
        model.train()
        training_loss = 0.0
        training_psnr = 0.0
        training_ssim = 0.0

        for lr_images, hr_images in tqdm(DataLoader, total=len(DataLoader)):
            lr_images = lr_images.to(device)
            hr_images = hr_images.to(device)

            optimizer.zero_grad()
            sr_images = model(lr_images)  # Generate super-resolved images
            loss = criterion(sr_images, hr_images)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()

            # Calculate metrics
            training_psnr += calculate_psnr(sr_images, hr_images)
            training_ssim += calculate_ssim(sr_images, hr_images)

        training_loss /= len(DataLoader)
        training_psnr /= len(DataLoader)
        training_ssim /= len(DataLoader)
        training_losses.append(training_loss)
        training_psnrs.append(training_psnr)
        training_ssims.append(training_ssim)

        model.eval()
        validation_loss = 0.0
        validation_psnr = 0.0
        validation_ssim = 0.0

        with torch.no_grad():
            for lr_images, hr_images in tqdm(ValDataLoader, total=len(ValDataLoader)):
                lr_images = lr_images.to(device)
                hr_images = hr_images.to(device)

                sr_images = model(lr_images)
                loss = criterion(sr_images, hr_images)
                validation_loss += loss.item()

                validation_psnr += calculate_psnr(sr_images, hr_images)
                validation_ssim += calculate_ssim(sr_images, hr_images)

        validation_loss /= len(ValDataLoader)
        validation_psnr /= len(ValDataLoader)
        validation_ssim /= len(ValDataLoader)
        validation_losses.append(validation_loss)
        validation_psnrs.append(validation_psnr)
        validation_ssims.append(validation_ssim)

        if scheduler is not None:
            scheduler.step()

        if epoch % 10 == 0:

            print(
                f"\nEpoch {epoch+1}/{epochs} -\n" +
                f"\tTraining Loss: {training_loss:.4f}, PSNR: {training_psnr:.4f}, SSIM: {training_ssim:.4f}\n"

                f"\tVal Loss: {validation_loss:.4f}, PSNR: {validation_psnr:.4f}, SSIM: {validation_ssim:.4f}\n"
            )


    print("Final Performance:")
    print(f"\tTraining Loss: {np.mean(training_losses)}")
    print(f"\tTest Loss: {np.mean(validation_losses)}")
    print(f"\tTraining PSNR : {np.mean(training_psnrs)}")
    print(f"\tTest PSNR : {np.mean(validation_psnrs)}")
    print(f"\tTraining SSIM : {np.mean(training_ssims)}")
    print(f"\tTest SSIM : {np.mean(validation_ssims)}")


    ax[0].plot(training_losses, label="Training Loss", color="blue")
    ax[0].plot(validation_losses, label="Validation Loss", color="green")
    ax[0].set_title("Loss")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Loss")
    ax[0].legend()

    ax[1].plot(training_psnrs, label="Training PSNR", color="blue")
    ax[1].plot(validation_psnrs, label="Validation PSNR", color="green")
    ax[1].set_title("PSNR (dB)")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("PSNR (dB)")
    ax[1].legend()

    ax[2].plot(training_ssims, label="Training SSIM", color="blue")
    ax[2].plot(validation_ssims, label="Validation SSIM", color="green")
    ax[2].set_title("SSIM")
    ax[2].set_xlabel("Epochs")
    ax[2].set_ylabel("SSIM")
    ax[2].legend()

    plt.tight_layout()
    plt.show()

    print('Finished Training')
