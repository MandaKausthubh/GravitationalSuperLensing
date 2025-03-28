import numpy as np

def augment_images(images, num_replications=5):
    """
    Augment images of shape (B, C, H, W) by performing single rolls in each direction.
    
    Args:
        images (numpy.ndarray): Input images, shape (B, C, H, W) where:
                                B is the batch size, C is the number of channels (e.g., 1 for grayscale),
                                H is the height, W is the width.
        num_replications (int): Number of augmented versions to generate per image (default: 5).
                               Must be at least 5 (original + 4 rolls).
    
    Returns:
        numpy.ndarray: Augmented images, shape (B * num_replications, C, H, W).
    """
    # Ensure num_replications is at least 5 (original + 4 rolls)
    if num_replications < 5:
        raise ValueError("num_replications must be at least 5 (original + 4 rolls)")

    # Verify input shape
    if len(images.shape) != 4:
        raise ValueError("Input images must be of shape (B, C, H, W)")

    B, C, H, W = images.shape
    augmented_images = []

    # Process each image in the batch
    for i in range(B):
        img = images[i]  # Shape: (C, H, W)
        augmented_versions = [img]  # Start with the original image

        # 1. Roll rows up by 1 pixel (first row to bottom)
        img_roll_rows_up = np.roll(img, shift=10, axis=1)  # Roll along H axis (axis=1 in (C, H, W))
        augmented_versions.append(img_roll_rows_up)

        # 2. Roll rows down by 1 pixel (last row to top)
        img_roll_rows_down = np.roll(img, shift=-10, axis=1)  # Roll along H axis
        augmented_versions.append(img_roll_rows_down)

        # 3. Roll columns left by 1 pixel (first column to right)
        img_roll_cols_left = np.roll(img, shift=1, axis=2)  # Roll along W axis (axis=2 in (C, H, W))
        augmented_versions.append(img_roll_cols_left)

        # 4. Roll columns right by 1 pixel (last column to left)
        img_roll_cols_right = np.roll(img, shift=-10, axis=2)  # Roll along W axis
        augmented_versions.append(img_roll_cols_right)

        # Stack the augmented versions for this image
        augmented_versions = np.stack(augmented_versions, axis=0)  # Shape: (num_replications, C, H, W)
        augmented_images.append(augmented_versions)

    # Combine all augmented images into a single array
    augmented_images = np.concatenate(augmented_images, axis=0)  # Shape: (B * num_replications, C, H, W)

    return augmented_images

# Example usage with visualization
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Simulate a batch of 2 grayscale images (1 channel, 64x64)
    batch_images = np.random.rand(2, 1, 64, 64)  # Shape: (2, 1, 64, 64)

    # Augment the images
    augmented = augment_images(batch_images, num_replications=5)
    print(f"Original batch shape: {batch_images.shape}")
    print(f"Augmented batch shape: {augmented.shape}")  # Should be (10, 1, 64, 64) = (2 * 5, 1, 64, 64)

    # Visualize the first image and its augmented versions
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    titles = ["Original", "Rows Up", "Rows Down", "Cols Left", "Cols Right"]
    for i in range(5):
        axes[i].imshow(augmented[i, 0], cmap='viridis')  # Display the first channel (grayscale)
        axes[i].set_title(titles[i])
        axes[i].axis('off')
    plt.show()