import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def get_subdirectories_paths(parent_path):
    """
    Get a list of paths to subdirectories in a specified directory.

    Args:
        parent_path (str): Path to the parent directory.

    Returns:
        list: List of paths to subdirectories.
    """
    # List comprehension to find all subdirectories
    return [os.path.join(parent_path, d) for d in os.listdir(parent_path) if os.path.isdir(os.path.join(parent_path, d))]

def get_file_paths(parent_path):
    """
    Get a list of paths to all files in a specified directory.

    Args:
        parent_path (str): Path to the parent directory.

    Returns:
        list: List of paths to files.
    """
    # List comprehension to find all files
    return [os.path.join(parent_path, f) for f in os.listdir(parent_path) if os.path.isfile(os.path.join(parent_path, f))]

def load_image(img_path):
    """
    Helper function to load an image and convert it to a numpy array.

    Parameters:
        img_path (str): Path to the image file.

    Returns:
        np.ndarray: Loaded image as a numpy array with shape (H, W) for grayscale images
                    or (H, W, C) for color images (e.g., RGB or RGBA).
    """
    img = Image.open(img_path)
    return np.array(img)

def plot_image(image: np.ndarray, title: str = "Image", cmap: str = None):
    """
    Plot an image with an optional title and colormap.

    Parameters:
        image (np.ndarray): The image to plot. Can be 2D (grayscale) or 3D (RGB/RGBA).
        title (str): The title of the plot. Default is "Image".
        cmap (str): Colormap to use if the image is grayscale (2D). Default is None.

    Returns:
        None
    """
    plt.figure(figsize=(6, 6))
    if image.ndim == 2:
        plt.imshow(image, cmap=cmap)
    else:
        plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()

def save_image(image: np.ndarray, save_path: str):
    """
    Save the image to a specified path.

    Parameters:
        image (np.ndarray): The image to save. Should be in RGB, RGBA, or grayscale format.
        save_path (str): Path to save the image, including the filename and extension.
    
    Returns:
        None
    """
    # Create directory if it does not exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Check if the input is a numpy array
    if not isinstance(image, np.ndarray):
        raise TypeError("Input must be a numpy array.")

    # Convert numpy array to Image object
    try:
        img_to_save = Image.fromarray(image)
    except ValueError as e:
        raise ValueError("Failed to convert numpy array to image. Ensure the array format is correct.") from e

    # Save image to the specified path
    try:
        img_to_save.save(save_path)
        print(f"Image successfully saved at: {save_path}")
    except Exception as e:
        raise IOError(f"Failed to save image at {save_path}.") from e

def plot_loss(losses, output_dir="output"):
    """
    Plot the loss over iterations and save the plot as a PNG image.

    Parameters:
        losses (list or np.ndarray): List or array of loss values for each iteration.
        output_dir (str): Directory to save the plot image. Default is "output".
    
    Returns:
        None
    """
    # Check if losses is not empty
    if not losses:
        raise ValueError("The 'losses' list is empty. Cannot plot loss.")

    # Create output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Create the plot
    figure, ax = plt.subplots()
    ax.plot(losses, label='Loss', color='blue')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Loss per Iteration')
    ax.legend()

    # Save the plot image
    save_path = os.path.join(output_dir, "loss.png")
    figure.savefig(save_path)
    print(f"Loss plot saved at: {save_path}")
    plt.close(figure)  # Close the figure to free up resources

def histogram(dct, x_label="Labels", y_label="Count", title="Number of Items per Label"):
    """
    Plot a histogram from a dictionary of label counts.

    Parameters:
        dct (dict): Dictionary with labels as keys and counts as values.
        x_label (str): Label for the x-axis. Default is "Labels".
        y_label (str): Label for the y-axis. Default is "Count".
        title (str): Title of the histogram. Default is "Number of Items per Label".
    
    Returns:
        None
    """
    plt.figure(figsize=(20, 8))
    plt.bar(dct.keys(), dct.values(), color='blue')
    plt.xticks(rotation=90)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

def load_checkpoint(filename='checkpoint.pth.tar', device='cpu'):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename,weights_only=True, map_location=device)
        return checkpoint
    else:
        return None