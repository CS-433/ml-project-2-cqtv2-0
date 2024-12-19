import numpy as np
import torchvision.transforms as transforms
from scipy.ndimage import rotate, sobel, map_coordinates, gaussian_filter, shift
from math import floor, ceil
import gc
from skimage.transform import resize
SIZE = 400

def shift_images(images, labels, ver=(-30, 30), hor=(-30, 30), rng=np.random.RandomState(seed=42)):
    """This method is used to shift an image horizontally.

    Args:
        arr (_type_): _description_
        n (int, optional): _description_. Defaults to 0.
    """
    shifteds = []
    h = rng.randint(ver[0], ver[1])
    v = rng.randint(hor[0], hor[1])
    shifted_images = np.array([shift(image, shift = (0, h, v), mode='reflect') for image in images])
    shifted_labels = np.array([shift(label, shift = (h, v), mode='reflect') for label in labels])
    return shifted_images, shifted_labels

def apply_gaussian_noise(image, mean=0, std=0.1, rng=np.random.RandomState(seed=42)):
    """
    Apply Gaussian noise to the image.

    Parameters:
    - image: Image tensor of shape (3, 400, 400).
    - mean: Mean of the Gaussian noise.
    - std: Standard deviation of the Gaussian noise.
    - rng: Random number generator.

    Returns:
    - Image tensor with Gaussian noise.
    """
    noise = rng.normal(mean, std, image.shape)
    noisy_image = np.clip(image + noise, 0, 1)
    return noisy_image

def convert_to_grayscale(image):
    """
    Convert an image to grayscale without dropping the channels.

    Args:
        image (np.array): The image to convert, with shape (height, width, channels).

    Returns:
        np.array: The grayscale image with the same number of channels.
    """
    # Calculate the grayscale values by averaging the color channels
    image = np.transpose(image, (1, 2, 0))
    grayscale = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()
    ])
    grayscale_image = grayscale(image).numpy()
    # Replicate the grayscale values across the original number of channels
    return grayscale_image

def flip_tensors(tensor_rgb, tensor_gray, horizontal=False, vertical=False):
    """
    Flip tensors horizontally and/or vertically.

    Parameters:
    - tensor_rgb: Tensor of shape (n, 3, 400, 400)
    - tensor_gray: Tensor of shape (n, 400, 400)
    - horizontal: Boolean flag to perform horizontal flip
    - vertical: Boolean flag to perform vertical flip

    Returns:
    - Flipped RGB and grayscale tensors
    """
    flipped_x = []
    flipped_y = []
    if horizontal:
        flipped_rgb = tensor_rgb[:, :, :, ::-1]
        flipped_gray = tensor_gray[:, :, ::-1]
        flipped_x.append(flipped_rgb)
        flipped_y.append(flipped_gray)
    if vertical:
        flipped_rgb = tensor_rgb[:, :, ::-1, :]
        flipped_gray = tensor_gray[:, ::-1, :]
        flipped_x.append(flipped_rgb)
        flipped_y.append(flipped_gray)
    
    flipped_x = np.concatenate(flipped_x, axis=0)
    flipped_y = np.concatenate(flipped_y, axis=0)
    return flipped_x, flipped_y


def approximate_color(y, threshold=0.9):
    y[y < threshold] = 0
    return y

def join(x, y):
    """This function joins an image and its corresponding mask.

    Args:
        x (np.array): The image to join.
        y (np.array): The mask to join.

    Returns:
        np.array: The joined image and mask.
    """
    return np.concatenate([x, y], axis=0)

def change_brightness(image, min_value=0.5, max_value=1.5, rng=np.random.RandomState(seed=42)):
    """This function changes the brightness of an image.

    Args:
        image (np.array): The image to change its brightness.
        min_value (float, optional): The minimum value of the brightness. Defaults to 0.5.
        max_value (float, optional): The maximum value of the brightness. Defaults to 1.5.
        rng (np.random.RandomState): The random number generator.

    Returns:
        np.array: The image with changed brightness.
    """
    x = rng.uniform(min_value, max_value)
    return np.clip(image * x, 0, 1)

def rotate_images_and_masks(images, masks, angle=90):
    """
    Rotate a batch of images and their corresponding masks by a specified angle.

    Args:
        images (np.array): Batch of images with shape (n, 3, m, m).
        masks (np.array): Batch of masks with shape (n, m, m).
        angle (float): The angle to rotate the images and masks.

    Returns:
        np.array: The rotated images.
        np.array: The rotated masks.
    """
    rotated_images = np.zeros_like(images)
    rotated_masks = np.zeros_like(masks)
    
    for i in range(images.shape[0]):
        for j in range(images.shape[1]):
            rotated_images[i, j] = rotate(images[i, j], angle, reshape=False, mode='reflect')
        rotated_masks[i] = rotate(masks[i], angle, reshape=False, mode='reflect')
    
    return rotated_images, rotated_masks

def elastic_transform(image, alpha=50., sigma=0.07):
    """This function applies an elastic transformation to an image.

    Args:
        image (np.array): The image to apply the transformation to.
        alpha (int, optional): Magnitude of displacement. Defaults to 50..
        sigma (float, optional): Smoothness of displacements. Defaults to 0.07.

    Returns:
        np.array: The transformed image
    """
    if image.ndim == 3:
        image = np.transpose(image, (1, 2, 0))
        
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ElasticTransform(alpha=alpha, sigma=sigma),
        transforms.ToTensor()
    ])
    return transform(image).numpy()
    
def balance_data(x_train, y_train, test_ind, diag_roads, rng=np.random.RandomState(seed=42)):
    """This function balances the data.

    Args:
        x (np.darray): The images to balance.
        y (np.darray): The masks to balance.
        test_ind (np.darray): The indices of the test set.
        diag_roads (np.darray): The indices to balance.
        rng (np.random.RandomState, optional): Random generator. Defaults to np.random.RandomState(seed=42).
    """
    possible_indices = [i for i in range(100) if i not in diag_roads and not i in test_ind]
    str_ind = rng.choice(diag_roads, size=len(possible_indices), replace=True)
    indices = np.concatenate([str_ind, possible_indices])
    x_test, y_test = x_train[test_ind], y_train[test_ind]
    x_train, y_train = x_train[indices], y_train[indices]
    return x_train, y_train, x_test, y_test

def data_augmentation(x_train, y_train, x_test, y_test, pipeline, rng=np.random.RandomState(42)):
    """This function preprocesses the input data.

    Args:
        x_train (n, 3, m, m): The training images.
        y_train (n, m, m): The training masks.
        x_test (r, 3, m, m): The test images.
        y_test (r, m, m): The test masks.
        rng (np.random.RandomState, optional): The random generator. Defaults to np.random.RandomState(42).

    Returns:
        _type_: _description_
    """
    for operation in pipeline:
        match operation:
            case 'rot':
                angles = [45, 90, 135, 180]
                x_c, y_c = x_train.copy(), y_train.copy()
                for angle in angles:
                    x, y = rotate_images_and_masks(x_c, y_c, angle=angle)
                    x_train = np.concatenate([x_train, x], axis=0)
                    y_train = np.concatenate([y_train, y], axis=0)
            case 'bright':
                x = change_brightness(x_train, rng=rng)
                y = y_train.copy()
                x_train = np.concatenate([x_train, x], axis=0)
                y_train = np.concatenate([y_train, y], axis=0)
            case 'flip':
                x, y = flip_tensors(x_train, y_train, horizontal=True, vertical=True)
                x_train = np.concatenate([x_train, x], axis=0)
                y_train = np.concatenate([y_train, y], axis=0)
            case 'gaussian':
                x = apply_gaussian_noise(x_train, rng=rng)
                y = y_train.copy()
                x_train = np.concatenate([x_train, x], axis=0)
                y_train = np.concatenate([y_train, y], axis=0)
            case 'shift':
                x, y = shift_images(x_train, y_train, ver=(-30, 30), hor=(-30, 30), rng=rng)
                x_train = np.concatenate([x_train, x], axis=0)
                y_train = np.concatenate([y_train, y], axis=0)
            case 'elastic':
                x = np.array([elastic_transform(x) for x in x_train])
                y = np.array([elastic_transform(y)[0] for y in y_train])
                x_train = np.concatenate([x_train, x], axis=0)
                y_train = np.concatenate([y_train, y], axis=0)
            case 'greyscale':
                x = convert_to_grayscale(x_train)
                y = y_train.copy()
                x_train = np.concatenate([x_train, x], axis=0)
                y_train = np.concatenate([y_train, y], axis=0)
            case 'resize':
                x_train = np.array([resize(x, output_shape=(3, 384, 384)) for x in x_train])
                y_train = np.array([resize(y, output_shape=(384, 384)) for y in y_train])
                if x_test is not None or y_test is not None:
                    x_test = np.array([resize(x, output_shape=(3, 384, 384)) for x in x_test])
                    y_test = np.array([resize(y, output_shape=(384, 384)) for y in y_test])
                x, y = None, None
            case _:
                raise ValueError(f"Operation not supported")
        del x, y
        gc.collect()
        print(f"Operation {operation} done, shape is {x_train.shape}")
                
    n = x_train.shape[0]
    indices = rng.permutation(n)
    x_train = x_train[indices]
    y_train = y_train[indices]
    x_train, y_train = x_train.astype(np.float32), y_train.astype(np.float32)
    return np.clip(x_train, 0, 1), np.clip(y_train, 0, 1), x_test, y_test


