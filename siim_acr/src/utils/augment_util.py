import cv2
from imgaug import augmenters as iaa
from config.config import *

def train_multi_augment9(image, mask):
    augment_func_list = [
        augment_default,
        augment_fliplr,
        augment_random_rotate,
        augment_random_crop,
        augment_random_cover,
        augment_random_brightness_multiply,
        augment_random_brightness_shift,
        augment_random_Gaussian,
    ]
    c = np.random.choice(len(augment_func_list))
    image, mask = augment_func_list[c](image, mask)

    return image, mask

###########################################################################################

def augment_default(image, mask=None):
    return image, mask

def unaugment_default(prob):
    return prob

def augment_fliplr(image, mask=None):
    image = np.fliplr(image)
    if mask is not None:
        mask = np.fliplr(mask)
    return image, mask

def unaugment_fliplr(prob):
    prob = np.fliplr(prob)
    return prob

def augment_random_brightness_shift(image, mask=None, limit=0.2):
    alpha = np.random.uniform(-1 * limit, limit) * image.max()

    image = image + alpha
    image = np.clip(image, 0, 255).astype('uint8')
    return image, mask

def augment_random_brightness_multiply(image, mask=None, limit=0.2):
    alpha = np.random.uniform(-1 * limit, limit)
    image = image * (1-alpha)
    image = np.clip(image, 0, 255).astype('uint8')
    return image, mask

def augment_random_rotate(image, mask=None, limit=30):
    cols, rows = image.shape[:2]
    assert cols == rows
    size = cols

    rotate = np.random.randint(-1 * limit, limit)

    M = cv2.getRotationMatrix2D((int(size / 2), int(size / 2)), rotate, 1)
    image = cv2.warpAffine(image, M, (size, size))
    if mask is not None:
        mask = cv2.warpAffine(mask, M, (size, size))
    return image, mask

def augment_random_cover(image, mask=None, cover_ratio=0.2):
    cols, rows = image.shape[:2]
    cols == min([cols, rows])
    size = cols

    cover_size = max(int(size * cover_ratio), 1)
    if cover_size >= size:
        return image
    min_row = np.random.choice(size - cover_size)
    min_col = np.random.choice(size - cover_size)

    image[min_row:min_row+cover_size, min_col:min_col+cover_size] = 0
    if mask is not None:
        mask[min_row:min_row + cover_size, min_col:min_col + cover_size] = 0
    return image, mask

def augment_random_crop(image, mask=None, limit=0.10):

    H, W = image.shape[:2]

    dy = int(H*limit)
    y0 =   np.random.randint(0,dy)
    y1 = H-np.random.randint(0,dy)

    dx = int(W*limit)
    x0 =   np.random.randint(0,dx)
    x1 = W-np.random.randint(0,dx)

    image, mask = do_random_crop( image, mask, x0, y0, x1, y1 )
    return image, mask

def do_random_crop( image, mask=None, x0=0, y0=0, x1=1, y1=1 ):
    height, width = image.shape[:2]
    image = image[y0:y1,x0:x1]
    image = cv2.resize(image,dsize=(width,height), interpolation=cv2.INTER_LINEAR)

    if mask is not None:
        mask  = mask [y0:y1,x0:x1]
        mask  = cv2.resize(mask,dsize=(width,height), interpolation=cv2.INTER_LINEAR)

    return image, mask

def augment_random_Gaussian(image, mask=None, limit=0.3):
    sigma = np.random.uniform(0, limit)
    aug = iaa.GaussianBlur(sigma=sigma)
    image = aug.augment_images([image])[0]
    return image, mask
