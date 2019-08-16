''' Datasets
    This file contains definitions for our CIFAR, ImageFolder, and HDF5 datasets
'''
import os
import os.path
import sys
from PIL import Image,ImageDraw
import numpy as np
from tqdm import tqdm, trange

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.datasets.utils import download_url, check_integrity
import torch.utils.data as data
from torch.utils.data import DataLoader
import os,torchvision
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
  with open(path, 'rb') as f:
    img = Image.open(f)
    return img.convert('RGB')


def accimage_loader(path):
  import accimage
  try:
    return accimage.Image(path)
  except IOError:
    # Potentially a decoding problem, fall back to PIL.Image
    return pil_loader(path)
def check_img(img):

  os.makedirs('/data/tmp/dogs',exist_ok=True)
  torchvision.utils.save_image((img+1)/2, f'/data/tmp/dogs/img_{str(np.random.randint(0,1000)).zfill(4)}.jpg')
  #
def default_loader(path):
  from torchvision import get_image_backend
  if get_image_backend() == 'accimage':
    return accimage_loader(path)
  else:
    return pil_loader(path)

from torch.utils.data import Dataset


def find_classes(dir):
  classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
  classes.sort()
  class_to_idx = {classes[i]: i for i in range(len(classes))}
  return classes, class_to_idx



class DogDataset(Dataset):
  def __init__(self, root, transform,image_size=64,load_in_mem=True,index_filename='imagenet_imgs.npz', **kwargs):
    self.transform1 = transforms.Resize(image_size, kwargs['resize_mode'])
    self.transform2 = transform
    self.use_dog_cnt = kwargs['use_dog_cnt']
    cached_fname = root#
    self.cropped_imgs = pickle.load(open(cached_fname, 'rb'))
    self.imgs = []
    self.labels=[]
    classes=[]
    # for dog_name, dog_type, img_cropped in tqdm(self.cropped_imgs):
    for dog_name, dog_type, img_cropped in tqdm(self.cropped_imgs):
      if self.transform1 is not None:
        img = self.transform1(img_cropped)
      if dog_type not in classes:
        classes.append(dog_type)
      self.imgs.append(img)
      self.labels.append(dog_type)
    self.classes=classes
    self.class_to_idx = {classes[i]: i for i in range(len(classes))}

  def __getitem__(self, index):
    img = self.imgs[index]

    if self.transform2 is not None:
      img = self.transform2(img)
    # check_img(img)
    label=self.class_to_idx[self.labels[index]]

    if self.use_dog_cnt:
      return img, [label,0]
    else:
      return img,label

  def __len__(self):
    return len(self.imgs)


import xml.etree.ElementTree as ET
def get_image_bboxes(image_path):
    bbox_fname = image_path.replace('origin_dogs', 'origin_dogs_annotation')
    bbox_fname = os.path.splitext(bbox_fname)[0]

    # Get bounding box
    tree = ET.parse(bbox_fname)
    root = tree.getroot()
    objects = root.findall('object')
    bboxes = []
    for o in objects:
        bndbox = o.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        bbox = (xmin, ymin, xmax, ymax)
        bboxes.append(bbox)
    return bboxes


def make_dataset(dir, class_to_idx, use_bbox):
    images = []
    dir = os.path.expanduser(dir)
    # for target in tqdm(sorted(os.listdir(dir))):
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    image_path = os.path.join(root, fname)
                    if use_bbox:
                        bboxes = get_image_bboxes(image_path)
                        item = (image_path, class_to_idx[target], np.array(bboxes, dtype=int))
                    else:
                        item = (image_path, class_to_idx[target])
                    images.append(item)

    return images


def get_origin_max_crop(img, bbox, image_size):
    xmin, ymin, xmax, ymax = bbox
    size = img.size
    bbox_size = (xmax - xmin, ymax - ymin)
    extend = (np.max(bbox_size) - np.min(bbox_size)) // 2
    if bbox_size[1] > bbox_size[0]:
        left = bbox[0]
        right = size[0] - bbox[2]
        max_extend = np.min((left, right))
        extend = np.minimum(extend, max_extend)
        bbox[0] -= extend
        bbox[2] += extend
    else:
        top = bbox[1]
        bottom = size[1] - bbox[3]
        max_extend = np.min((top, bottom))
        extend = np.minimum(extend, max_extend)
        bbox[1] -= extend
        bbox[3] += extend

    cropped = img.crop(bbox)
    cropped = cropped.resize((image_size, image_size), Image.ANTIALIAS)
    return cropped


def resize_img_bboxes(img, bboxes, image_size):
    short_side = np.max(img.size)
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox
        bbox_size = (xmax - xmin, ymax - ymin)
        side = np.min(bbox_size)
        if side < short_side:
            short_side = side

    ratio = image_size / short_side
    resize_to = (int(img.size[0]*ratio), int(img.size[1]*ratio))
    img = img.resize(resize_to, Image.ANTIALIAS)
    for bbox in bboxes:
        for i in range(len(bbox)):
            bbox[i] = int(bbox[i] * ratio)

    return img, bboxes


class ImageFolder(data.Dataset):
  """A generic data loader where the images are arranged in this way: ::

      root/dogball/xxx.png
      root/dogball/xxy.png
      root/dogball/xxz.png

      root/cat/123.png
      root/cat/nsdf3.png
      root/cat/asd932_.png

  Args:
      root (string): Root directory path.
      transform (callable, optional): A function/transform that  takes in an PIL image
          and returns a transformed version. E.g, ``transforms.RandomCrop``
      target_transform (callable, optional): A function/transform that takes in the
          target and transforms it.
      loader (callable, optional): A function to load an image given its path.

   Attributes:
      classes (list): List of the class names.
      class_to_idx (dict): Dict with items (class_name, class_index).
      imgs (list): List of (image path, class_index) tuples
  """

  def __init__(self, root, transform=None, target_transform=None,
               loader=default_loader, load_in_mem=False,
               index_filename='imagenet_imgs.npz', **kwargs):
    classes, class_to_idx = find_classes(root)
    self.crop_mode = kwargs['crop_mode']
    self.image_size = kwargs['image_size']
    self.use_dog_cnt=kwargs['use_dog_cnt']
    self.n_classes = kwargs['n_classes']
    self.without_multi_dogs = kwargs['without_multi_dogs']
    self.use_bbox = self.crop_mode > 0
    # Load pre-computed image directory walk
    if os.path.exists(index_filename):
      print('Loading pre-saved Index file %s...' % index_filename)
      imgs = np.load(index_filename)['imgs']
    # If first time, walk the folder directory and save the
    # results to a pre-computed file.
    else:
      print('Generating  Index file %s...' % index_filename)
      imgs = make_dataset(root, class_to_idx, self.use_bbox)
      # np.savez_compressed(index_filename, **{'imgs' : imgs})
    if len(imgs) == 0:
      raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                           "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

    self.root = root
    self.imgs = imgs
    self.classes = classes
    self.class_to_idx = class_to_idx
    self.transform = transform
    self.target_transform = target_transform
    self.loader = loader
    self.debug = kwargs['debug']

    if self.debug:
        self.imgs = self.imgs[:100]

    print('Loading all images into memory...')
    self.data, self.labels, self.cropped_data, self.bboxes_list = [], [], [], []
    for index in tqdm(range(len(self.imgs)),desc='load images'):
    # for index in range(len(self.imgs)):
      path, target = imgs[index][0], imgs[index][1]
      img = self.loader(path)

      if self.use_bbox:
        bboxes = imgs[index][2]
        self.bboxes_list.append(bboxes)
        cropped = [get_origin_max_crop(img, bbox, self.image_size) for bbox in bboxes]
        if self.without_multi_dogs and len(cropped) > 1:
            continue
        if self.crop_mode >= 2:
            img = transforms.Resize(self.image_size, Image.ANTIALIAS)(img)
        else:
            img = None
        self.cropped_data.append(cropped)
      else:
        img = img.resize((self.image_size, self.image_size), Image.ANTIALIAS)
      self.data.append(img)
      self.labels.append(target)
    print('dataset len', len(self.data))


  def __getitem__(self, index):
    """
    Args:
        index (int): Index

    Returns:
        tuple: (image, target) where target is class_index of the target class.
    """
    img = self.data[index]
    target = self.labels[index]
    if self.crop_mode > 0:
      cropped_list = self.cropped_data[index]
      ix = np.random.randint(len(cropped_list))
      cropped = cropped_list[ix]
      if self.crop_mode == 1:
          img = cropped

      elif self.crop_mode == 2:
          ix = np.random.randint(2)
          if ix == 1:
              img = cropped

      elif self.crop_mode == 3:
          ix = np.random.randint(3)
          if ix == 1:
              img = cropped
          elif ix == 2:
              img = RandomCropLongEdge()(img)

      elif self.crop_mode == 4:
          if len(cropped_list) > 1:
              img = cropped
          else:
              ix = np.random.randint(3)
              if ix == 1:
                  img = cropped
              elif ix == 2:
                  img = RandomCropLongEdge()(img)

      elif self.crop_mode == 5:
          v = np.random.rand()
          if v < 0.4:
              img = cropped
          elif v < 0.8:
              img = RandomCropLongEdge()(img)

      elif self.crop_mode == 6:
          v = np.random.rand()
          if v < 0.4:
              img = cropped
          elif v < 0.6:
              img = RandomCropLongEdge()(img)

      elif self.crop_mode == 7:
          v = np.random.rand()
          if v < 0.2:
              img = cropped
          elif v < 0.6:
              img = RandomCropLongEdge()(img)

      elif self.crop_mode == 8:
          v = np.random.rand()
          if v < 0.5:
              img = cropped
          else:
              img = RandomCropLongEdge()(img)

      img = img.resize((self.image_size, self.image_size), Image.ANTIALIAS)

    if self.debug:
        output_dir = '/data4/data/gan_dogs/result/tmp/crop_mode%d'%self.crop_mode
        os.makedirs(output_dir, exist_ok=True)
        img_cnt = len(os.listdir(output_dir))
        if img_cnt < 300:
            img.save(f'{output_dir}/{str(img_cnt)}.jpg')

    if self.transform is not None:
      img = self.transform(img)
    if self.target_transform is not None:
      target = self.target_transform(target)

    if  self.use_dog_cnt:
      dog_cnt=len(cropped_list)
      if dog_cnt>4:
        dog_cnt=4
      return img, [int(target),dog_cnt-1]
    else:
      return img, int(target)

  def __len__(self):
    return len(self.data)

  def __repr__(self):
    fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
    fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
    fmt_str += '    Root Location: {}\n'.format(self.root)
    tmp = '    Transforms (if any): '
    fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
    tmp = '    Target Transforms (if any): '
    fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
    return fmt_str


class RandomCropLongEdge(object):
  """Crops the given PIL Image on the long edge with a random start point.
  Args:
      size (sequence or int): Desired output size of the crop. If size is an
          int instead of sequence like (h, w), a square crop (size, size) is
          made.
  """
  def __call__(self, img):
    """
    Args:
        img (PIL Image): Image to be cropped.
    Returns:
        PIL Image: Cropped image.
    """
    size = (min(img.size), min(img.size))
    # Only step forward along this edge if it's the long edge
    i = (0 if size[0] == img.size[0]
          else np.random.randint(low=0,high=img.size[0] - size[0]))
    j = (0 if size[1] == img.size[1]
          else np.random.randint(low=0,high=img.size[1] - size[1]))
    return transforms.functional.crop(img, j, i, size[0], size[1])

  def __repr__(self):
    return self.__class__.__name__


''' ILSVRC_HDF5: A dataset to support I/O from an HDF5 to avoid
    having to load individual images all the time. '''
import h5py as h5
import torch
class ILSVRC_HDF5(data.Dataset):
  def __init__(self, root, transform=None, target_transform=None,
               load_in_mem=False, train=True,download=False, validate_seed=0,
               val_split=0, **kwargs): # last four are dummies

    self.root = root
    self.num_imgs = len(h5.File(root, 'r')['labels'])

    # self.transform = transform
    self.target_transform = target_transform

    # Set the transform here
    self.transform = transform

    # load the entire dataset into memory?
    self.load_in_mem = load_in_mem

    # If loading into memory, do so now
    if self.load_in_mem:
      print('Loading %s into memory...' % root)
      with h5.File(root,'r') as f:
        self.data = f['imgs'][:]
        self.labels = f['labels'][:]

  def __getitem__(self, index):
    """
    Args:
        index (int): Index

    Returns:
        tuple: (image, target) where target is class_index of the target class.
    """
    # If loaded the entire dataset in RAM, get image from memory
    if self.load_in_mem:
      img = self.data[index]
      target = self.labels[index]

    # Else load it from disk
    else:
      with h5.File(self.root,'r') as f:
        img = f['imgs'][index]
        target = f['labels'][index]


    # if self.transform is not None:
        # img = self.transform(img)
    # Apply my own transform
    img = ((torch.from_numpy(img).float() / 255) - 0.5) * 2

    if self.target_transform is not None:
      target = self.target_transform(target)

    return img, int(target)

  def __len__(self):
      return self.num_imgs
      # return len(self.f['imgs'])

import pickle
class CIFAR10(dset.CIFAR10):

  def __init__(self, root, train=True,
           transform=None, target_transform=None,
           download=True, validate_seed=0,
           val_split=0, load_in_mem=True, **kwargs):
    self.root = os.path.expanduser(root)
    self.transform = transform
    self.target_transform = target_transform
    self.train = train  # training set or test set
    self.val_split = val_split

    if download:
      self.download()

    if not self._check_integrity():
      raise RuntimeError('Dataset not found or corrupted.' +
                           ' You can use download=True to download it')

    # now load the picked numpy arrays
    self.data = []
    self.labels= []
    for fentry in self.train_list:
      f = fentry[0]
      file = os.path.join(self.root, self.base_folder, f)
      fo = open(file, 'rb')
      if sys.version_info[0] == 2:
        entry = pickle.load(fo)
      else:
        entry = pickle.load(fo, encoding='latin1')
      self.data.append(entry['data'])
      if 'labels' in entry:
        self.labels += entry['labels']
      else:
        self.labels += entry['fine_labels']
      fo.close()

    self.data = np.concatenate(self.data)
    # Randomly select indices for validation
    if self.val_split > 0:
      label_indices = [[] for _ in range(max(self.labels)+1)]
      for i,l in enumerate(self.labels):
        label_indices[l] += [i]
      label_indices = np.asarray(label_indices)

      # randomly grab 500 elements of each class
      np.random.seed(validate_seed)
      self.val_indices = []
      for l_i in label_indices:
        self.val_indices += list(l_i[np.random.choice(len(l_i), int(len(self.data) * val_split) // (max(self.labels) + 1) ,replace=False)])

    if self.train=='validate':
      self.data = self.data[self.val_indices]
      self.labels = list(np.asarray(self.labels)[self.val_indices])

      self.data = self.data.reshape((int(50e3 * self.val_split), 3, 32, 32))
      self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

    elif self.train:
      print(np.shape(self.data))
      if self.val_split > 0:
        self.data = np.delete(self.data,self.val_indices,axis=0)
        self.labels = list(np.delete(np.asarray(self.labels),self.val_indices,axis=0))

      self.data = self.data.reshape((int(50e3 * (1.-self.val_split)), 3, 32, 32))
      self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
    else:
      f = self.test_list[0][0]
      file = os.path.join(self.root, self.base_folder, f)
      fo = open(file, 'rb')
      if sys.version_info[0] == 2:
        entry = pickle.load(fo)
      else:
        entry = pickle.load(fo, encoding='latin1')
      self.data = entry['data']
      if 'labels' in entry:
        self.labels = entry['labels']
      else:
        self.labels = entry['fine_labels']
      fo.close()
      self.data = self.data.reshape((10000, 3, 32, 32))
      self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

  def __getitem__(self, index):
    """
    Args:
        index (int): Index
    Returns:
        tuple: (image, target) where target is index of the target class.
    """
    img, target = self.data[index], self.labels[index]

    # doing this so that it is consistent with all other datasets
    # to return a PIL Image
    img = Image.fromarray(img)

    if self.transform is not None:
      img = self.transform(img)

    if self.target_transform is not None:
      target = self.target_transform(target)

    return img, target

  def __len__(self):
      return len(self.data)


class CIFAR100(CIFAR10):
    base_folder = 'cifar-100-python'
    url = "http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
