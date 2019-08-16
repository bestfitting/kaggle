import xml.etree.ElementTree as ET
import torchvision
import os
import mlcrate as mlc
import numpy as np
from PIL import Image
from tqdm import tqdm
opj=os.path.join
ope=os.path.exists
from PIL import Image, ImageDraw
import shutil

# save dogs to dir by type
data_dir='../input/'
out_dir='../output/data/generative-dog-images/origin_dogs'
if ope(out_dir):
    shutil.rmtree(out_dir)
os.makedirs(out_dir, exist_ok=True)
input_img_dir=opj(data_dir,'all-dogs/all-dogs')
annotation_dir=opj(data_dir,'annotation/Annotation')

annotation_outdir = f'{out_dir}_annotation'
if ope(annotation_outdir):
    shutil.rmtree(annotation_outdir)
os.makedirs(annotation_outdir, exist_ok=True)
annotation_list = os.listdir(annotation_dir)
for annotation in annotation_list:
    annotation_dirname = annotation.split('-')[1]
    shutil.copytree(f'{annotation_dir}/{annotation}', f'{annotation_outdir}/{annotation_dirname}')

def process_an_image(dog_fname):
    dog_fullname=opj(input_img_dir, dog_fname)
    img = torchvision.datasets.folder.default_loader(dog_fullname)  # default loader

    # Get bounding box
    annotation_basename = os.path.splitext(dog_fname)[0]
    annotation_dirname = next(dirname for dirname in os.listdir(annotation_dir) if
                              dirname.startswith(annotation_basename.split('_')[0]))
    # annotation_filename = os.path.join(annotation_dir, annotation_dirname, annotation_basename)
    dog_type_dir=opj(out_dir,annotation_dirname.split('-')[1])
    os.makedirs(dog_type_dir,exist_ok=True)
    img.save(opj(dog_type_dir,dog_fname))
    return dog_fname


dog_fnames=os.listdir(input_img_dir)
# for i,dog_fname in enumerate(dog_fnames):
#     process_an_image(dog_fname)
pool=mlc.SuperPool()
imgs=pool.map(process_an_image,dog_fnames)
print(out_dir)

