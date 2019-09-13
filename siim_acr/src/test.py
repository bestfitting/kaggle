# coding: utf-8
import argparse
from tqdm import tqdm
import pandas as pd

import torch
import torch.optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
from torch.nn import DataParallel
import torch.nn.functional as F
from torch.autograd import Variable

from config.config import *
from utils.common_util import *
from networks.imageunet import init_network
from datasets.datasets import SiimDataset
from utils.augment_util import *
from utils.log_util import Logger
from utils.mask_functions import run_length_encode

datasets_names = ['train', 'test', 'val', 'nih', 'chexpert']
split_types = ['split', 'cropv3_split', 'lung_split']
split_names = ['random_folds4', 'random_folds10']
augment_list = ['default', 'fliplr']

parser = argparse.ArgumentParser(description='PyTorch SIIM Segmentation')
parser.add_argument('--out_dir', type=str, help='destination where predicted result should be saved')
parser.add_argument('--gpu_id', default='0', type=str, help='gpu id used for predicting (default: 0)')
parser.add_argument('--arch', default='unet_resnet34_cbam_v0a', type=str,
                    help='model architecture (default: unet_resnet34_cbam_v0a)')
parser.add_argument('--img_size', default=768, type=int, help='image size (default: 768)')
parser.add_argument('--batch_size', default=4, type=int, help='train mini-batch size (default: 4)')
parser.add_argument('--workers', default=3, type=int, help='number of data loading workers (default: 3)')
parser.add_argument('--split_type', default='split', type=str, choices=split_types,
                    help='split type options: ' + ' | '.join(split_types) + ' (default: split)')
parser.add_argument('--split_name', default='random_folds10', type=str, choices=split_names,
                    help='split name options: ' + ' | '.join(split_names) + ' (default: random_folds10)')
parser.add_argument('--fold', default=0, type=int, help='index of fold (default: 0)')
parser.add_argument('--augment', default='default', type=str, help='test augmentation (default: default)')
parser.add_argument('--dataset', default='test', type=str, choices=datasets_names,
                    help='dataset options: ' + ' | '.join(datasets_names) + ' (default: test)')
parser.add_argument('--predict_epoch', default='final', type=str, help='number epoch to predict (eg. final, top1, 015)')
parser.add_argument('--crop_version', default=None, type=str, help='the cropped version (default: None)')
parser.add_argument('--threshold', default=0.5, type=float, help='the threshold (default: 0.5)')
parser.add_argument('--predict_pos', type=int, default=0, help='only predict positive samples (default: False)')
parser.add_argument('--ema', action='store_true', default=False)

def main():
    args = parser.parse_args()

    log_out_dir = opj(RESULT_DIR, 'logs', args.out_dir, 'fold%d' % args.fold)
    if not ope(log_out_dir):
        os.makedirs(log_out_dir)
    log = Logger()
    log.open(opj(log_out_dir, 'log.submit.txt'), mode='a')

    if args.ema:
        network_path = opj(RESULT_DIR, 'models', args.out_dir, 'fold%d' % args.fold, '%s_ema.pth' % args.predict_epoch)
    else:
        network_path = opj(RESULT_DIR, 'models', args.out_dir, 'fold%d' % args.fold, '%s.pth' % args.predict_epoch)

    submit_out_dir = opj(RESULT_DIR, 'submissions', args.out_dir, 'fold%d' % args.fold, 'epoch_%s' % args.predict_epoch)
    log.write(">> Creating directory if it does not exist:\n>> '{}'\n".format(submit_out_dir))
    if not ope(submit_out_dir):
        os.makedirs(submit_out_dir)

    # setting up the visible GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    args.augment = args.augment.split(',')
    for augment in args.augment:
        if augment not in augment_list:
            raise ValueError('Unsupported or unknown test augmentation: {}!'.format(augment))

    model_params = {}
    model_params['architecture'] = args.arch
    model = init_network(model_params)

    log.write(">> Loading network:\n>>>> '{}'\n".format(network_path))
    checkpoint = torch.load(network_path)
    model.load_state_dict(checkpoint['state_dict'])
    log.write(">>>> loaded network:\n>>>> epoch {}\n".format(checkpoint['epoch']))

    # moving network to gpu and eval mode
    model = DataParallel(model)
    model.cuda()
    model.eval()

    # Data loading code
    dataset = args.dataset
    if dataset == 'train':
        test_split_file = opj(DATA_DIR, args.split_type, 'train.csv')
    elif dataset == 'test':
        test_split_file = opj(DATA_DIR, args.split_type, 'test.csv')
    elif dataset == 'val':
        test_split_file = opj(DATA_DIR, args.split_type, args.split_name, 'random_valid_cv%d.csv' % args.fold)
    elif dataset == 'nih':
        test_split_file = opj(DATA_DIR, args.split_type, 'nih_112120.csv')
    elif dataset == 'chexpert':
        test_split_file = opj(DATA_DIR, args.split_type, 'chexpert_188521.csv')
    else:
        raise ValueError('Unsupported or unknown dataset: {}!'.format(dataset))
    test_dataset = SiimDataset(
        test_split_file,
        img_size=args.img_size,
        mask_size=args.img_size,
        transform=None,
        return_label=False,
        crop_version=args.crop_version,
        dataset=args.dataset,
        predict_pos=args.predict_pos,
    )
    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    for augment in args.augment:
        test_loader.dataset.transform = eval('augment_%s' % augment)
        unaugment_func = eval('unaugment_%s' % augment)
        sub_submit_out_dir = opj(submit_out_dir, augment)
        if not ope(sub_submit_out_dir):
            os.makedirs(sub_submit_out_dir)
        with torch.no_grad():
            predict(test_loader, model, sub_submit_out_dir, dataset, args, unaugment_func=unaugment_func)

def predict(test_loader, model, submit_out_dir, dataset, args, unaugment_func=None):
    all_probs = []
    img_ids = np.array(test_loader.dataset.img_ids)
    for it, iter_data in tqdm(enumerate(test_loader, 0), total=len(test_loader)):
        images, indices = iter_data
        images = Variable(images.cuda(), volatile=True)
        outputs = model(images)
        logits = outputs

        probs = F.sigmoid(logits).data
        probs = probs.cpu().numpy().reshape(-1, args.img_size, args.img_size)
        probs = (probs * 255).astype('uint8')

        for pred_prob in probs:
            pred_prob = unaugment_func(pred_prob)
            all_probs.append(pred_prob)
    all_probs = np.array(all_probs)
    img_ids = img_ids[:len(all_probs)]

    rles = []
    for i in tqdm(range(len(all_probs))):
        if args.dataset in ['test', 'val'] and args.crop_version is None:
            pred_prob = all_probs[i]
            pred_prob = cv2.resize(pred_prob, dsize=(IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
            mask = (pred_prob > args.threshold * 255).astype(np.float32)
            if mask.sum() == 0:
                rle = '-1'
            else:
                rle = run_length_encode(mask)
        else:
            rle = '-1'
        rles.append(rle)
    result_df = pd.DataFrame({ID: img_ids, TARGET: rles})

    np.savez_compressed(opj(submit_out_dir, 'prob_%s.npz' % dataset), data=all_probs)
    result_df.to_csv(opj(submit_out_dir, 'results_%s.csv.gz' % dataset), index=False, compression='gzip')
    np.save(opj(submit_out_dir, 'img_ids_%s.npy' % dataset), img_ids)

if __name__ == '__main__':
    print('%s: calling main function ... \n' % os.path.basename(__file__))
    main()
    print('\nsuccess!')
