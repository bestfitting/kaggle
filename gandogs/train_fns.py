''' train_fns.py
Functions for the main loop of training different conditional image models
'''
import torch
import torch.nn as nn
import torchvision
import os

import utils
import losses
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
# Dummy training function for debugging
def dummy_training_function():
  def train(x, y):
    return {}
  return train


def GAN_training_function(G, D, GD, z_, y_, ema, state_dict, config):
  if config['loss_version']=='hinge2':
    generator_loss = losses.loss_hinge_gen
    discriminator_loss = losses.loss_hinge_dis2
  elif config['loss_version']=='rals':
    generator_loss = losses.loss_rals_gen
    discriminator_loss = losses.loss_rals_dis
  elif config['loss_version']=='hinge_rals':
    generator_loss = losses.loss_hinge_rals_gen
    discriminator_loss = losses.loss_hinge_rals_dis
  else:
    generator_loss = losses.loss_hinge_gen
    discriminator_loss = losses.loss_hinge_dis

  def train_mode_seeing(x, y):
    G.optim.zero_grad()
    D.optim.zero_grad()
    # How many chunks to split x and y into?
    x = torch.split(x, config['batch_size'])
    y = torch.split(y, config['batch_size'])
    counter = 0

    # Optionally toggle D and G's "require_grad"
    if config['toggle_grads']:
      utils.toggle_grad(D, True)
      utils.toggle_grad(G, False)

    for step_index in range(config['num_D_steps']):
      # If accumulating gradients, loop multiple times before an optimizer step
      D.optim.zero_grad()
      for accumulation_index in range(config['num_D_accumulations']):
        z_.sample_()
        y_.sample_()
        D_fake, D_fake_features, D_real, D_real_features = GD(z_[:config['batch_size']], y_[:config['batch_size']],
                                                              x[counter], y[counter], train_G=False,
                                                              split_D=config['split_D'])
        # Compute components of D's loss, average them, and divide by
        # the number of gradient accumulations
        D_loss_real, D_loss_fake = discriminator_loss(D_fake, D_real)
        D_loss = (D_loss_real + D_loss_fake) / float(config['num_D_accumulations'])
        D_loss.backward()
        counter += 1

      # Optionally apply ortho reg in D
      if config['D_ortho'] > 0.0:
        # Debug print to indicate we're using ortho reg in D.
        # print('using modified ortho reg in D')
        utils.ortho(D, config['D_ortho'])

      if config['clip_norm'] is not None:
        torch.nn.utils.clip_grad_norm_(D.parameters(), config['clip_norm'])
      D.optim.step()

    # Optionally toggle "requires_grad"
    if config['toggle_grads']:
      utils.toggle_grad(D, False)
      utils.toggle_grad(G, True)

    # Zero G's gradients by default before training G, for safety
    G.optim.zero_grad()

    # If accumulating gradients, loop multiple times
    for accumulation_index in range(config['num_G_accumulations']):
      z_.sample_()
      y_.sample_()
      z1=z_.data.clone().detach()
      D_fake1, _, fake_image1= GD(z1, y_, train_G=True, split_D=config['split_D'],return_G_z=True)
      G_loss1 = generator_loss(D_fake1,D_real.detach()) / float(config['num_G_accumulations'])

      z_.sample_()
      z2 =z_.data.clone().detach()
      D_fake2, _ ,fake_image2= GD(z2, y_, train_G=True, split_D=config['split_D'],return_G_z=True)
      G_loss2 = generator_loss(D_fake2,D_real.detach()) / float(config['num_G_accumulations'])

      G_loss_gan=G_loss1+G_loss2

      # mode seeking loss
      lz = torch.mean(torch.abs(fake_image2 - fake_image1)) / torch.mean( torch.abs(z2 - z1))
      eps = 1 * 1e-5
      loss_lz = 1 / (lz + eps)
      G_loss=G_loss_gan+loss_lz
      G_loss.backward()

    # Optionally apply modified ortho reg in G
    if config['G_ortho'] > 0.0:
      # print('using modified ortho reg in G') # Debug print to indicate we're using ortho reg in G
      # Don't ortho reg shared, it makes no sense. Really we should blacklist any embeddings for this
      utils.ortho(G, config['G_ortho'],
                  blacklist=[param for param in G.shared.parameters()])

    if config['clip_norm'] is not None:
        torch.nn.utils.clip_grad_norm_(G.parameters(), config['clip_norm'])
    G.optim.step()

    # If we have an ema, update it, regardless of if we test with it or not
    if config['ema']:
      ema.update(state_dict['itr'])

    out = {'G_loss': float(G_loss.item()),
            'D_loss_real': float(D_loss_real.item()),
            'D_loss_fake': float(D_loss_fake.item())}
    # Return G's loss and the components of D's loss.
    return out

  def train(x, y):
    G.optim.zero_grad()
    D.optim.zero_grad()
    # How many chunks to split x and y into?
    x = torch.split(x, config['batch_size'])
    y = torch.split(y, config['batch_size'])
    counter = 0

    # Optionally toggle D and G's "require_grad"
    if config['toggle_grads']:
      utils.toggle_grad(D, True)
      utils.toggle_grad(G, False)

    for step_index in range(config['num_D_steps']):
      # If accumulating gradients, loop multiple times before an optimizer step
      D.optim.zero_grad()
      for accumulation_index in range(config['num_D_accumulations']):
        z_.sample_()
        y_.sample_()
        ret = GD(z_[:config['batch_size']], y_[:config['batch_size']],
                                                              x[counter], y[counter], train_G=False,
                                                              split_D=config['split_D'])
        if len(ret)>2:
          D_fake, D_fake_features, D_real, D_real_features=ret
        else:
          D_fake,  D_real = ret
        # Compute components of D's loss, average them, and divide by
        # the number of gradient accumulations
        D_loss_real, D_loss_fake = discriminator_loss(D_fake, D_real)
        D_loss = (D_loss_real + D_loss_fake) / float(config['num_D_accumulations'])
        D_loss.backward()
        counter += 1

      # Optionally apply ortho reg in D
      if config['D_ortho'] > 0.0:
        # Debug print to indicate we're using ortho reg in D.
        # print('using modified ortho reg in D')
        utils.ortho(D, config['D_ortho'])

      if config['clip_norm'] is not None:
        torch.nn.utils.clip_grad_norm_(D.parameters(), config['clip_norm'])
      D.optim.step()

    # Optionally toggle "requires_grad"
    if config['toggle_grads']:
      utils.toggle_grad(D, False)
      utils.toggle_grad(G, True)

    # Zero G's gradients by default before training G, for safety
    G.optim.zero_grad()

    # If accumulating gradients, loop multiple times
    for accumulation_index in range(config['num_G_accumulations']):
      z_.sample_()
      y_.sample_()
      ret= GD(z_, y_, train_G=True, split_D=config['split_D'])
      if len(ret)==2:
        D_fake, D_fake_features=ret
      else:
        D_fake = ret
      G_loss = generator_loss(D_fake, D_real.detach()) / float(config['num_G_accumulations'])
      if config['gdpp_loss']:
        gdpp_loss = losses.GDPPLoss(D_fake_features, D_real_features.detach(), backward=False)
        gdpp_loss = gdpp_loss / float(config['num_G_accumulations'])
        G_loss += gdpp_loss

      G_loss.backward()

    # Optionally apply modified ortho reg in G
    if config['G_ortho'] > 0.0:
      # print('using modified ortho reg in G') # Debug print to indicate we're using ortho reg in G
      # Don't ortho reg shared, it makes no sense. Really we should blacklist any embeddings for this
      utils.ortho(G, config['G_ortho'],
                  blacklist=[param for param in G.shared.parameters()])

    if config['clip_norm'] is not None:
      torch.nn.utils.clip_grad_norm_(G.parameters(), config['clip_norm'])
    G.optim.step()

    # If we have an ema, update it, regardless of if we test with it or not
    if config['ema']:
      ema.update(state_dict['itr'])

    out = {'G_loss': float(G_loss.item()),
           'D_loss_real': float(D_loss_real.item()),
           'D_loss_fake': float(D_loss_fake.item())}
    # Return G's loss and the components of D's loss.
    return out
  if config['mode_seeking_loss']:
    return train_mode_seeing
  else:
    return train

def generate_submission(sample, config, experiment_name):
    print('generate submission...')
    image_num = 10000
    output_dir = f"{config['samples_root']}/{experiment_name}/submission"
    os.makedirs(output_dir, exist_ok=True)
    image_list = []
    cnt = 0
    with torch.no_grad():
        while cnt < image_num:
            images, labels_val = sample()
            image_list += [images.data.cpu()]
            cnt += len(images)

    image_list = torch.cat(image_list, 0)[:image_num]
    for i,image in enumerate(image_list):
        image_fname = f'{output_dir}/{i}.png'
        image = transforms.ToPILImage()((image+1)/2)
        image = image.resize((64, 64), Image.ANTIALIAS)
        image.save(image_fname)

    import shutil
    shutil.make_archive('images', 'zip', output_dir)

    log_dir = f"{config['logs_root']}/{experiment_name}"
    if os.path.exists(log_dir):
        log_list = os.listdir(log_dir)
        for i in log_list:
            if i.count('loss') or i.count('metalog'):
                shutil.copy(f'{log_dir}/{i}', f'./')

    print('generate submission done')

  
''' This function takes in the model, saves the weights (multiple copies if 
    requested), and prepares sample sheets: one consisting of samples given
    a fixed noise seed (to show how the model evolves throughout training),
    a set of full conditional sample sheets, and a set of interp sheets. '''
def save_and_sample(G, D, G_ema, z_, y_, fixed_z, fixed_y, 
                    state_dict, config, experiment_name):
  utils.save_weights(G, D, state_dict, config['weights_root'],
                     experiment_name, None, G_ema if config['ema'] else None)
  # Save an additional copy to mitigate accidental corruption if process
  # is killed during a save (it's happened to me before -.-)
  if config['num_save_copies'] > 0:
    utils.save_weights(G, D, state_dict, config['weights_root'],
                       experiment_name,
                       'copy%d' %  state_dict['save_num'],
                       G_ema if config['ema'] else None)
    state_dict['save_num'] = (state_dict['save_num'] + 1 ) % config['num_save_copies']
    
  # Use EMA G for samples or non-EMA?
  which_G = G_ema if config['ema'] and config['use_ema'] else G
  
  # Accumulate standing statistics?
  if config['accumulate_stats']:
    utils.accumulate_standing_stats(G_ema if config['ema'] and config['use_ema'] else G,
                           z_, y_, config['n_classes'],
                           config['num_standing_accumulations'])
  
  # Save a random sample sheet with fixed z and y      
  with torch.no_grad():
    # Get Generator output given noise
    if config['use_dog_cnt'] and config['G_shared']:
      ye0 = G.shared(fixed_y[:, 0])
      ye1 = G.dog_cnt_shared(fixed_y[:, 1])
      gyc = torch.cat([ye0, ye1], 1)
    else:
      gyc = G.shared(fixed_y)

    if config['parallel']:
      fixed_Gz =  nn.parallel.data_parallel(which_G, (fixed_z,gyc))
    else:
      fixed_Gz = which_G(fixed_z, gyc)
  if not os.path.isdir('%s/%s' % (config['samples_root'], experiment_name)):
    os.mkdir('%s/%s' % (config['samples_root'], experiment_name))
  image_filename = '%s/%s/fixed_samples%d.jpg' % (config['samples_root'], 
                                                  experiment_name,
                                                  state_dict['itr'])
  torchvision.utils.save_image(fixed_Gz.float().cpu(), image_filename,
                             nrow=int(fixed_Gz.shape[0] **0.5), normalize=True)
  # # For now, every time we save, also save sample sheets
  num_classes = int(np.minimum(120, config['n_classes']))
  utils.sample_sheet(which_G,
                     classes_per_sheet=utils.classes_per_sheet_dict[config['dataset']],
                     num_classes=num_classes,
                     samples_per_class=10, parallel=config['parallel'],
                     samples_root=config['samples_root'],
                     experiment_name=experiment_name,
                     folder_number=state_dict['itr'],
                     z_=z_,
                     G_shared=config['G_shared'],
                     use_dog_cnt=config['use_dog_cnt'],
                     )
  if not config['use_dog_cnt']:
    # Also save interp sheets
    for fix_z, fix_y in zip([False, False, True], [False, True, False]):
      utils.interp_sheet(which_G,
                         num_per_sheet=16,
                         num_midpoints=8,
                         num_classes=num_classes,
                         parallel=config['parallel'],
                         samples_root=config['samples_root'],
                         experiment_name=experiment_name,
                         folder_number=state_dict['itr'],
                         sheet_number=0,
                         fix_z=fix_z, fix_y=fix_y, device='cuda')


  
''' This function runs the inception metrics code, checks if the results
    are an improvement over the previous best (either in IS or FID, 
    user-specified), logs the results, and saves a best_ copy if it's an 
    improvement. '''
def test(G, D, G_ema, z_, y_, state_dict, config, sample, get_inception_metrics,
         experiment_name, test_log):
  print('Gathering inception metrics...')
  if config['accumulate_stats']:
    utils.accumulate_standing_stats(G_ema if config['ema'] and config['use_ema'] else G,
                           z_, y_, config['n_classes'],
                           config['num_standing_accumulations'])
  IS_mean, IS_std, FID = get_inception_metrics(sample, 
                                               config['num_inception_images'],
                                               num_splits=10)
  print('Itr %d: PYTORCH UNOFFICIAL Inception Score is %3.3f +/- %3.3f, PYTORCH UNOFFICIAL FID is %5.4f' % (state_dict['itr'], IS_mean, IS_std, FID))
  # If improved over previous best metric, save approrpiate copy
  if ((config['which_best'] == 'IS' and IS_mean > state_dict['best_IS'])
    or (config['which_best'] == 'FID' and FID < state_dict['best_FID'])):
    print('%s improved over previous best, saving checkpoint...' % config['which_best'])
    utils.save_weights(G, D, state_dict, config['weights_root'],
                       experiment_name, 'best%d' % state_dict['save_best_num'],
                       G_ema if config['ema'] else None)
    state_dict['save_best_num'] = (state_dict['save_best_num'] + 1 ) % config['num_best_copies']
  state_dict['best_IS'] = max(state_dict['best_IS'], IS_mean)
  state_dict['best_FID'] = min(state_dict['best_FID'], FID)
  # Log results to file
  test_log.log(itr=int(state_dict['itr']), IS_mean=float(IS_mean),
               IS_std=float(IS_std), FID=float(FID))