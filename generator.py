import os.path
import io
import IPython.display
import numpy as np
import cv2
import PIL.Image
import argparse

import torch

from models.model_settings import MODEL_POOL
from models.pggan_generator import PGGANGenerator
from models.stylegan_generator import StyleGANGenerator
from utils.manipulator import linear_interpolate

import pika
import json

def parse_args():
  """Parses arguments."""
  parser = argparse.ArgumentParser(
      description='Arguments for image processing.')
  parser.add_argument('-n', '--num_img', type=int, default=1,
                      help='Number of generated images ')
  parser.add_argument('--seed', type=int, default=132,
                      help='Seed')

  return parser.parse_args()

def build_generator(model_name):
  """Builds the generator by model name."""
  gan_type = MODEL_POOL[model_name]['gan_type']
  if gan_type == 'pggan':
    generator = PGGANGenerator(model_name)
  elif gan_type == 'stylegan':
    generator = StyleGANGenerator(model_name)
  return generator



def sample_codes(generator, num, latent_space_type='Z', seed=0):
  np.random.seed(seed)
  codes = generator.easy_sample(num)
  if generator.gan_type == 'stylegan' and latent_space_type == 'W':
    codes = torch.from_numpy(codes).type(torch.FloatTensor).to(generator.run_device)
    codes = generator.get_value(generator.model.mapping(codes))
  return codes

args = parse_args()
def main():
  f = open('config.json',) 
  conf = json.load(f) 
  num_samples = args.num_img
  noise_seed = args.seed
  model_name = "stylegan_ffhq"
  latent_space_type = "W"

  generator = build_generator(model_name)


  ATTRS = ['age', 'eyeglasses', 'gender', 'pose', 'smile']
  boundaries = {}
  for i, attr_name in enumerate(ATTRS):
    boundary_name = f'{model_name}_{attr_name}'
    if generator.gan_type == 'stylegan' and latent_space_type == 'W':
      boundaries[attr_name] = np.load(os.path.join(conf["bound_dir"], f'{boundary_name}_w_boundary.npy'))
    else:
      boundaries[attr_name] = np.load(os.path.join(conf["bound_dir"], f'{boundary_name}_boundary.npy'))

  latent_codes = sample_codes(generator, num_samples, latent_space_type, noise_seed)
  if generator.gan_type == 'stylegan' and latent_space_type == 'W':
    synthesis_kwargs = {'latent_space_type': 'W'}
  else:
    synthesis_kwargs = {}
  images = generator.easy_synthesize(latent_codes, **synthesis_kwargs)['image']
  save_npy_dir = 'test.npy'
  with open(save_npy_dir, 'wb') as f:
      np.save(f, latent_codes)
  i = 0
  for img in images:
    i += 1
    save_path = os.path.join(conf["output_dir"], f'original_image_{i}.jpg')
    cv2.imwrite(save_path, img[:, :, ::-1])

if __name__ == '__main__':
  main()