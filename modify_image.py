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
  parser.add_argument('--age', type=float, default=0,
                      help='degree of modification age')
  parser.add_argument('--smile', type=float, default=0,
                      help='degree of modification smile')
  parser.add_argument('--eyeglasses', type=float, default=0,
                      help='degree of modification eyeglasses')
  parser.add_argument('--gender', type=float, default=0,
                      help='degree of modification gender')
  parser.add_argument('--pose', type=float, default=0,
                      help='degree of modification pose')

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
  """Samples latent codes randomly."""
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
  model_name = "stylegan_ffhq"
  latent_space_type = "W"

  generator = build_generator(model_name)


  ATTRS = ['age', 'eyeglasses', 'gender', 'pose', 'smile']
  boundaries = {}
  for i, attr_name in enumerate(ATTRS):
    boundary_name = f'{model_name}_{attr_name}'
    if generator.gan_type == 'stylegan' and latent_space_type == 'W':
      boundaries[attr_name] = np.load(f'boundaries/{boundary_name}_w_boundary.npy')
    else:
      boundaries[attr_name] = np.load(f'boundaries/{boundary_name}_boundary.npy')

  if generator.gan_type == 'stylegan' and latent_space_type == 'W':
    synthesis_kwargs = {'latent_space_type': 'W'}
  else:
    synthesis_kwargs = {}
  with open('test.npy', 'rb') as f:
      new_codes = np.load(f)
  age = args.age
  eyeglasses = args.eyeglasses
  gender = args.gender
  pose = args.pose
  smile = args.smile

  for i, attr_name in enumerate(ATTRS):
    new_codes += boundaries[attr_name] * eval(attr_name)
  new_images = generator.easy_synthesize(new_codes, **synthesis_kwargs)['image']
  i = 0
  for img in new_images:
    i += 1
    save_path = os.path.join(conf["output_dir"], f'result_{i}.jpg')
    cv2.imwrite(save_path, img[:, :, ::-1])

if __name__ == '__main__':
  main()