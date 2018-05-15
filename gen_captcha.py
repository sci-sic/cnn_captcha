# -*- coding:utf-8 -*-
import argparse
import json
import string
import os
import shutil
import uuid
from captcha.image import ImageCaptcha
import random

import itertools

FLAGS = None
META_FILENAME = 'meta.json'

random.seed(2018)

def get_choices():
    choices = [
        (FLAGS.digit, map(str, range(10))),
        (FLAGS.lower, string.ascii_lowercase),
        (FLAGS.upper, string.ascii_uppercase),
        ]
    return tuple([i for is_selected, subset in choices for i in subset if is_selected])


def _gen_captcha(img_dir_train,img_dir_test, num_per_image, n, width, height, choices):
    if os.path.exists(img_dir_train):
        shutil.rmtree(img_dir_train)
    if not os.path.exists(img_dir_train):
        os.makedirs(img_dir_train)

    if os.path.exists(img_dir_test):
        shutil.rmtree(img_dir_test)
    if not os.path.exists(img_dir_test):
        os.makedirs(img_dir_test)


    image = ImageCaptcha(width=width, height=height)

    print('generating %s epoches of captchas in %s' % (n, img_dir_train))
    for _ in range(n):
        check=0
        for captcha in [''.join(i) for i in itertools.permutations(choices, num_per_image) if random.random()>=0.985]:
            if random.random()<0.8:
                fn = os.path.join(img_dir_train, '%s_.png' % (captcha))
            else:
                fn = os.path.join(img_dir_test, '%s_.png' % (captcha))
            image.write(captcha, fn)
            check=check+1
            if check%1000 ==0:
                print(check)


def build_file_path(x):
    return os.path.join(FLAGS.data_dir, 'char-%s-epoch-%s' % (FLAGS.npi, FLAGS.n), x)


def gen_dataset():
    n_epoch = FLAGS.n
    num_per_image = FLAGS.npi
    test_ratio = FLAGS.t

    choices = get_choices()

    width = 330
    height = 150

    # meta info
    meta = {
        'num_per_image': num_per_image,
        'label_size': len(choices),
        'label_choices': ''.join(choices),
        'n_epoch': n_epoch,
        'width': width,
        'height': height,
    }

    print('%s choices: %s' % (len(choices), ''.join(choices) or None))

    _gen_captcha(build_file_path('train'),build_file_path('test'), num_per_image, n_epoch, width, height, choices=choices)
    # _gen_captcha(build_file_path('test'), num_per_image, max(1, int(n_epoch * test_ratio)), width, height, choices=choices)

    meta_filename = build_file_path(META_FILENAME)
    with open(meta_filename, 'w') as f:
        json.dump(meta, f, indent=4)
    print('write meta info in %s' % meta_filename)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-n',
        default=1,
        type=int,
        help='epoch number of character permutations.')

    parser.add_argument(
        '-t',
        default=0.2,
        type=float,
        help='ratio of test dataset.')

    parser.add_argument(
        '-d', '--digit',
        action='store_true',
        help='use digits in dataset.')
    parser.add_argument(
        '-l', '--lower',
        action='store_true',
        help='use lowercase in dataset.')
    parser.add_argument(
        '-u', '--upper',
        action='store_true',
        help='use uppercase in dataset.')
    parser.add_argument(
        '--npi',
        default=1,
        type=int,
        help='number of characters per image.')
    parser.add_argument(
        '--data_dir',
        default='./images',
        type=str,
        help='where data will be saved.')

    FLAGS, unparsed = parser.parse_known_args()

    gen_dataset()