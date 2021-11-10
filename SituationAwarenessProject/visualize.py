import argparse
import json
import logging
import math
import nltk

import os
import random
import numpy as np
import torch
import torch.nn as nn

from PIL import Image
from pycocotools.coco import COCO
from termcolor import colored
from tqdm import tqdm

from torchvision import transforms
from torchvision.models import vgg19
from torch.serialization import default_restore_location

from captioner import models, utils
from captioner.data.dataset import CaptionDataset, BatchSampler
from captioner.data.dictionary import Dictionary
from captioner.generator import SequenceGenerator

import calculatebleu

def get_args():
    parser = argparse.ArgumentParser('Caption Generation')
    parser.add_argument('--seed', default=1000, type=int, help='pseudo random number generator seed')

    # Add data arguments
    parser.add_argument('--coco-path', default='your data path',help='path to COCO datasets')#
    parser.add_argument('--test-caption', default='test.json', help='reference captions')
    parser.add_argument('--test-image', default='your data path', help='path to test images')
    parser.add_argument('--caption-ids', type=int,nargs='+', help='caption ids')
    parser.add_argument('--image-size', type=int, default=256, help='size for resizing images')
    parser.add_argument('--crop_size', type=int, default=224, help='size for randomly cropping images')
    parser.add_argument('--checkpoint-path', default='checkpoint.pt', help='path to the model file')

    # Add generation arguments
    parser.add_argument('--beam-size', default=5, type=int, help='beam size')
    parser.add_argument('--max-len', default=200, type=int, help='maximum length of generated sequence')
    parser.add_argument('--stop-early', default='True', help='stop generation immediately after finalizing hypotheses')
    parser.add_argument('--normalize_scores', default='True', help='normalize scores by the length of the output')
    parser.add_argument('--len-penalty', default=1, type=float, help='length penalty: > 1.0 favors longer sentences')
    parser.add_argument('--unk-penalty', default=0, type=float, help='unknown word penalty: >0 produces fewer unks')
    return parser.parse_args()

import pandas as pd
import time
from nltk.translate.meteor_score import meteor_score

def main(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load arguments from checkpoint (no need to load pretrained embeddings or write to log file)
    state_dict = torch.load(args.checkpoint_path, map_location=lambda s, l: default_restore_location(s, 'cpu'))
    args = argparse.Namespace(**{**vars(state_dict['args']), **vars(args), 'embed_path': None, 'log_file': None})
    utils.init_logging(args)

    # Load dictionary
    dictionary = Dictionary.load(os.path.join(args.data, 'dict.txt'))
    logging.info('Loaded a dictionary of {} words'.format(len(dictionary)))

    # Load dataset
    coco = COCO(os.path.join(args.coco_path, args.test_caption))
    print("path:",os.path.join(args.coco_path, args.test_caption))
    if args.caption_ids is None:
        args.caption_ids = np.random.choice(list(coco.anns.keys()), 8, replace=False)
        # args.caption_ids = list(coco.anns.keys())[0:5]
        print('captions_id(not image_id):',args.caption_ids)
    image_ids = [coco.anns[id]['image_id'] for id in args.caption_ids]
    print("image_id:", image_ids)
    reference_captions = [coco.anns[id]['caption'] for id in args.caption_ids]
    image_names = [os.path.join(args.coco_path, args.test_image, coco.loadImgs(id)[0]['file_name']) for id in image_ids]

    # Transform image
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(0.0),
    ])
    images = [transform(Image.open(filename)) for filename in image_names]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    images_raw = [Image.open(filename) for filename in image_names]
    sample = torch.stack([transform(image.convert('RGB')) for image in images], dim=0)

    # Extract image features
    vgg = vgg19(pretrained=True).eval().cuda()
    model = nn.Sequential(*list(vgg.features.children())[:-2])
    image_features = model(utils.move_to_cuda(sample)) # [x,512,14,14]
    image_features = image_features.view(*image_features.size()[:-2], -1) # [x,512,196]
    # B x C x (H x W) -> B x (H x W) x C
    image_features = image_features.transpose(1, 2) # [x,196,512]

    # Load model and build generator
    model = models.build_model(args, dictionary).cuda()
    model.load_state_dict(state_dict['model'])
    logging.info('Loaded a model from checkpoint {}'.format(args.checkpoint_path))
    generator = SequenceGenerator(
        model, dictionary, beam_size=args.beam_size, maxlen=args.max_len, stop_early=eval(args.stop_early),
        normalize_scores=eval(args.normalize_scores), len_penalty=args.len_penalty, unk_penalty=args.unk_penalty,
    )

    # Generate captions
    with torch.no_grad():
        hypos = generator.generate(image_features)
    with open("data/annotations/captions_newtest2020.json") as fw:
        data = json.load(fw)
        anno_list = data["annotations"]
        length = len(images)
        candidate_list = []
        captions_truly_list = []
        for i, (id, image, image_raw, reference_caption) in enumerate(zip(args.caption_ids, images, images_raw, reference_captions)):
            output_image = os.path.join('D:\下载\\528SAProject\data\images', '{}.jpg'.format(id))#j for j in args.caption_ids
            attention = hypos[i][0]['attention'].view(14, 14, -1).cpu().numpy()
            system_tokens = [dictionary.words[tok] for tok in hypos[i][0]['tokens'] if tok != dictionary.eos_idx]
            p1 = ",".join(system_tokens)
            p1 = p1.replace(",", " ")
            p1 = p1[:-2]
            captions_truly = p1 + "."
            print(id)
            print(captions_truly)
            captions_truly_list.append(captions_truly)
            for anno in anno_list:
                if anno["id"] == id:
                    candidate = anno["caption"]
                    print(candidate)
                    candidate_list.append(candidate)


            utils.plot_image_finally(image_raw, captions_truly, output_image)
        # utils.plot_image_caption(image, output_image, system_tokens, reference_caption, attention) # 输出1 human + 3 machine
        # utils.plot_image(image, output_image, system_tokens, reference_caption, attention) #

def calculatemeteor(candidate_list, captions_truly_list, length):
    meteor = 0.0
    for candidate,captions_truly in zip(candidate_list, captions_truly_list):
        single_meteor = round(meteor_score([candidate], captions_truly, alpha=3, gamma=0.5), 4)
        if single_meteor > 1:
            pass
            length -= 1
        else:
            # print("single_meteor:", single_meteor)
            meteor += single_meteor
    return meteor/length

if __name__ == '__main__':
    args = get_args()
    main(args)
