import os
import time
import shutil
import neptune

import torch
from tqdm import tqdm
import numpy as np
import json

import data

from model import VSRNFinetune
from evaluation import AverageMeter, LogCollector, encode_finetune_data, i2t_finetune
import logging
import argparse

from transformers import BertTokenizer


def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='dataset',
                        help='path to datasets')
    parser.add_argument('--margin', default=0.2, type=float,
                        help='Rank loss margin.')
    parser.add_argument('--num_epochs', default=2, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--embed_size', default=2048, type=int,
                        help='Dimensionality of the joint embedding.')
    parser.add_argument('--grad_clip', default=2., type=float,
                        help='Gradient clipping threshold.')
    parser.add_argument('--learning_rate', default=5e-5, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--lr_update', default=15, type=int,
                        help='Number of epochs to update the learning rate.')
    parser.add_argument('--workers', default=10, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--log_step', default=10, type=int,
                        help='Number of steps to print and record the log.')
    parser.add_argument('--val_step', default=500, type=int,
                        help='Number of steps to run validation.')
    parser.add_argument('--save_step', default=2000, type=int,
                        help='Number of steps to run validation.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--max_violation', action='store_true',
                        help='Use max instead of sum in the rank loss.')
    parser.add_argument('--img_dim', default=2048, type=int,
                        help='Dimensionality of the image embedding.')
    parser.add_argument('--finetune', action='store_true',
                        help='Fine-tune the image encoder.')
    parser.add_argument('--measure', default='cosine',
                        help='Similarity measure used (cosine|order)')
    parser.add_argument('--use_abs', action='store_true',
                        help='Take the absolute value of embedding vectors.')
    parser.add_argument('--output_dir', default='output', type=str)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--gpu_id', type=int, required=True)
    parser.add_argument('--checkpoint_path', type=str)

    parser.add_argument(
        "--max_context_len",
        type=int,
        default=150,
        help='max length of context(containing <sos>,<eos>)')

    parser.add_argument(
        "--max_target_len",
        type=int,
        default=30,
        help='max length of target dialog(containing <sos>,<eos>)'
    )

    parser.add_argument(
        "--mode",
        type=str,
        default='finetune',
        help='finetune or inference'
    )

    parser.add_argument(
        "--task",
        type=str,
        default='current',
        help='current or next'
    )

    parser.add_argument(
        "--no_context",
        action="store_true"
    )
    parser.add_argument(
        "--no_image",
        action="store_true"
    )

    opt = parser.parse_args()
    print(opt)

    device = torch.device('cuda:{}'.format(opt.gpu_id))
    torch.cuda.set_device(device)

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    # Load Vocabulary Wrapper
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_tokens('<start>')
    tokenizer.add_tokens('<end>')
    print('Adding Special tokens : <start>, <end> : {}'.format(tokenizer.convert_tokens_to_ids(['<start>', '<end>'])))
    print('Vocab Size : ', tokenizer.vocab_size + 2)
    opt.vocab_size = tokenizer.vocab_size + 2

    model = VSRNFinetune(opt, tokenizer)
    model.to(device)

    if opt.mode == 'inference':
        checkpoint = torch.load(opt.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'])

        test_loader = data.get_test_loader(tokenizer, opt.batch_size, opt)

        test_rsum, test_info = validate(opt, test_loader, model, 0, opt.mode, split='test')
        return

    write_path = os.path.join(opt.output_dir, 'finetune_{}_{}'.format(opt.model_name, opt.task))

    if not os.path.exists(write_path):
        os.makedirs(write_path)
    else:
        res = input('Do you want to remove the directory : {} '.format(write_path))
        if res == 'y':
            shutil.rmtree(write_path)
            os.makedirs(write_path)
        else:
            return

    print('Making dataloader...')
    train_loader, val_loader, test_loader = data.get_loaders(tokenizer, opt.batch_size, opt)
    # return

    best_rsum = 0
    best_epoch = 0
    best_test_info = None

    with torch.no_grad():
        test_rsum, test_info = validate(opt, test_loader, model, 0, opt.mode, split='test')
        print(test_rsum, best_rsum)
        if test_rsum > best_rsum:
            best_rsum = test_rsum
            best_epoch = 0
            best_test_info = test_info
    
    for epoch in range(1, opt.num_epochs + 1):
        # adjust_learning_rate(opt, model.optimizer, epoch)
        train(train_loader, model, epoch)
        
        with torch.no_grad():
            # val_rsum, val_info = validate(opt, val_loader, model, epoch, mode='finetune', split='val')
            test_rsum, test_info = validate(opt, test_loader, model, epoch, mode=opt.mode, split='test')
            print(test_rsum, best_rsum)
            
            if test_rsum > best_rsum:
                best_rsum = test_rsum
                best_epoch = epoch
                best_test_info = test_info

                save_checkpoint({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'rsum': best_rsum,
                    'opt': opt,
                    'Eiters': model.finetune_Eiters,
                }, write_path, '{}_{}_{}.pth.tar'.format(opt.model_name, epoch, round(best_rsum, 2)))

        print(best_epoch, best_test_info)


def train(train_loader, model, epoch):
    # average meters to record the training statistics
    data_time = AverageMeter()
    train_logger = LogCollector()

    # switch to train mode
    model.train_start()

    end = time.time()
    for _, train_data in enumerate(tqdm(train_loader, desc='Epcoh {}'.format(epoch))):
        # if opt.reset_train:
            # Always reset to train mode, this is not the default behavior
        model.train_start()

        # measure data loading time
        data_time.update(time.time() - end)

        # make sure train logger is used
        model.logger = train_logger

        # Update the model
        model.finetune_emb(*train_data)


def validate(opt, val_loader, model, epoch, mode, split):
    print('Validate {} {}...'.format(mode, split))
    # compute the encoding for all the validation images and captions
    
    img_embs, cap_embs = encode_finetune_data(
        model, val_loader, opt.no_context, opt.no_image, opt.log_step, logging.info)

    # caption retrieval
    (r1, r5, r10, medr, meanr), (rank, top1) = i2t_finetune(img_embs, cap_embs, measure=opt.measure, return_ranks=True)
    logging.info("Image to text: %.2f, %.2f, %.2f, %.2f, %.2f" %
                 (r1, r5, r10, medr, meanr))

    if mode == 'inference':
        dataset = json.load(open(os.path.join(opt.data_path, 'MultiModalDialogue_test.json'), encoding='utf-8'))
        candidates = []
        for d in dataset:
            dialog = d['dialog']
            replaced_idx = d['replaced_idx']

            candidates.append(dialog[replaced_idx].strip())
        
        print(len(candidates))

        with open(os.path.join(opt.output_dir, 'wrong_ans.txt'), 'w') as f:
            for i, idx in enumerate(top1):
                if i != int(idx):
                    f.write('{} {}\n'.format(i, candidates[int(idx)]))

    # else:
    #     neptune.log_metric('{}_i2t_r1'.format(split), epoch, r1)
    #     neptune.log_metric('{}_i2t_r5'.format(split), epoch, r5)
    #     neptune.log_metric('{}_i2t_rsum'.format(split), epoch, r1+r5)
    #     neptune.log_metric('{}_i2t_meanr'.format(split), epoch, meanr)
    
    currscore = r1
    return currscore, (r1, r5, meanr)


def save_checkpoint(state, write_path, filename):
    torch.save(state, os.path.join(write_path, filename))
    # if is_best:
    #     shutil.copyfile(os.path.join(write_path, filename), os.path.join(write_path, 'model_best.pth.tar'))


def adjust_learning_rate(opt, optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every 30 epochs"""
    lr = opt.learning_rate * (0.1 ** (epoch // opt.lr_update))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()

