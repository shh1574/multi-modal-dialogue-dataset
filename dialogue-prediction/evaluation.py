from __future__ import print_function

import numpy
import time
import numpy as np
from collections import OrderedDict


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        """Log using tensorboard
        """
        for k, v in self.meters.items():
            tb_logger.log_value(prefix + k, v.val, step=step)


def encode_finetune_data(model, data_loader, no_context, no_image, log_step=10, logging=print):
    """Encode all images and captions loadable by `data_loader`
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    model.val_start()

    end = time.time()

    # numpy array to keep all the embeddings
    img_embs = None
    target_embs = None
    GCN_embs = None
    start = 0
    # val_loss = 0
    # ppl = 0

    for i, (images, contexts, targets, context_lengths, target_lengths, context_labels, target_labels, context_masks, target_masks) in enumerate(data_loader):
        # make sure val logger is used
        model.logger = val_logger
        ids = [i for i in range(start, start + images.shape[0])]
        start += images.shape[0]

        # compute the embeddings
        # img_emb, target_emb, GCN_img_emb = model.pretrained_model.forward_emb(images, targets, target_lengths, volatile=True)
        img_emb, target_emb = model.pretrained_model.forward_emb(images, targets, target_lengths, volatile=True)
        context_emb, _ = model.pretrained_model.forward_emb_context(contexts, context_lengths)

        # del images, contexts, targets, context_lengths, target_lengths, context_labels, target_labels, context_masks, target_masks

        if no_context:
            final_img_emb = img_emb
            # final_GCN_emb = GCN_img_emb
        elif no_image:
            final_img_emb = context_emb
        else:
            # final_img_emb = model.fusion(torch.cat((img_emb, context_emb), dim=1))
            
            # attention_output = model.fusion(torch.cat((img_emb.unsqueeze(1), context_emb.unsqueeze(1)), dim=1))
            # final_img_emb = attention_output[:, 0] + attention_output[:, 1]

            # final_img_emb = model.fusion(torch.cat((GCN_img_emb, context_emb.unsqueeze(1)), dim=1))[:,-1]
            
            # final_img_emb = torch.mul(img_emb, context_emb)
            
            final_img_emb = img_emb + context_emb

            # final_GCN_emb = model.fusion(torch.cat((GCN_img_emb, context_emb.unsqueeze(1).expand(-1, 36, -1)), dim=2))

        # initialize the numpy arrays given the size of the embeddings
        if img_embs is None:

            img_embs = np.zeros((len(data_loader.dataset), final_img_emb.size(1)))
            target_embs = np.zeros((len(data_loader.dataset), target_emb.size(1)))

        # preserve the embeddings by copying from gpu and converting to numpy
        img_embs[ids] = final_img_emb.data.cpu().numpy().copy()
        target_embs[ids] = target_emb.data.cpu().numpy().copy()

        del img_emb, target_emb, context_emb, final_img_emb

    return img_embs, target_embs


def i2t_finetune(images, captions, npts=None, measure='cosine', return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    npts = images.shape[0]
    index_list = []

    ranks = numpy.zeros(npts)
    top1 = numpy.zeros(npts)
    for index in range(npts):

        # Get query image
        im = images[index].reshape(1, images.shape[1])

        # Compute scores
        # if measure == 'order':
        #     bs = 100
        #     if index % bs == 0:
        #         mx = min(images.shape[0], 5 * (index + bs))
        #         im2 = images[5 * index:mx:5]
        #         d2 = order_sim(torch.Tensor(im2).cuda(),
        #                        torch.Tensor(captions).cuda())
        #         d2 = d2.cpu().numpy()
        #     d = d2[index % bs]
        # else:
        # if index + 100 > npts:
        #     cand_captions = np.concatenate((captions[index: npts], captions[:100 - npts + index]), axis=0)
        #     ids = [i for i in range(index, npts)] + [i for i in range(100 - npts + index)]
        # else:
        #     cand_captions = captions[index: index + 100]
        #     ids = [i for i in range(index, index + 100)]
                
        # d = numpy.dot(im, cand_captions.T).flatten()
        # inds = numpy.argsort(d)[::-1]
        # index_list.append(inds[0])

        # # Score
        # rank = numpy.where(inds == 0)[0][0]
        # ranks[index] = rank
        # # print(inds[0])
        # top1[index] = ids[inds[0]]


        d = numpy.dot(im, captions.T).flatten()
        inds = numpy.argsort(d)[::-1]
        index_list.append(inds[0])

        # Score
        rank = numpy.where(inds == index)[0][0]
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)
