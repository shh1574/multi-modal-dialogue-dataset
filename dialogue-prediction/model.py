import torch
import torch.nn as nn
import torch.nn.init
from torch.autograd import Variable
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
import torch.nn.functional as F

import misc.utils as utils
import torch.optim as optim

import neptune
# from misc.rewards import get_self_critical_reward, init_cider_scorer
from models import resnext101_32x8d
from torch import nn

from transformers import BertModel

def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())


def order_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
    return score


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0, measure=False, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        if measure == 'order':
            self.sim = order_sim
        else:
            self.sim = cosine_sim

        self.max_violation = max_violation

    def forward(self, im, s):
        # compute image-sentence score matrix
        scores = self.sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()


class VSRNwithBERT(nn.Module):
    """
    rkiros/uvs model
    """

    def __init__(self, opt, tokenizer):
        super(VSRNwithBERT, self).__init__()

        # Build Models
        self.device = torch.device('cuda:{}'.format(opt.gpu_id))
        self.tokenizer = tokenizer
        self.grad_clip = opt.grad_clip

        self.img_enc = resnext101_32x8d(True)
        
        self.txt_enc = BertModel.from_pretrained('bert-base-uncased')
        self.txt_linear = torch.nn.Linear(768, opt.embed_size)
        self.img_linear = torch.nn.Linear(opt.embed_size, opt.embed_size)
        
        self.crit = utils.LanguageModelCriterion()
        self.rl_crit = utils.RewardCriterion()


        # Loss and Optimizer
        self.criterion = ContrastiveLoss(margin=opt.margin,
                                         measure=opt.measure,
                                         max_violation=opt.max_violation)


    def calculate_caption_loss(self, fc_feats, labels, masks):
        torch.cuda.synchronize()
        labels = labels.cuda()
        masks = masks.cuda()

        seq_probs, _ = self.caption_model(fc_feats, labels, 'train')
        loss = self.crit(seq_probs, labels[:, 1:], masks[:, 1:])

        return loss

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()


    def forward_emb_img(self, images):
        images = Variable(images)
        if torch.cuda.is_available():
            images = images.cuda()
        img_emb, GCN_img_emb = self.img_enc(images)
        return img_emb, GCN_img_emb

    def forward_emb_cap(self, captions):
        captions = Variable(captions)
        if torch.cuda.is_available():
            captions = captions.cuda()
        cap_emb = self.txt_enc(captions)[1]
        cap_emb = self.linear(cap_emb)

        return cap_emb


    def forward_emb_context(self, contexts, lengths):

        if torch.cuda.is_available():
            contexts = contexts.cuda()

        tmp = self.txt_enc(contexts)
        full_emb = tmp[0]
        context_emb = tmp[1]

        full_emb = F.relu(self.txt_linear(full_emb))
        context_emb = F.relu(self.txt_linear(context_emb))

        del contexts

        return context_emb, full_emb


    def forward_emb(self, images, captions, lengths, volatile=False):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        #images = Variable(images, volatile=volatile)
        #captions = Variable(captions, volatile=volatile)
        images = Variable(images)
        captions = Variable(captions)
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()

        # Forward
        cap_emb = self.txt_enc(captions)[1]
        cap_emb = self.txt_linear(cap_emb)
        cap_emb = F.relu(cap_emb)
        
        img_emb =  self.img_enc(images)
        img_emb = F.relu(self.img_linear(img_emb))

        del images, captions

        return img_emb, cap_emb

        # img_emb, GCN_img_emd = self.img_enc(images)
        # img_emb = F.relu(self.img_linear(img_emb))
        # GCN_img_emd = F.relu(self.img_linear(GCN_img_emd))

        # return img_emb, cap_emb, GCN_img_emd

    def forward_loss(self, img_emb, cap_emb, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        if torch.cuda.is_available():
            img_emb = img_emb.cuda()
            cap_emb = cap_emb.cuda()
        loss = self.criterion(img_emb, cap_emb)
        # self.logger.update('Le', loss.data[0], img_emb.size(0))
        # self.logger.update('Le_retrieval', loss.item(), img_emb.size(0))
        return loss

    def finetune_emb(self, images, contexts, targets, context_lengths, target_lengths, context_labels, target_labels, context_masks, target_masks, *args):
        """One training step given images and captions.
        """
        self.Eiters += 1

        images = images.to(self.device)
        contexts = context.to(self.device)
        targets = targets.to(self.device)

        # compute the embeddings
        img_emb, target_emb, GCN_img_emb = self.forward_emb(images, targets)
        context_emb = self.forward_emb_context(contexts, context_lengths)

        # Add Context
        final_emb = self.fusion(torch.cat((img_emb, context_emb), dim=1))
        final_GCN_emb = self.fusion(torch.cat((GCN_img_emb, context_emb.unsqueeze(1).expand(-1, 36, -1))))
        # final_GCN_emb = torch.mul(GCN_img_emb, context_emb.unsqueeze(1))
        # final_emb = img_emb
        # final_GCN_emb = GCN_img_emb

        # calcualte captioning loss
        self.optimizer.zero_grad()

        caption_loss = self.calcualte_caption_loss(final_GCN_emb, target_labels, target_masks)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        retrieval_loss = self.forward_loss(final_emb, target_emb)

        loss = retrieval_loss + caption_loss
        
        # neptune.log_metric('retrieval_loss', retrieval_loss.item())
        # neptune.log_metric('caption_loss', caption_loss.item())

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm(self.params, self.grad_clip)
        self.optimizer.step()


class VSRNFinetune(nn.Module):
    def __init__(self, opt, tokenizer):
        super(VSRNFinetune, self).__init__()
        self.pretrained_model = VSRNwithBERT(opt, tokenizer)
        self.tokenizer = tokenizer
        self.device = torch.device('cuda:{}'.format(opt.gpu_id))
        self.grad_clip = opt.grad_clip
        self.no_context = opt.no_context
        self.no_image = opt.no_image

        # self.txt_linear = torch.nn.Linear(768, opt.embed_size)
        # self.img_linear = torch.nn.Linear(opt.embed_size, opt.embed_size)
        
        # self.fusion = torch.nn.Linear(opt.embed_size * 2, opt.embed_size)

        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=opt.embed_size, nhead=4)
        self.fusion = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # params = list(self.pretrained_model.parameters())
        params = list(self.fusion.parameters())
        params += list(self.pretrained_model.txt_enc.parameters())
        params += list(self.pretrained_model.txt_linear.parameters())
        # params += list(self.pretrained_model.img_enc.parameters())
        params += list(self.pretrained_model.img_linear.parameters())

        # params += list(self.pretrained_model.decoder.parameters())
        # params += list(self.pretrained_model.encoder.parameters())
        # params += list(self.pretrained_model.caption_model.parameters())
        
        self.params = params
        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)

        self.finetune_Eiters = 0
        self.dialog_Eiters = 0


    def train_start(self):
        """switch to train mode
        """
        self.pretrained_model.img_enc.train()
        self.pretrained_model.txt_enc.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.pretrained_model.img_enc.eval()
        self.pretrained_model.txt_enc.eval()

    def generate_target(self, GCN_embs):
        total_instance = GCN_embs.shape[0]
        batch_size = 64
        generated_str = []
        for i in range((total_instance // batch_size) + 1):
            target_GCN_embs = GCN_embs[i * batch_size : min((i + 1) * batch_size, total_instance)]
            target_GCN_embs = target_GCN_embs.cuda()
    
            _, pred = self.pretrained_model.caption_model(target_GCN_embs, mode='inference')
            
            del target_GCN_embs
            
            for p in pred:
                # print(p)
                tokens = self.tokenizer.convert_ids_to_tokens(p)
                s = self.tokenizer.convert_tokens_to_string(tokens)
                idx = s.find('<end>')
                if idx != -1:
                    generated_str.append(s[:idx])
                else:
                    generated_str.append(s)

        return generated_str


    def finetune_emb(self, images, contexts, targets, context_lengths, target_lengths, context_labels, target_labels, context_masks, target_masks, *args):
        """One training step given images and captions.
        """
        self.finetune_Eiters += 1
        # self.logger.update('Eit', self.Eiters)
        # self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        batch_size = images.shape[0]

        images = images.to(self.device)
        contexts = contexts.to(self.device)
        targets = targets.to(self.device)

        # compute the embeddings
        # img_emb, target_emb, GCN_img_emb = self.pretrained_model.forward_emb(images, targets, target_lengths)
        img_emb, target_emb = self.pretrained_model.forward_emb(images, targets, target_lengths)
        context_emb, _ = self.pretrained_model.forward_emb_context(contexts, context_lengths)

        if self.no_context:
            final_emb = img_emb
            # final_emb = self.fusion(img_emb.unsqueeze(1)).squeeze(1)
        elif self.no_image:
            final_emb = context_emb
            # final_emb = self.fusion(context_emb.unsqueeze(1)).squeeze(1)
            # final_GCN_emb = GCN_img_emb
        else:
            # final_emb = self.fusion(torch.cat((img_emb, context_emb), dim=1))
            
            # final_emb = torch.mul(img_emb, context_emb)
            
            final_emb = img_emb + context_emb
            
            # attention_output = self.fusion(torch.cat((img_emb.unsqueeze(1), context_emb.unsqueeze(1)), dim=1))
            # final_emb = attention_output[:, 0] + attention_output[:, 1]

            # final_emb = self.fusion(torch.cat((GCN_img_emb, context_emb.unsqueeze(1)), dim=1))[:,-1]

            # final_GCN_emb = self.fusion(torch.cat((GCN_img_emb, context_emb.unsqueeze(1).expand(-1, 36, -1)), dim=2))
        
        # final_GCN_emb = torch.mul(GCN_img_emb, context_emb.unsqueeze(1))

        # final_emb = img_emb
        # final_GCN_emb = GCN_img_emb

        self.optimizer.zero_grad()
        retrieval_loss = self.pretrained_model.forward_loss(final_emb, target_emb)

        loss = retrieval_loss
        # loss = retrieval_loss + caption_loss
        # loss = caption_loss
        
        # neptune.log_metric('retrieval_loss', retrieval_loss.item() / batch_size)
        # neptune.log_metric('finetune_caption_loss', self.finetune_Eiters, caption_loss.item())

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm(self.params, self.grad_clip)
        self.optimizer.step()