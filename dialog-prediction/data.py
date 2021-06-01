import torch
import torch.utils.data as data
import numpy as np
import json as jsonmod

class FinetuneDataset(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f8k, f30k, coco, 10crop
    """

    def __init__(self, data_path, data_split, tokenizer, opt):
        self.tokenizer = tokenizer
        loc = data_path + '/'
        self.data_split = data_split

        task = opt.task
        
        self.max_context_len = opt.max_context_len
        self.max_target_len = opt.max_target_len

        # Context
        self.contexts = []
        self.targets = []
        self.threshold_img_idx = []

        json_idx = []
        with open(loc + 'MultiModalDialogue_{}.json'.format(data_split), 'r') as f:
            json_data = jsonmod.load(f)
            for i, instance in enumerate(json_data):
                replaced_idx = instance['replaced_idx']
                dialog = instance['dialog']

                if task == 'next':
                    if replaced_idx == 0 or len(dialog) == replaced_idx + 1:
                        continue
                    
                    replaced_idx += 1
                json_idx.append(i)
                target = dialog[replaced_idx]
                try:
                    context = ' [SEP] '.join(dialog[max(0, replaced_idx - 3) : replaced_idx])
                except UnicodeEncodeError:
                    continue

                self.contexts.append(context)
                self.targets.append(target)

        self.images = np.load(loc + 'MultiModalDialogue_{}_ims.npy'.format(data_split))
        self.images = self.images[json_idx]
        self.length = len(self.targets)

        print('Total Instances : ', self.length)
        print('Image shape : ', self.images.shape)


    def __getitem__(self, index):
        if self.data_split == 'train':
            image = torch.Tensor(self.images[self.threshold_img_idx[index]])
        else:
            image = torch.Tensor(self.images[index])
        context = self.contexts[index]
        target = self.targets[index]

        context_tokens = self.tokenizer.tokenize(context)
        

        target_tokens = self.tokenizer.tokenize(target)
        
        context_tensor = torch.Tensor(self.tokenizer.convert_tokens_to_ids(['[CLS]'] + context_tokens + ['[SEP]']))
        target_tensor = torch.Tensor(self.tokenizer.convert_tokens_to_ids(['[CLS]'] + target_tokens + ['[SEP]']))
    
        if len(context_tensor) > self.max_context_len:
            context_tensor = context_tensor[:self.max_context_len]

        if len(target_tensor) > self.max_target_len:
            target_tensor = target_tensor[:self.max_target_len]

        ##### deal with caption model data
        # label = np.zeros(self.max_len)
        c_mask = np.zeros(self.max_context_len + 1)
        context_gts = np.zeros((self.max_context_len + 1))
        cap_context = ['<start>'] + context_tokens + ['<end>']
        if len(cap_context) > self.max_context_len - 1:
            cap_context = cap_context[:self.max_context_len]
            cap_context[-1] = '<end>'

        context_label = self.tokenizer.convert_tokens_to_ids(cap_context)
        context_gts[:len(context_label)] = context_label

        context_non_zero = (context_gts == 0).nonzero()

        c_mask[:int(context_non_zero[0][0]) + 1] = 1

        context_label = torch.from_numpy(context_gts).type(torch.LongTensor)
        context_mask = torch.from_numpy(c_mask).type(torch.FloatTensor)

        t_mask = np.zeros(self.max_target_len + 1)
        target_gts = np.zeros((self.max_target_len + 1))

        cap_target = ['<start>'] + target_tokens + ['<end>']
        if len(cap_target) > self.max_target_len - 1:
            cap_target = cap_target[:self.max_target_len]
            cap_target[-1] = '<end>'

        target_label = self.tokenizer.convert_tokens_to_ids(cap_target)
        target_gts[:len(target_label)] = target_label

        target_non_zero = (target_gts == 0).nonzero()

        t_mask[:int(target_non_zero[0][0]) + 1] = 1

        target_label = torch.from_numpy(target_gts).type(torch.LongTensor)
        target_mask = torch.from_numpy(t_mask).type(torch.FloatTensor)

        return image, context_tensor, target_tensor, context_label, target_label, context_mask, target_mask

    def __len__(self):
        return self.length


def finetune_collate_fn(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption) tuple.
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length
    # data.sort(key=lambda x: len(x[2]), reverse=True)
    images, contexts, targets, context_labels, target_labels, context_masks, target_masks = zip(*data)
    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)

    context_labels_ = torch.stack(context_labels, 0)
    context_masks_ = torch.stack(context_masks, 0)

    target_labels_ = torch.stack(target_labels, 0)
    target_masks_ = torch.stack(target_masks, 0)

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    context_lengths = [len(cap) for cap in contexts]
    context_targets = torch.zeros(len(contexts), max(context_lengths)).long()
    for i, cap in enumerate(contexts):
        end = context_lengths[i]
        context_targets[i, :end] = cap[:end]

    target_lengths = [len(cap) for cap in targets]
    targets_ = torch.zeros(len(targets), max(target_lengths)).long()
    for i, cap in enumerate(targets):
        end = target_lengths[i]
        targets_[i, :end] = cap[:end]

    return images, context_targets, targets_, context_lengths, target_lengths, context_labels_, target_labels_, context_masks_, target_masks_


def get_precomp_loader(data_path, data_split, tokenizer, opt, batch_size=100, shuffle=True):
    dset = FinetuneDataset(data_path, data_split, tokenizer, opt)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                            batch_size=batch_size,
                                            shuffle=shuffle,
                                            pin_memory=True,
                                            collate_fn=finetune_collate_fn)

    return data_loader


def get_loaders(tokenizer, batch_size, opt):
    train_loader = get_precomp_loader(opt.data_path, 'train', tokenizer, opt, batch_size, True)
    val_loader = get_precomp_loader(opt.data_path, 'dev', tokenizer, opt, batch_size, False)

    test_loader = get_precomp_loader(opt.data_path, 'test', tokenizer, opt, batch_size, False)

    return train_loader, val_loader, test_loader


def get_test_loader(tokenizer, batch_size, opt):
    test_loader = get_precomp_loader(opt.data_path, 'test', tokenizer, opt,
                                    batch_size, False)

    return test_loader 