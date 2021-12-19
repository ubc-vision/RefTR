# Builder for visual grouding datasets
from .grounding_datasets import FlickrMultiPhraseDataset
from PIL import Image
import datasets.transforms as T
import torch

class flickr30k(FlickrMultiPhraseDataset):
    def __init__(self, data_root, im_dir, split, transforms,
                 max_seq_len=90, bert_model='bert-base-uncased', lstm=False):
        super(flickr30k, self).__init__(
            data_root=data_root,
            im_dir=im_dir,
            dataset='flickr',
            split=split,
            max_seq_len=max_seq_len,
            lstm=lstm,
            bert_model=bert_model
        )
        self._transforms = transforms
    
    def __getitem__(self, idx):
        input_sample, target = super(flickr30k, self).__getitem__(idx)
        target = {k: torch.as_tensor(v) for k, v in target.items()}
        # target['boxes'] = torch.as_tensor(target['boxes'])
        img = Image.fromarray(input_sample["img"])
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        input_sample["img"] = img
        return input_sample, target


def make_refer_transforms(img_size=224 ,max_img_size=1333 ,test=False):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if not test:
        return T.Compose([
            # T.RandomHorizontalFlip(),
            T.RandomIntensitySaturation(),
            T.RandomResize([img_size], max_size=max_img_size),
            # T.RandomAffineTransform(degrees=(-5,5), translate=(0.1, 0.1),
            #                         scale=(0.9, 1.1)),
            normalize
        ])
    else:
        return T.Compose([
            T.RandomResize([img_size], max_size=max_img_size),
            normalize
        ])


def build_flickr30k(split='train', 
                    data_root="./data/annotations",
                    im_dir="./data/flickr30k/f30k_images",
                    bert_model='bert-base-uncased',
                    img_size=224, 
                    max_img_size=1333):
    istest = split != 'train'
    return flickr30k(
        data_root=data_root,
        im_dir=im_dir,
        transforms=make_refer_transforms(img_size, max_img_size, test=istest),
        split=split,
        bert_model=bert_model
    )


if __name__ == "__main__":
    # comment out normalize in make_refer_transforms when testing
    from PIL import Image, ImageDraw
    # flickr
    d_train = build_flickr30k(split='train', bert_model="./configs/VinVL_VQA_base")
    d_val = build_flickr30k(split='val', bert_model="./configs/VinVL_VQA_base")
    d_test = build_flickr30k(split='test', bert_model="./configs/VinVL_VQA_base")
    print(f"flickr30k datasets have : {len(d_train)} Training samples")
    print(f"flickr30k datasets have : {len(d_test)} Testing samples")
    for i in range(0, 200, 50):
        samples, target = d_train[i]
        img = samples['img']
        img1 = ImageDraw.Draw(img)
        print(img)
        print(target['boxes'])
        img1.rectangle(target['boxes'][0].numpy().tolist(), outline='red')
        # if target['boxes'].shape[0] > 1:
        img1.rectangle(target['boxes'][1].numpy().tolist(), outline='blue')
        img.save(f"./exps/flickr_train_sample{i}.jpg")
    
    # # refcoco
    # d_train = build_refcoco(split='trainval', version='refcoco')
    # d_testA = build_refcoco(split='testA', version='refcoco')
    # d_testB = build_refcoco(split='testB', version='refcoco')
    # print(f"Refcoco datasets have : {len(d_train)} Training samples")
    # print(f"Refcoco datasets have : {len(d_testA)}/{len(d_testB)} Testing samples")
    # for i in range(0, 200, 50):
    #     samples, target = d_train[i]
    #     img = samples['img']
    #     img1 = ImageDraw.Draw(img)
    #     img1.rectangle(target['boxes'][0].numpy().tolist(), outline='red')
    #     img.save(f"./exps/refcoco_train_sample{i}.jpg")

    # # refcoco
    # d_train = build_refcoco(split='trainval', version='refcoco+')
    # d_testA = build_refcoco(split='testA', version='refcoco+')
    # d_testB = build_refcoco(split='testB', version='refcoco+')
    # print(f"Refcoco+ datasets have : {len(d_train)} Training samples")
    # print(f"Refcoco+ datasets have : {len(d_testA)}/{len(d_testB)} Testing samples")
    # for i in range(0, 200, 50):
    #     samples, target = d_train[i]
    #     img = samples['img']
    #     img1 = ImageDraw.Draw(img)
    #     img1.rectangle(target['boxes'][0].numpy().tolist(), outline='red')
    #     img.save(f"./exps/refcoco+_train_sample{i}.jpg")

    # # referit
    # d_train = build_referit(split='trainval')
    # d_test = build_referit(split='test')
    # print(f"ReferIt datasets have : {len(d_train)} Training samples")
    # print(f"ReferIt datasets have : {len(d_test)} Testing samples")
    # for i in range(0, 200, 50):
    #     samples, target = d_train[i]
    #     img = samples['img']
    #     img1 = ImageDraw.Draw(img)
    #     img1.rectangle(target['boxes'][0].numpy().tolist(), outline='red')
    #     img.save(f"./exps/referit_train_sample{i}.jpg")

# class GeneralReferDataset(torch.utils.data.Dataset):
#     """
#         A collection of datasets.
#     """
#     def __init__(self, datasets):
#         super(GeneralReferDataset, self).__init__()
#         self.datasets = datasets
#         self.dataset_len = [len(x) for x in datasets]
    
#     def __getitem__(self, idx):
#         for i, dataset in enumerate(self.datasets):
#             if idx >= self.dataset_len[i]:
#                 idx = idx - self.dataset_len[i]
#             else:
#                 return dataset.__getitem__(idx)

#     def __len__(self):
#         return sum(self.dataset_len)

# class RefCOCO(ReferDataset):
#     def __init__(self, data_root, im_dir, split, transforms, version="unc",
#                  max_query_len=128, lstm=False, bert_model='bert-base-uncased'):
#         super(RefCOCO, self).__init__(
#             data_root=data_root,
#             im_dir=im_dir,
#             dataset=version,
#             split=split,
#             max_query_len=max_query_len,
#             lstm=lstm,
#             bert_model=bert_model
#         )
#         self._transforms = transforms
    
#     def __getitem__(self, idx):
#         input_sample, target = super(RefCOCO, self).__getitem__(idx)
#         target = {k: torch.as_tensor(v) for k, v in target.items()}
#         # target['boxes'] = torch.as_tensor(target['boxes'])
#         img = Image.fromarray(input_sample["img"])
#         if self._transforms is not None:
#             img, target = self._transforms(img, target)
#         input_sample["img"] = img
#         return input_sample, target


# class ReferIt(ReferDataset):
#     def __init__(self, data_root, im_dir, split, transforms, 
#                  max_query_len=128, lstm=False, bert_model='bert-base-uncased'):
#         super(ReferIt, self).__init__(
#             data_root=data_root,
#             im_dir=im_dir,
#             dataset='referit',
#             split=split,
#             max_query_len=max_query_len,
#             lstm=lstm,
#             bert_model=bert_model
#         )
#         self._transforms = transforms
    
#     def __getitem__(self, idx):
#         input_sample, target = super(ReferIt, self).__getitem__(idx)
#         target = {k: torch.as_tensor(v) for k, v in target.items()}
#         # target['boxes'] = torch.as_tensor(target['boxes'])
#         img = Image.fromarray(input_sample["img"])
#         if self._transforms is not None:
#             img, target = self._transforms(img, target)
#         input_sample["img"] = img
#         return input_sample, target
# def build_referit(split='train',
#                   data_root="./data/annotations",
#                   im_dir="./data/referit/images"):
#     istest = split != 'train'
#     return ReferIt(
#         data_root=data_root,
#         im_dir=im_dir,
#         transforms=make_refer_transforms(test=istest),
#         split=split,
#         lstm=False
#     )


# def build_refcoco(split='train', 
#                   version='refcoco',
#                   data_root="./data/annotations",
#                   im_dir="./data/refcoco/train2014"):
#     istest = split != 'train'
#     if version == 'refcoco':
#         version = 'unc'
#     elif version == 'refcoco+':
#         version = 'unc+'
#     elif version == 'refcocog':
#         version = 'gref'
#     else:
#         raise NotImplementedError

#     return RefCOCO(
#         data_root=data_root,
#         im_dir=im_dir,
#         version=version,
#         transforms=make_refer_transforms(test=istest),
#         split=split,
#         lstm=False
#     )


# def build_refer_collections():
#     flickr30k_d = build_flickr30k(split='train')
#     refcoco_d = build_refcoco(split='trainval', version='refcoco')
#     refcocop_d = build_refcoco(split='trainval', version='refcoco+')
#     referit = build_referit(split='trainval')
#     return GeneralReferDataset(datasets=[flickr30k_d, refcoco_d, refcocop_d, referit])