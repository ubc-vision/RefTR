# Builder for visual grouding datasets
from datasets.grounding_datasets.resc_refer_dataset import ReferDataset
from PIL import Image
import datasets.transforms as T
import torch

class GeneralReferDataset(torch.utils.data.Dataset):
    """
        A collection of datasets.
    """
    def __init__(self, datasets):
        super(GeneralReferDataset, self).__init__()
        self.datasets = datasets
        self.dataset_len = [len(x) for x in datasets]
    
    def __getitem__(self, idx):
        for i, dataset in enumerate(self.datasets):
            if idx >= self.dataset_len[i]:
                idx = idx - self.dataset_len[i]
            else:
                return dataset.__getitem__(idx)

    def __len__(self):
        return sum(self.dataset_len)

class flickr30k(ReferDataset):
    def __init__(self, data_root, im_dir, split, transforms,
                 max_query_len=40, lstm=False, bert_model='bert-base-uncased'):
        super(flickr30k, self).__init__(
            data_root=data_root,
            im_dir=im_dir,
            dataset='flickr',
            split=split,
            max_query_len=max_query_len,
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

class RefCOCO(ReferDataset):
    def __init__(self, data_root, im_dir, split, transforms, version="unc",
                 max_query_len=128, lstm=False, bert_model='bert-base-uncased'):
        super(RefCOCO, self).__init__(
            data_root=data_root,
            im_dir=im_dir,
            dataset=version,
            split=split,
            max_query_len=max_query_len,
            lstm=lstm,
            bert_model=bert_model
        )
        self._transforms = transforms
    
    def __getitem__(self, idx):
        input_sample, target = super(RefCOCO, self).__getitem__(idx)
        target = {k: torch.as_tensor(v) for k, v in target.items()}
        # target['boxes'] = torch.as_tensor(target['boxes'])
        img = Image.fromarray(input_sample["img"])
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        input_sample["img"] = img
        return input_sample, target


class ReferIt(ReferDataset):
    def __init__(self, data_root, im_dir, split, transforms, 
                 max_query_len=128, lstm=False, bert_model='bert-base-uncased'):
        super(ReferIt, self).__init__(
            data_root=data_root,
            im_dir=im_dir,
            dataset='referit',
            split=split,
            max_query_len=max_query_len,
            lstm=lstm,
            bert_model=bert_model
        )
        self._transforms = transforms
    
    def __getitem__(self, idx):
        input_sample, target = super(ReferIt, self).__getitem__(idx)
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


def build_flickr30k_resc(
        split='train',
        data_root="./data/annotations_resc",
        im_dir="./data/flickr30k/f30k_images",
        img_size=224,
        max_img_size=1333,
        max_query_len=40,
        bert_model='bert-base-uncased'):
    istest = not split in ['train', 'trainval'] 
    return flickr30k(
        data_root=data_root,
        im_dir=im_dir,
        transforms=make_refer_transforms(img_size, max_img_size, test=istest),
        split=split,
        max_query_len=max_query_len,
        bert_model=bert_model
    )

def build_referit_resc(
        split='train',
        data_root="./data/annotations_resc",
        im_dir="./data/referit/images",
        max_query_len=40,
        img_size=224, 
        max_img_size=1333,
        bert_model='bert-base-uncased'):
    istest = not split in ['train', 'trainval'] 
    return ReferIt(
        data_root=data_root,
        im_dir=im_dir,
        transforms=make_refer_transforms(img_size, max_img_size, test=istest),
        split=split,
        max_query_len=max_query_len,
        lstm=False,
        bert_model=bert_model
    )


def build_refcoco_resc(
        split='train',
        version='unc',
        data_root="./data/annotations_resc",
        im_dir="./data/refcoco/train2014",
        max_query_len=40,
        img_size=224,
        max_img_size=1333,
        bert_model='bert-base-uncased'):
    istest = not split in ['train', 'trainval'] 
    return RefCOCO(
        data_root=data_root,
        im_dir=im_dir,
        version=version,
        transforms=make_refer_transforms(img_size, max_img_size, test=istest),
        split=split,
        max_query_len=max_query_len,
        lstm=False,
        bert_model=bert_model
    )


def build_visualgenome(
        split='all',
        data_root="./data/annotations_resc",
        im_dir="./data/visualgenome/VG_100K",
        max_query_len=40,
        img_size=224,
        max_img_size=1333,
        bert_model='bert-base-uncased'):
    istest = False
    return RefCOCO(
        data_root=data_root,
        im_dir=im_dir,
        version='vg',
        transforms=make_refer_transforms(img_size, max_img_size, test=istest),
        split=split,
        max_query_len=max_query_len,
        lstm=False,
        bert_model=bert_model
    )

# def build_refer_collections():
#     flickr30k_d = build_flickr30k(split='train')
#     refcoco_d = build_refcoco(split='trainval', version='refcoco')
#     refcocop_d = build_refcoco(split='trainval', version='refcoco+')
#     referit = build_referit(split='trainval')
#     return GeneralReferDataset(datasets=[flickr30k_d, refcoco_d, refcocop_d, referit])

if __name__ == "__main__":
    # comment out normalize in make_refer_transforms when testing
    from PIL import Image, ImageDraw
    # flickr
    d_train = build_flickr30k(split='train')
    d_val = build_flickr30k(split='val')
    d_test = build_flickr30k(split='test')
    print(f"flickr30k datasets have : {len(d_train)} Training samples")
    print(f"flickr30k datasets have : {len(d_test)} Testing samples")
    for i in range(0, 200, 50):
        samples, target = d_train[i]
        img = samples['img']
        img1 = ImageDraw.Draw(img)
        img1.rectangle(target['boxes'][0].numpy().tolist(), outline='red')
        img.save(f"./exps/flickr_train_sample{i}.jpg")
    
    # refcoco
    d_train = build_refcoco(split='trainval', version='refcoco')
    d_testA = build_refcoco(split='testA', version='refcoco')
    d_testB = build_refcoco(split='testB', version='refcoco')
    print(f"Refcoco datasets have : {len(d_train)} Training samples")
    print(f"Refcoco datasets have : {len(d_testA)}/{len(d_testB)} Testing samples")
    for i in range(0, 200, 50):
        samples, target = d_train[i]
        img = samples['img']
        img1 = ImageDraw.Draw(img)
        img1.rectangle(target['boxes'][0].numpy().tolist(), outline='red')
        img.save(f"./exps/refcoco_train_sample{i}.jpg")

    # refcoco
    d_train = build_refcoco(split='trainval', version='refcoco+')
    d_testA = build_refcoco(split='testA', version='refcoco+')
    d_testB = build_refcoco(split='testB', version='refcoco+')
    print(f"Refcoco+ datasets have : {len(d_train)} Training samples")
    print(f"Refcoco+ datasets have : {len(d_testA)}/{len(d_testB)} Testing samples")
    for i in range(0, 200, 50):
        samples, target = d_train[i]
        img = samples['img']
        img1 = ImageDraw.Draw(img)
        img1.rectangle(target['boxes'][0].numpy().tolist(), outline='red')
        img.save(f"./exps/refcoco+_train_sample{i}.jpg")

    # referit
    d_train = build_referit(split='trainval')
    d_test = build_referit(split='test')
    print(f"ReferIt datasets have : {len(d_train)} Training samples")
    print(f"ReferIt datasets have : {len(d_test)} Testing samples")
    for i in range(0, 200, 50):
        samples, target = d_train[i]
        img = samples['img']
        img1 = ImageDraw.Draw(img)
        img1.rectangle(target['boxes'][0].numpy().tolist(), outline='red')
        img.save(f"./exps/referit_train_sample{i}.jpg")