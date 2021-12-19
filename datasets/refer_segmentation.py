# Builder for visual grouding datasets
from .grounding_datasets import ReferSegDataset
from PIL import Image
import datasets.transforms as T
import torch

class RefCOCO(ReferSegDataset):
    def __init__(self, data_root, im_dir, seg_dir, split, transforms, version="refcoco_unc",
                 max_query_len=40, bert_model='bert-base-uncased'):
        super(RefCOCO, self).__init__(
            data_root=data_root,
            im_dir=im_dir,
            seg_dir=seg_dir,
            dataset=version,
            split=split,
            max_query_len=max_query_len,
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


def make_refer_seg_transforms(img_size=224 ,max_img_size=1333 ,test=False):
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


def build_refcoco_segmentation(
        split='train', 
        version='refcoco_unc',
        data_root="./data/refcoco/anns",
        im_dir="./data/refcoco/images/train2014",
        seg_dir="./data/refcoco/masks",
        img_size=224, 
        max_img_size=1333,
        bert_model='bert-base-uncased'
    ):
    '''
        'refcoco_unc'
        'refcoco+_unc'
        'refcocog_google'
        'refcocog_umd'
    '''
    istest = split != 'train'

    return RefCOCO(
        data_root=data_root,
        im_dir=im_dir,
        seg_dir=seg_dir,
        version=version,
        transforms=make_refer_seg_transforms(img_size, max_img_size, test=istest),
        split=split,
        bert_model=bert_model
    )

if __name__ == "__main__":
    # comment out normalize in make_refer_transforms when testing
    from PIL import Image, ImageDraw
    import numpy as np
    # flickr
    d_train = build_refcoco_segmentation(split='train')
    d_val = build_refcoco_segmentation(split='val')
    d_test = build_refcoco_segmentation(split='testA')
    print(f"flickr30k datasets have : {len(d_train)} Training samples")
    print(f"flickr30k datasets have : {len(d_val)} Val samples")
    print(f"flickr30k datasets have : {len(d_test)} Testing samples")
    for i in range(0, 200, 50):
        samples, target = d_train[i]
        img = samples['img']
        mask = target['masks']
        img1 = ImageDraw.Draw(img)
        img1.rectangle(target['boxes'][0].numpy().tolist(), outline='red')
        img.save(f"./exps/refcoco_train_sample{i}.jpg")
        
        print(mask.shape, mask.dtype)
        mask = mask.numpy().astype(np.uint8)[0] * 255
        print(mask)
        mask = Image.fromarray(mask)
        mask.save(f"./exps/refcoco_mask_sample{i}.jpg")