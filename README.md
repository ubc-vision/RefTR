# RefTR

Code for paper "Referring Transformer: A One-step Approach to Multi-task Visual Grounding"

TODO List:
- [x] Release full code
- [ ] Release pre-processed dataset annotations
- [ ] Release pre-trained Models

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

```
chmod +x tools/run_dist_slurm.sh 
```

## Setting up dataset
Flicker30k Entities: http://bryanplummer.com/Flickr30kEntities/

MSCOCO: http://mscoco.org/dataset/#overview

Visual Genome Images: https://visualgenome.org/api/v0/api_home.html

Extract dataset in the /data folder.(Tips: you can use softlinks to avoid putting data and code in the same directory.)
The data/ folder should look like this:
```
data
├── annotations
├── annotations_resc
│   ├── flickr
│   ├── gref
│   ├── gref_umd
│   ├── referit
│   ├── unc
│   ├── unc+
│   └── vg
├── flickr30k
│   └── f30k_images
├── refcoco
|   ├── anns
│   ├── images
|   │   ├──train2014  # images from train 2014
│   ├── masks
├── referit
│   ├── images
├── visualgenome
└───└──  VG_100K

```

## Training

To train the model(s) in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

## Pre-trained Models

##  Bibtext

If you find this code is useful for your research, please cite our paper

```
@inproceedings{muchen2021referring,
  title={Referring Transformer: A One-step Approach to Multi-task Visual Grounding},
  author={Muchen, Li and Leonid, Sigal},
  booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
  year={2021}
}