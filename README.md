# RefTR

Code for paper "Referring Transformer: A One-step Approach to Multi-task Visual Grounding"

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

data/annotations: https://drive.google.com/file/d/19qJ8b5sxijKmtN0XG9leWbt2sPkIVqlc/view?usp=sharing

refcoco/masks: https://drive.google.com/file/d/1oGUewiDtxjouT8Qp4dRzrPfGkc0LZaIT/view?usp=sharing

refcoco/anns: https://drive.google.com/file/d/1Prhrgm3t2JeY68Ni_1Ig_a4dfZvGC9vZ/view?usp=sharing

annotations_resc/vg/vg_all.pth: https://drive.google.com/file/d/1_GbWl0sSB1y26fFM9W7DDkXLRR8Ld3IH/view?usp=sharing

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

To train the model, run:
```train
# using slurm system
MASTER_PORT=${Master Port} GPUS_PER_NODE={GPU per node} ./tools/run_dist_slurm.sh RefTR ${Number Of GPU} ${config file name}
```

Example:
```python
MASTER_PORT=29501 GPUS_PER_NODE=4  ./tools/run_dist_slurm.sh  RefTR 4 configs/flickr30k/RefTR_flickr.sh 
```

## Evaluation

To evaluate the model, run:
```eval
MASTER_PORT=${Master Port} GPUS_PER_NODE={GPU per node} ./tools/run_dist_slurm.sh RefTR ${Number Of GPU} ${config file name} --eval --resume=${path to checkpoint}
```

Example:
```python
MASTER_PORT=29501 GPUS_PER_NODE=4  ./tools/run_dist_slurm.sh  RefTR 4 configs/flickr30k/RefTR_flickr.sh --eval --resume=./exps/flickr30k/checkpoint.pth
```

## Pretrained checkpoint for refcoco res/rec
| Checkpoint Name      | Dataset/Link | Description|
| ----------- | ----------- | --- |
| refcoco_SEG_PT_res50_6_epochs.pth  | [refcoco](https://drive.google.com/file/d/151XGTlGTbwGyQ6HMEn2sTEwEeFY9Csjx/view?usp=sharing) | Pretrained 6 epochs on VG |
| refcoco+_SEG_PT_res50_6_epochs.pth | [refcoco+](https://drive.google.com/file/d/1KKd80NReZJ500G6pnY1iRXoWqhJRDn5T/view?usp=sharing) | Pretrained 6 epochs on VG |
| refcocog_SEG_PT_res50_6_epochs.pth | [refcocog](https://drive.google.com/file/d/1oStrCvyJ2KyumXciMg6n8CdvefS9Qjsi/view?usp=sharing) | Pretrained 6 epochs on VG |

## Bibtext

If you find this code is useful for your research, please cite our paper

```
@inproceedings{muchen2021referring,
  title={Referring Transformer: A One-step Approach to Multi-task Visual Grounding},
  author={Muchen, Li and Leonid, Sigal},
  booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
  year={2021}
}
