import json
import os
import shutil
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def convert_from_log(log_dir):
    if os.path.exists(f'{log_dir}/tb'):
        shutil.rmtree(f'{log_dir}/tb')
    tb_writter = SummaryWriter(log_dir=f'{log_dir}/tb')
    with open(f"{log_dir}/log.txt", 'r') as f:
        lines = f.readlines()
        for epoch, line in tqdm(enumerate(lines)):
            line = line.strip()
            if line == '':
                break
            info = json.loads(line)

            tb_writter.add_scalar('Loss/train', info['train_loss'], epoch)
            tb_writter.add_scalar('Loss_bbox/train', info['train_loss_bbox_unscaled'], epoch)
            tb_writter.add_scalar('Loss_ce/train', info['train_loss_ce_unscaled'], epoch)
            
            tb_writter.add_scalar('Loss/test', info['test_loss'], epoch)
            tb_writter.add_scalar('Loss_bbox/test', info['test_loss_bbox_unscaled'], epoch)
            tb_writter.add_scalar('Loss_ce/test', info['test_loss_ce_unscaled'], epoch)
            tb_writter.add_scalar('Accuracy/test', info['test_accuracy_iou0.5'], epoch)
            tb_writter.add_scalar('Miou/test', info['test_miou'], epoch)
    tb_writter.close()


if __name__ == '__main__':
    exp_path = './exps'
    for x in os.listdir(exp_path):
        if os.path.isdir(f'{exp_path}/{x}') and os.path.exists(f'{exp_path}/{x}/log.txt'):
            convert_from_log(f'{exp_path}/{x}')
