import torch
import os
import numpy as np
from datasets.crowd import Crowd
from models.model import CSRNet
import argparse
import matplotlib.pyplot as plt


args = None


def parse_args():
    parser = argparse.ArgumentParser(description='Generate images ')
    parser.add_argument('--data-dir', default='/home/matijamasaibb/codesmatijamasa/Bayesian-Crowd-Counting/data/ShanghaiA-processed/ShanghaiA-processed/',
                        help='training data directory')
    parser.add_argument('--save-dir', default='/home/matijamasaibb/codesmatijamasa/Bayesian-CSRNet/save/sha/1229-184744-best',
                        help='model directory')
    parser.add_argument('--device', default='0', help='assign device')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu

    datasets = Crowd(os.path.join(args.data_dir, 'test'), 512, 8, is_gray=False, method='val')
    dataloader = torch.utils.data.DataLoader(datasets, 1, shuffle=False,
                                             num_workers=8, pin_memory=False)
    model = CSRNet()
    device = torch.device('cuda')
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_model.pth'), device))
    epoch_minus = []

    for inputs, count, name in dataloader:
        if name[0] != "img_0022":
            continue
        inputs = inputs.to(device)
        assert inputs.size(0) == 1, 'the batch size should equal to 1'
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            #print(outputs.shape)
            print(inputs.size)
            print(inputs.shape)
            temp_minu = count[0].item() - torch.sum(outputs).item()
            print(name, temp_minu, count[0].item(), torch.sum(outputs).item())
            epoch_minus.append(temp_minu)
            gen_gt = outputs.cpu()
            reshaped = gen_gt.reshape(gen_gt.shape[2],gen_gt.shape[3])
            print(reshaped)
            plt.imshow(reshaped)
            plt.axis('off')
            plt.savefig(name[0] + '.png', bbox_inches='tight')


    epoch_minus = np.array(epoch_minus)
    mse = np.sqrt(np.mean(np.square(epoch_minus)))
    mae = np.mean(np.abs(epoch_minus))
    log_str = 'Final Test: mae {}, mse {}'.format(mae, mse)
    print(log_str)
