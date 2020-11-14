import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import math
import numpy as np
import os, argparse
import time
import shutil
import subprocess

from s3dg import S3D
from dataset import VideoClipDataset, RandomSequenceSampler
from preprocessing import Preprocessing

parser = argparse.ArgumentParser(description='HowTo100M clip-level video feature extractor')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--id2path', type=str, help='csv file with an map from id to video path')
parser.add_argument('--ann_file', type=str, help='caption.json in HowTo100M')
parser.add_argument('--data_root', type=str, default='./', help='data root of all the relative path')
args = parser.parse_args()
# python main.py --batch_size 32 --id2path test_id2path.csv --ann_file ../annotation/caption.json --data_root ./

# Instantiate the model
# TODO-1: Multi-GPU
net = S3D('./s3d_dict.npy', 512) # text module
net = net.cuda()
net.load_state_dict(torch.load('./s3d_howto100m.pth')) # S3D

# Video input should be of size Batch x 3 x T x H x W and normalized to [0, 1] 
dataset = VideoClipDataset(
    args.id2path,
    args.ann_file,
    args.data_root,
    framerate=16,
    size=224,
    centercrop=True, # TODO: use ?*224 or ?*224 + centercrop or 224*224
)

n_dataset = len(dataset)
sampler = RandomSequenceSampler(n_dataset, 10)
loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=8,
    sampler=sampler if n_dataset > 10 else None,
)
preprocess = Preprocessing(framenum=16)

device_id = os.environ['CUDA_VISIBLE_DEVICES']
# Evaluation mode
net = net.eval()
 
with torch.no_grad():
    time_s = time.time()
    for k, data in enumerate(loader):
        input_file = data['input_path'][0]
        # Load
        clip_lengths = []
        output_files = []
        video = []
        for clip_meta in data['input']:
            output_file = clip_meta['output_path'][0]
            clip = clip_meta['data'].squeeze(0)
            # clip in fact. with Frames * 3 * H * W 
            if len(clip.shape) == 4:
                # Clip_Len * 3 * T * H * W // Clip_Len depends on how many Frames/framenum in this clip
                # TODO: ZeroPad to framenum in process
                _tmp = preprocess(clip)
                video.append(_tmp)
                clip_lengths.append(len(_tmp))
                output_files.append(output_file)
        # Inference
        # 'Video_Len' * 3 * T * H * W
        video = torch.cat(video, dim=0)
        n_chunk = len(video)
        features = torch.cuda.FloatTensor(n_chunk, 1024).fill_(0)
        n_iter = int(math.ceil(n_chunk / float(args.batch_size)))
        for i in range(n_iter):
            min_ind = i * args.batch_size
            max_ind = (i + 1) * args.batch_size
            video_batch = video[min_ind:max_ind].cuda()
            # batch_size * 1024
            batch_features = net(video_batch)['mixed_5c']
            if False:
                batch_features = F.normalize(batch_features, dim=1)
            features[min_ind:max_ind] = batch_features
        features = features.cpu().numpy()
        if False:
            features = features.astype('float16')
        # Save
        clip_end = 0
        for clip_idx, output_file in enumerate(output_files):
            clip_begin = clip_end
            clip_end = clip_begin + clip_lengths[clip_idx]
            np.save(output_file, features[ clip_begin : clip_end ])    
        print('{} Computing features of video {}/{}: {}, estimation: {}'.format(device_id, k + 1, n_dataset, input_file, (time.time() - time_s) * (n_dataset-k-1) / (k+1) / 3600))

        # Zip & remove.
        output_dir = data['output_path'][0]
        video_id = input_file.split('/')[-1].split('.')[0]
        cmd = f'zip -0 -q {output_dir}.zip {output_dir}/*'
        subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL)
        shutil.rmtree(output_dir)
# Text inference
# Text_output = net.text_module(['open door', 'cut tomato'])
