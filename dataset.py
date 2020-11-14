import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import pandas as pd
import os, pathlib
import numpy as np
import ffmpeg
import json
# import jsonlines

class RandomSequenceSampler(Sampler):

    def __init__(self, n_sample, seq_len):
        self.n_sample = n_sample
        self.seq_len = seq_len

    def _pad_ind(self, ind):
        zeros = np.zeros(self.seq_len - self.n_sample % self.seq_len)
        ind = np.concatenate((ind, zeros))
        return ind

    def __iter__(self):
        idx = np.arange(self.n_sample)
        if self.n_sample % self.seq_len != 0:
            idx = self._pad_ind(idx)
        idx = np.reshape(idx, (-1, self.seq_len))
        np.random.shuffle(idx)
        idx = np.reshape(idx, (-1))
        return iter(idx.astype(int))

    def __len__(self):
        return self.n_sample + (self.seq_len - self.n_sample % self.seq_len)

class VideoClipDataset(Dataset):
    """Pytorch video & clip dataset."""

    def __init__(
            self,
            id2path,
            ann_file,
            data_root,
            framerate=16,
            size=256,
            centercrop=True,
    ):
        """
        Args:
        """
        self.data_root = data_root
        #jsonlines.open(ann_file) "VID": {"start": [list], "end": [list], "text": [list]}
        self.annotation = json.load(open(ann_file))
        self.id2path = pd.read_csv(id2path)
        self.centercrop = centercrop
        self.size = size
        self.framerate = framerate

    def __len__(self):
        return len(self.id2path)

    def _get_video_dim(self, video_path):
        probe = ffmpeg.probe(video_path)
        video_stream = next((stream for stream in probe['streams']
                             if stream['codec_type'] == 'video'), None)
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        return height, width

    def _get_output_dim(self, h, w):
        def toint2(num):
            # S3D need even width and height
            if int(num) % 2 == 1:
                return int(num) - 1
            else:
                return (int(num))
        if isinstance(self.size, tuple) and len(self.size) == 2:
            return self.size
        elif h >= w:
            if not self.centercrop:
                return toint2(h * self.size / w), self.size
            else:
                return int(h * self.size / w), self.size
        else:
            if not self.centercrop:
                return self.size, toint2(w * self.size / h)
            else:
                return self.size, int(w * self.size / h)

    def __getitem__(self, idx):
        video_id = self.id2path['video_id'].values[idx]
        video_path = os.path.join(self.data_root, self.id2path['video_path'].values[idx])
        output_file = os.path.join('/data/home/v-yixwe/100M', self.id2path['feature_path'].values[idx])
        pathlib.Path(output_file).mkdir(parents=True, exist_ok=True)
        caption_info = self.annotation[video_id]
        start_list = caption_info['start']
        end_list = caption_info['end']
        
        video_ext = ".webm"
        if os.path.isfile(os.path.join(video_path, video_id+video_ext)):
            video_path = os.path.join(video_path, video_id+video_ext)
        else:
            video_ext = ".mp4"
            video_path = os.path.join(video_path, video_id+video_ext)
        if os.path.isfile(video_path):
            # print('Decoding video: {}'.format(video_path))
            try:
                h, w = self._get_video_dim(video_path)
            except:
                print('ffprobe failed at: {}'.format(video_path))
                return {'video': torch.zeros(1), 'input': video_path}
            height, width = self._get_output_dim(h, w)
            cmd = (
                ffmpeg
                .input(video_path)
                .filter('fps', fps=self.framerate)
                .filter('scale', width, height)
            )
            if self.centercrop:
                x = int((width - self.size) / 2.0)
                y = int((height - self.size) / 2.0)
                cmd = cmd.crop(x, y, self.size, self.size)
            out, _ = (
                cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
                .run(capture_stdout=True, quiet=True)
            )
            # print('Finish Decoding video: {}'.format(video_path))
            if self.centercrop and isinstance(self.size, int):
                height, width = self.size, self.size

            raw_video = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
            raw_video = torch.from_numpy(raw_video.astype('float32'))
            raw_video = raw_video.permute(0, 3, 1, 2)
            # video = []
            # for _start, _end in zip(start_list, end_list):
            #     clip_out, _ = (
            #         ffmpeg
            #         .input('pipe:') #https://kkroening.github.io/ffmpeg-python/index.html?highlight=input#ffmpeg.input
            #         .trim(start =_start, end =_end) #https://kkroening.github.io/ffmpeg-python/index.html?highlight=trim#ffmpeg.trim
            #         .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            #         .run(capture_stdout=True, quiet=True)
            #     )
            #     clip_out = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
            #     clip_out = torch.from_numpy(clip_out.astype('float32'))
            #     clip_out = clip_out.permute(0, 3, 1, 2)
            #     video.append({'name': output_file + '/' + video_id + f'_{_start}_{_end}.npy', 'video': clip_out})
            clips = []
            for _start, _end in zip(start_list, end_list):
                _start_frame = int(_start * self.framerate)
                _end_frame = int(_end * self.framerate)
                clip_out = raw_video[ _start_frame : _end_frame,:,:,:]
                clips.append({'output_path': output_file + '/' + video_id + f'_{_start}_{_end}.npy', 'data': clip_out})
        else:
            video = torch.zeros(1)
            raise IOError('Invalid Video')
            
        return {'input': clips, 'input_path': video_path, 'output_path': output_file}