import os,pathlib
import time
import argparse
from joblib import delayed
from joblib import Parallel

parser = argparse.ArgumentParser('HTM')
parser.add_argument('--a', type=str, help='data dir')
parser.add_argument('--idx')
args = parser.parse_args()

sbs = [args.idx]
for subdir_index in sbs:
    if int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK']) == 0:
        os.system('pip install ffmpeg-python --user')
        to_split = f'{args.a}/annotations/id2path_part_{subdir_index}.csv'
        with open(to_split) as f:
            lines = f.readlines()[1:]
        length = len(lines) // 8 + 1
        pathlib.Path(f'{args.a}/annotations/ttmp').mkdir(parents=True, exist_ok=True)
        for i in range(8):
            with open(f'{args.a}/annotations/ttmp/id2path_part_{subdir_index}_{i}.csv', 'w') as f:
                f.write('video_id,video_path,feature_path\n')
                idx_begin = length * i
                idx_end = min(length * (i+1), len(lines))
                for _ in range(idx_begin, idx_end):
                    f.write(lines[_])

    else:
        time.sleep(120)
    # def run(job_idx, offset, data_root):
    #     os.system(f'export CUDA_VISIBLE_DEVICES={job_idx}')
    #     i = job_idx + offset
    #     cmd = f'CUDA_VISIBLE_DEVICES={job_idx} python main.py --batch_size 96 --id2path {data_root}/annotations/id2path_part_{i}.csv --ann_file {data_root}/annotations/caption_{i}.json --data_root {data_root}'
    #     print(cmd)
    #     os.system(cmd)
    # Parallel(n_jobs=args.g)(delayed(run)(job_idx, args.o, args.a) for job_idx in range(args.g))
for subdir_index in sbs:
    os.system(f'python main.py --batch_size 96 --id2path {args.a}/annotations/ttmp/id2path_part_{subdir_index}_ --ann_file {args.a}/annotations/caption_{subdir_index}.json --data_root {args.a} --ITP')
