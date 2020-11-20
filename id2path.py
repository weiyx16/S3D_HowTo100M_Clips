split_size = 128
with open('../howto100m_videos.txt', 'r') as f:

    lines = f.readlines()
    print(len(lines))
    length = int(len(lines) / 128)

count=0
with open('./id2path.txt', 'a') as f:
    f.write('video_id,video_path,feature_path\n')
    for i in range(split_size):
        idx_begin = length * i
        idx_end = min(length * (i+1), len(lines)) if i != split_size-1 else len(lines)
        path = 'howto100m_split_{:07}_{:07}'.format(idx_begin, idx_end)
        
        #f.write(f'{path}/S3D_HTM_Feature/'+'\n')
        for _ in range(idx_begin, idx_end):
            _id = lines[_].split('/')[-1].split('.')[0] # .mp4\n
            f.write(_id + ',' + path + ',' + f'{path}/S3D_HTM_Feature/' + '\n')
            count+=1
print(count)

with open('./id2path.csv', 'r') as f:
    lines = f.readlines()[1:]
    length = int(len(lines)) // split_size + 1
    for i in range(split_size):
        with open('./id2path_part_{}.csv'.format(i), 'w') as ff:
            ff.write('video_id,video_path,feature_path\n')
            idx_begin = length * i
            idx_end = min(length * (i+1), len(lines))
            for _ in range(idx_begin, idx_end):
                ff.write(lines[_])

import json
ann_file = '../annotation/caption.json'
print('Loading Json')
annotation = json.load(open(ann_file))
unfinded_ = []
for i in range(split_size):
    tmp_json = {}
    print(f'deal with Json {i}')
    with open('./id2path_part_{}.csv'.format(i), 'r') as f:
        lines = f.readlines()[1:]
        for _id in lines:
            _id = _id.split(',')[0]
            try:
                info = annotation[_id]   
            except KeyError:
                unfinded_.append(_id)
            else:
                tmp_json[_id] = info
    
    with open('./caption_{}.json'.format(i), 'w') as f:
        json.dump(tmp_json, f)

with open('./no-caption-video.txt', 'w') as f:
    for _id in unfinded_:
        f.write(_id + '\n')
