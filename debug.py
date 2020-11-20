with open('./id2path.txt', 'r') as f:
    lines = f.readlines()

import os
begin = 44
lines = lines[begin:48]
for idx, line in enumerate(lines):
    print(idx+begin)
    os.system('ls ~/data_new/{}'.format(line[:-1]))
