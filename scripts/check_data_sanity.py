import os
from tqdm import tqdm

objaverse_root = '/workspace/ssd1/obj_40v/'

with open('data/objaverse_80k_40v.txt', 'r') as f:
    items = f.readlines()

items = [i.strip() for i in items]

complete_set = []

for i in tqdm(items):
    is_complete = True
    for v in range(40):
        if not os.path.exists(os.path.join(objaverse_root, i, f'{v:03d}.png')):
            is_complete = False
            # break
        if not os.path.exists(os.path.join(objaverse_root, i, f'{v:03d}_depth.png')):
            is_complete = False
            # break
    if not os.path.exists(os.path.join(objaverse_root, i, 'transforms.json')):
        is_complete = False
    if not is_complete:
        print(f'{i} is incomplete')

    if is_complete:
        complete_set.append(i)

print(len(complete_set))

with open('data/objaverse_80k_40v_.txt', 'w') as f:
    for item in complete_set:
        f.write(item + '\n')