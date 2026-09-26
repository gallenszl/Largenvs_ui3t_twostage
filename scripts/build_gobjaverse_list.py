import os 
import glob
import tqdm

gobjaverse_root = '/root/mochu_ws/gobjaverse_ws/data'
obj_list = glob.glob('*/*/', root_dir=gobjaverse_root)

sort_by_int_1 = lambda x: int(x.split('/')[-2])

obj_list = sorted(obj_list, key=sort_by_int_1)

def check_ok(i):
    for view_idx in range(40):
        if not os.path.exists(f'{gobjaverse_root}/{i}/{view_idx:03d}.png'):
            print(f'missing {gobjaverse_root}/{i}/{view_idx:03d}.png')
            return False
        if not os.path.exists(f'{gobjaverse_root}/{i}/{view_idx:03d}_depth.png'):
            print(f'missing {gobjaverse_root}/{i}/{view_idx:03d}_depth.png')
            return False
    if not os.path.exists(f'{gobjaverse_root}/{i}/transforms.json'):
        print(f'missing {gobjaverse_root}/{i}/transforms.json')
        return False
    return True

safe_list = []
for i in tqdm.tqdm(obj_list):
    # print(i)
    if not check_ok(i):
        print(i)
    else:
        safe_list.append(i)
    # break

with open('data/gobjaverse.txt', 'w') as f:
    f.write('\n'.join(safe_list))