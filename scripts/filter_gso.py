import os 

gso_path = '/workspace/ssd1/gso_render_rv/'

gso_list = os.listdir(gso_path)

def is_ok(one):
    return os.path.exists(f'{gso_path}/{one}/transforms.json')

gso_list = [i for i in gso_list if is_ok(i)]

with open('data/gso_ours_rv.txt', 'w') as f:
    f.writelines('\n'.join(gso_list))