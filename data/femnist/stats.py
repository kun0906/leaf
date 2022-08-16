import json
import os

in_dir = 'data/all_data_raw'

X = []
y = []
total = 0
n_users = 0  
n_imgs = 0 
for i, json_file in enumerate(os.listdir(in_dir)):
	with open(os.path.join(in_dir, json_file), 'rb') as f:
		vs = json.load(f)
	n1 = len(vs['users'])
	n_users += n1
	n2 = 0 
	for user in vs['users']:
		tmp = vs['user_data'][user]
		y = tmp['y']
		n2 += len(y)
	n_imgs += n2 
	print(i, json_file, n1, n2)

print(f"n_users: {n_users}\n")
print(f"n_imgs: {n_imgs}\n")

