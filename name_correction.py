import os
import numpy as np


ann = os.listdir('annotations/')
img = os.listdir('images/')

path_ann = 'annotations/'
path_img = 'images/'
for current_name in ann:
	num = int(current_name[:-5])
	new_name = "%06d" % (num,) + '.png'
	# print new_name
	os.rename(path_ann+current_name, path_ann+new_name)
