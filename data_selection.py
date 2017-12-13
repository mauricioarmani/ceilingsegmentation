import os
import numpy as np

folder_path = '/home/mauricio/rmi_final_project/FCN.tensorflow/logs/'

folder = os.listdir(folder_path+'dataset/')
files = np.array([], dtype='int32')

#SegmentationClass
#JPEGImages

for file_name in folder:
	file_current_path = folder_path+'dataset/'+file_name
	if file_name[-5] == "_":
		file_new_path = folder_path+'SegmentationClass/'+file_name[:-5]+'.png'
	else:
		file_new_path = folder_path+'JPEGImages/'+file_name
	# print file_current_path
	# print file_new_path
	# print " "
	os.rename(file_current_path, file_new_path)

# for file in x:
# 	num = int(file[:-4])
# 	files = np.append(files, num)

# data = sorted(np.random.choice(files, 200, replace=False))

# for num in data:
# 	image_name = "%06d" % (num,) + '.png'
# 	current_path = '/home/mauricio/rmi_final_project/dataset/'+image_name
# 	new_path = '/home/mauricio/rmi_final_project/dataset_200_3/'+image_name


