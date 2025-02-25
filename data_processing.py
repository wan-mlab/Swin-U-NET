import numpy as np
import tensorflow as tf
from collections import defaultdict
import random
from sklearn import preprocessing
import os

def check_mask(mask, central_points, semi_block_size_output, semi_block_size_output2):

	current_shape = mask.shape

	control=0
	padding_dimensions=[]
	
	for _ in range(3):

		dim_list = []

		if central_points[_]-semi_block_size_output < 0:			
			dim_list.append(np.abs(central_points[_]-semi_block_size_output))
			control+=1

		else:

			dim_list.append(0)

		if central_points[_]+semi_block_size_output2 > current_shape[_]:
			dim_list.append(np.abs(central_points[_]+semi_block_size_output2 - current_shape[_]))
			control+=1
		else:
			dim_list.append(0)

		padding_dimensions.append(tuple(dim_list))

	if control > 0:

		padding_dimensions = tuple(padding_dimensions)
		mask = np.pad(mask, padding_dimensions, mode='constant', constant_values = 0.0)	
		central_points = [central_points[_]+padding_dimensions[_][0] for _ in range(3)]


	plm = mask[central_points[0]-semi_block_size_output:central_points[0]+semi_block_size_output2,
		central_points[1]-semi_block_size_output:central_points[1]+semi_block_size_output2,
		central_points[2]-semi_block_size_output:central_points[2]+semi_block_size_output2]

	return plm


def one_hot_encoder(input,dim_output,list_values):

	dictionar=defaultdict()
	for value,control in zip(list_values,np.arange(dim_output)):
		dictionar[value] = control

	object = np.zeros(shape=(input.shape[0],dim_output))
	for i in range(input.shape[0]):

		object[i,dictionar[int(input[i,0])]] = 1.0

	return object	


def blockshaped(arr, nrows, ncols):

	############################################################    
	#### Used to get non-overlapping patches from an image #####
	############################################################

    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))


def unblockshaped(arr, h, w):
	
	"""
	Return an array of shape (h, w) where
	h * w = arr.size

	If arr is of shape (n, nrows, ncols), n sublocks of shape (nrows, ncols),
	then the returned array preserves the "physical" layout of the sublocks.
	"""
	
	n, nrows, ncols = arr.shape
	return (arr.reshape(h//nrows, -1, nrows, ncols)
		.swapaxes(1,2)
		.reshape(h, w))


'''
### this is extracting random patches of equal size in raw input modality and output segmentation ####
### this is an old function which is not used anymore ################################################

def extract_patches_segmentation(inputul, outputul, mask):

	## inputul -- shape (num_batch, width, height, 3)
	## outputul -- shape (num_batch, width, height, 1)
	list_of_random_seeds = random.sample(range(1, 9999), outputul.shape[0])

	list_patches_input = []
	list_patches_segmentation = []
	list_patches_mask = []

	for _ in range(outputul.shape[0]):

		##### segmentation patches ###
		patches_segmentation = np.expand_dims(extract_patches_2d(image=outputul[_,...,0],patch_size=(100,100),max_patches=0.00001,random_state=list_of_random_seeds[_]),axis=-1)
		patches_mask = np.expand_dims(extract_patches_2d(image=mask[_,...,0],patch_size=(100,100),max_patches=0.00001,random_state=list_of_random_seeds[_]),axis=-1)

		##### input patches #######
		patches_1_dimension = np.expand_dims(extract_patches_2d(image=inputul[_,...,0],patch_size=(100,100),max_patches=0.00001,random_state=list_of_random_seeds[_]),axis=-1)
		patches_2_dimension = np.expand_dims(extract_patches_2d(image=inputul[_,...,1],patch_size=(100,100),max_patches=0.00001,random_state=list_of_random_seeds[_]),axis=-1)
		patches_3_dimension = np.expand_dims(extract_patches_2d(image=inputul[_,...,2],patch_size=(100,100),max_patches=0.00001,random_state=list_of_random_seeds[_]),axis=-1)
		patches_input = np.concatenate((patches_1_dimension,patches_2_dimension,patches_3_dimension),axis=-1)

		list_patches_input.append(patches_input)
		list_patches_segmentation.append(patches_segmentation)
		list_patches_mask.append(patches_mask)

	list_patches_input = np.concatenate(list_patches_input, axis=0)
	list_patches_segmentation = np.concatenate(list_patches_segmentation, axis=0)
	list_patches_mask = np.concatenate(list_patches_mask, axis=0)

	return list_patches_input, list_patches_segmentation, list_patches_mask
'''

def output_transformation(inputul):

	inputul = np.round(inputul)

	return inputul

def apply_mask(array, mask):

	array[mask < 1.0] = 0.0

	return array

def resize_by_axis(image, dim_1, dim_2, ax):

	### ax -- has to be one of 1,2,3 ####
	### ax cannot be 0 ####

	#### image -- shape (num_batch, height, width, depth) ####

	resized_list = []
	#### unpack along ax ####
	unstack_img_depth_list = [x for x in tf.unstack(image, axis = ax)]

	#### resample over the two other dimensions ####
	for i in unstack_img_depth_list:
		resized_list.append(tf.expand_dims(tf.image.resize_images(images = i, size = (dim_1, dim_2),
			method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=False, preserve_aspect_ratio=False),axis=ax))
	
	#### pack back again along the ax #####
	stack_img = tf.concat(resized_list,axis=ax)
	print('inside the resizing hack')
	print(stack_img.get_shape().as_list())

	return stack_img

##################################################################
####### This is for splitting 3D objects into 3d cubes ###########
##################################################################

def cubify(arr, newshape):

	###############################################
	#### non-overlapping cubes from 3D block ######
	###############################################

    oldshape = np.array(arr.shape)
    repeats = (oldshape / newshape).astype(int)
    tmpshape = np.column_stack([repeats, newshape]).ravel()
    order = np.arange(len(tmpshape))
    order = np.concatenate([order[::2], order[1::2]])
    # newshape must divide oldshape evenly or else ValueError will be raised
    return arr.reshape(tmpshape).transpose(order).reshape(-1, *newshape)

def uncubify(arr, oldshape):

	###################################################
	#### gather small cubes into bigger 3D block ######
	###################################################

    N, newshape = arr.shape[0], arr.shape[1:]
    oldshape = np.array(oldshape)    
    repeats = (oldshape / newshape).astype(int)
    tmpshape = np.concatenate([repeats, newshape])
    order = np.arange(len(tmpshape)).reshape(2, -1).ravel(order='F')
    return arr.reshape(tmpshape).transpose(order).reshape(oldshape)

def crop_2D_block(image, central_points, semi_block_size1, semi_block_size2):

	#### basically crops a small 2D cube from a bigger 2D object ####

	### image -- shape (height, width, channels)
	### central_points -- (c1,c2)
	### semi_block_size -- (l1,l2)

	plm = image[central_points[0]-semi_block_size1:central_points[0]+semi_block_size2,
		central_points[1]-semi_block_size1:central_points[1]+semi_block_size2,:]
	#print(plm.shape)

	return plm

def crop_3D_block(image,central_points,semi_block_size1,semi_block_size2):

	#### basically crops a small 3D cube from a bigger 3D object

	### image -- shape (height, width, depth, channels)
	### central_points -- (c1,c2,c3)
	### semi_block_size -- (l1,l2,l3)

	plm = image[central_points[0]-semi_block_size1:central_points[0]+semi_block_size2,
		central_points[1]-semi_block_size1:central_points[1]+semi_block_size2,
		central_points[2]-semi_block_size1:central_points[2]+semi_block_size2,:]
	#print(plm.shape)

	return plm

def check_and_add_zero_padding(input_image, output_image, central_points, semi_block_size1, semi_block_size2):

	#### checks if extracting a block need padding or not 
	#### accounts for the case where the central_points are close to the boundary of the brain scan and expands it with 0s

	### image -- shape (height, width, depth, channels)
	### central_points -- (c1,c2,c3)
	### semi_block_size -- (l1,l2,l3)

	current_shape = input_image.shape 
	min_value_image = np.min(input_image)
	padding_dimensions = []
	control=0				

	for _ in range(3):

		dim_list = []

		if central_points[_]-semi_block_size1 < 0:			
			dim_list.append(np.abs(central_points[_]-semi_block_size1))
			control+=1

		else:

			dim_list.append(0)

		if central_points[_]+semi_block_size2 > current_shape[_]:
			dim_list.append(np.abs(central_points[_]+semi_block_size2 - current_shape[_]))
			control+=1
		else:
			dim_list.append(0)

		padding_dimensions.append(tuple(dim_list))

	if control > 0:

		padding_dimensions = tuple(padding_dimensions)
		padding_dimensions_extra = list(padding_dimensions)
		padding_dimensions_extra.append(tuple([0,0]))
		padding_dimensions_extra = tuple(padding_dimensions_extra)
		#print(padding_dimensions_extra)
		print('size of input image')
		print(input_image.shape)

		input_image = np.pad(input_image, padding_dimensions_extra, mode='constant', constant_values = min_value_image)
		print('size of output image')
		print(output_image.shape)

		output_image = np.pad(output_image, padding_dimensions_extra, mode='constant')
		central_points = [central_points[_]+padding_dimensions[_][0] for _ in range(3)]

	return input_image, output_image, central_points
	
def check_and_add_zero_padding_regression(input_image, central_points, semi_block_size1, semi_block_size2):

	#### checks if extracting a block need padding or not 
	#### accounts for the case where the central_points are close to the boundary of the brain scan and expands it with 0s

	### image -- shape (height, width, depth, channels)
	### central_points -- (c1,c2,c3)
	### semi_block_size -- (l1,l2,l3)

	current_shape = input_image.shape 
	min_value_image = np.min(input_image)
	padding_dimensions = []
	control=0				

	for _ in range(3):

		dim_list = []

		if central_points[_]-semi_block_size1 < 0:			
			dim_list.append(np.abs(central_points[_]-semi_block_size1))
			control+=1

		else:

			dim_list.append(0)

		if central_points[_]+semi_block_size2 > current_shape[_]:
			dim_list.append(np.abs(central_points[_]+semi_block_size2 - current_shape[_]))
			control+=1
		else:
			dim_list.append(0)

		padding_dimensions.append(tuple(dim_list))

	if control > 0:

		padding_dimensions = tuple(padding_dimensions)
		padding_dimensions_extra = list(padding_dimensions)
		padding_dimensions_extra.append(tuple([0,0]))
		padding_dimensions_extra = tuple(padding_dimensions_extra)
		input_image = np.pad(input_image, padding_dimensions_extra, mode='constant', constant_values = min_value_image)	
		central_points = [central_points[_]+padding_dimensions[_][0] for _ in range(3)]

	return input_image, central_points



def check_and_add_zero_padding_2d_image(input_image, output_image, central_points, semi_block_size1, semi_block_size2):

	#### checks if extracting a patch need padding or not 
	#### accounts for the case where the central_points are close to the boundary of the image and expands it with the minimum of the image

	### image -- shape (height, width, channels)
	### central_points -- (c1,c2)
	### semi_block_size -- (l1,l2)

	current_shape = input_image.shape 
	min_value_image = np.min(input_image)
	padding_dimensions = []
	control=0				

	for _ in range(2):

		dim_list = []

		if central_points[_]-semi_block_size1 < 0:			
			dim_list.append(np.abs(central_points[_]-semi_block_size1))
			control+=1

		else:

			dim_list.append(0)

		if central_points[_]+semi_block_size2 > current_shape[_]:
			dim_list.append(np.abs(central_points[_]+semi_block_size2 - current_shape[_]))
			control+=1
		else:
			dim_list.append(0)

		padding_dimensions.append(tuple(dim_list))


	if control > 0:

		padding_dimensions = tuple(padding_dimensions)
		padding_dimensions_extra = list(padding_dimensions)
		padding_dimensions_extra.append(tuple([0,0]))
		padding_dimensions_extra = tuple(padding_dimensions_extra)
		#print(padding_dimensions_extra)
		input_image = np.pad(input_image, padding_dimensions_extra, mode='constant', constant_values = min_value_image)
		output_image = np.pad(output_image, padding_dimensions_extra, mode='constant')
		central_points = [central_points[_]+padding_dimensions[_][0] for _ in range(2)]

	return input_image, output_image, central_points

def extract_2d_blocks_training(inputul, outputul, iteration, block_size_input, block_size_output, dim_output):

	## inputul -- shape (num_batch, width, height, num_imaging_modalities)
	## outputul -- shape (num_batch, width, height, num_imaging_modalitie)

	#### this will extract 4 training examples ####

	lista = np.arange(inputul.shape[0])
	np.random.seed(iteration)
	np.random.shuffle(lista)
	current_index = lista[:2]
	semi_block_size_input = int(block_size_input//2)
	semi_block_size_input2 = block_size_input - semi_block_size_input
	semi_block_size_output = int(block_size_output//2)
	semi_block_size_output2 = block_size_output - semi_block_size_output
	list_blocks_input = []
	list_blocks_segmentation = []
	
	for _ in current_index:

		##### iterating over 2D images #####
		### pad current input and output scan to avoid problems ####

		current_input = inputul[_,...]
		current_output = outputul[_,...]

		#### shape of current scan ####
		current_shape = inputul[_,...].shape

		#################################################################################################################
		#### random places being extracted -- most likely not containing any segmentation besides background class ######
		#################################################################################################################

		list_of_random_places1 = random.sample(range(semi_block_size_output, current_shape[0]-semi_block_size_output2), 2)
		list_of_random_places2 = random.sample(range(semi_block_size_output, current_shape[1]-semi_block_size_output2), 2)

		for __ in range(2):
			
			#### iterate over the 2 locations of the 3D cubes #####
			central_points = [list_of_random_places1[__], list_of_random_places2[__]]

			current_input_padded, current_output_padded, central_points = check_and_add_zero_padding_2d_image(current_input, 
				current_output, central_points, semi_block_size_input, semi_block_size_input2)

			list_blocks_segmentation.append(crop_2D_block(current_output_padded, central_points, semi_block_size_output, semi_block_size_output2))
			list_blocks_input.append(crop_2D_block(current_input_padded, central_points, semi_block_size_input, semi_block_size_input2))

		###############################################################################################
		##### specifically extract 2D patches with a non-background class #############################
		###############################################################################################

		#########################
		##### Class number 1 ####
		#########################

		indices_tumor = np.where(current_output[...,0] == 1.0)
		indices_tumor_dim1 = indices_tumor[0]
		indices_tumor_dim2 = indices_tumor[1]

		if len(indices_tumor_dim1)==0:

			print('tumor not found')

		else:
					
			list_of_random_places = random.sample(range(0,len(indices_tumor_dim1)), 2)

			for __ in range(2):

				central_points = [indices_tumor_dim1[list_of_random_places[__]],
					indices_tumor_dim2[list_of_random_places[__]]]
				
				current_input_padded, current_output_padded, central_points = check_and_add_zero_padding_2d_image(current_input,
					current_output, central_points, semi_block_size_input, semi_block_size_input2)

				list_blocks_segmentation.append(crop_2D_block(current_output_padded, central_points, semi_block_size_output,semi_block_size_output2))
				list_blocks_input.append(crop_2D_block(current_input_padded, central_points, semi_block_size_input,semi_block_size_input2))

	list_blocks_input = np.stack(list_blocks_input)
	list_blocks_segmentation = np.stack(list_blocks_segmentation)

	shape_of_seg = list_blocks_segmentation.shape
	list_blocks_segmentation = list_blocks_segmentation.reshape((-1,1))
	#list_blocks_segmentation = output_transformation(list_blocks_segmentation)
	#enc = preprocessing.OneHotEncoder()
	#enc.fit(list_blocks_segmentation)
	#list_blocks_segmentation = enc.transform(list_blocks_segmentation).toarray()
	#list_blocks_segmentation = list_blocks_segmentation.reshape((-1,1))
	list_blocks_segmentation = OneHotEncoder(list_blocks_segmentation)
	list_blocks_segmentation = list_blocks_segmentation.reshape((shape_of_seg[0],shape_of_seg[1],shape_of_seg[2],dim_output))

	return list_blocks_input, list_blocks_segmentation


def extract_3d_blocks_training(inputul, outputul, iteration, block_size_input, block_size_output, dim_output):

	### size of brain_scan (240,240,155)
	## inputul -- shape (num_batch, width, height, depth, num_imaging_modalities)
	## outputul -- shape (num_batch, width, height, depth, num_imaging_modalitie)

	#### this wil extract 4 training examples ######


	lista = np.arange(len(inputul.keys()))
	np.random.seed(iteration)
	np.random.shuffle(lista)
	current_index = lista[:3]
	semi_block_size_input = int(block_size_input//2)
	semi_block_size_input2 = block_size_input - semi_block_size_input
	semi_block_size_output = int(block_size_output//2)
	semi_block_size_output2 = block_size_output - semi_block_size_output
	list_blocks_input = []
	list_blocks_segmentation = []
	
	for _ in current_index:

		##### iterating over brain scans #####

		### pad current input and output scan to avoid problems ####
		current_input = inputul[_]
		current_output = outputul[_]

		#### shape of current scan ####
		current_shape = inputul[_].shape

		##################################################################################
		#### random places being extracted -- most likely not containing any tumors ######
		##################################################################################

		list_of_random_places1 = random.sample(range(semi_block_size_output, current_shape[0]-semi_block_size_output2), 2)
		list_of_random_places2 = random.sample(range(semi_block_size_output, current_shape[1]-semi_block_size_output2), 2)
		list_of_random_places3 = random.sample(range(semi_block_size_output, current_shape[2]-semi_block_size_output2), 2)

		for __ in range(2):
			
			#### iterate over the 2 locations of the 3D cubes #####
			central_points = [list_of_random_places1[__], list_of_random_places2[__], list_of_random_places3[__]]

			current_input_padded, current_output_padded, central_points = check_and_add_zero_padding(current_input,
				current_output, central_points, semi_block_size_input, semi_block_size_input2)

			list_blocks_segmentation.append(crop_3D_block(current_output_padded, central_points, semi_block_size_output,semi_block_size_output2))
			list_blocks_input.append(crop_3D_block(current_input_padded, central_points, semi_block_size_input,semi_block_size_input2))

		####################################################################################
		##### specifically extract 3D blocks with tumor tissue #############################
		####################################################################################

		####################################################################################
		##### iterate over the remaining classes with tumors ###############################
		####################################################################################
		list_clasess = [1,2,4]
		for class_num in list_clasess:

			print('we are at class no. '+str(class_num))

			indices_tumor = np.where(current_output[...,0] == class_num)
			indices_tumor_dim1 = indices_tumor[0]
			indices_tumor_dim2 = indices_tumor[1]
			indices_tumor_dim3 = indices_tumor[2]

			if len(indices_tumor_dim1)==0:

				print('tumor not found')

			else:
						
				list_of_random_places = random.sample(range(0,len(indices_tumor_dim1)), 2)

				for __ in range(2):

					central_points = [indices_tumor_dim1[list_of_random_places[__]],
						indices_tumor_dim2[list_of_random_places[__]], indices_tumor_dim3[list_of_random_places[__]]]
					
					current_input_padded, current_output_padded, central_points = check_and_add_zero_padding(current_input,
						current_output, central_points, semi_block_size_input, semi_block_size_input2)

					list_blocks_segmentation.append(crop_3D_block(current_output_padded, central_points, semi_block_size_output,
						semi_block_size_output2))
					print(list_blocks_segmentation[-1].shape)
					list_blocks_input.append(crop_3D_block(current_input_padded, central_points, semi_block_size_input,
						semi_block_size_input2))
					print(list_blocks_input[-1].shape)

	list_blocks_input = np.stack(list_blocks_input)
	list_blocks_segmentation = np.stack(list_blocks_segmentation)

	shape_of_seg = list_blocks_segmentation.shape
	list_blocks_segmentation = list_blocks_segmentation.reshape((-1,1))
	#list_blocks_segmentation = output_transformation(list_blocks_segmentation)
	#enc = preprocessing.OneHotEncoder(categories=[0.0,1.0,2.0,4.0])
	#enc.fit(list_blocks_segmentation)
	#list_blocks_segmentation = enc.transform(list_blocks_segmentation).toarray()
	#list_blocks_segmentation = list_blocks_segmentation.reshape((-1,1))
	list_blocks_segmentation = one_hot_encoder(input = list_blocks_segmentation,dim_output =dim_output,list_values=[0,1,2,4])
	list_blocks_segmentation = list_blocks_segmentation.reshape((shape_of_seg[0],shape_of_seg[1],shape_of_seg[2],shape_of_seg[3],dim_output))
	#print(list_blocks_segmentation.shape)
	#print(list_blocks_input.shape)

	return list_blocks_input, list_blocks_segmentation

def extract_3d_blocks_training_regression(inputul, outputul, iteration, block_size_input, block_size_output, mask,
	num_subjects, num_voxels_per_subject, gender):

	### gender(num_batch, 1)
	### size of brain_scan (121, 145, 121)
	### mask -- (121, 145, 121) -- if using both GM and WM
	### inputul -- shape (num_batch, width, height, depth, num_imaging_modalities)
	### outputul -- shape (num_batch, 1)
	### current_shape = mask.shape 

	lista = np.arange(len(inputul.keys()))
	np.random.seed(iteration)
	np.random.shuffle(lista)
	current_index = lista[:num_subjects]
	semi_block_size_input = int(block_size_input//2)
	semi_block_size_input2 = block_size_input - semi_block_size_input
	semi_block_size_output = int(block_size_output//2)
	semi_block_size_output2 = block_size_output - semi_block_size_output
	list_blocks_input = []
	list_blocks_segmentation = []
	list_block_masks = []

	for _ in current_index:

		##### iterating over brain scans #####
		### pad current input and output scan to avoid problems ####
		current_input = inputul[_]
		current_output = outputul[_]
		current_gender = gender[_]

		#### shape of current scan ####
		current_shape = inputul[_].shape

		##################################################################################
		#### random places being extracted -- most likely not containing any tumors ######
		##################################################################################

		#list_of_random_places1 = random.sample(range(semi_block_size_output, current_shape[0]-semi_block_size_output2), 2)
		#list_of_random_places2 = random.sample(range(semi_block_size_output, current_shape[1]-semi_block_size_output2), 2)
		#list_of_random_places3 = random.sample(range(semi_block_size_output, current_shape[2]-semi_block_size_output2), 2)

		indices_tumor = np.where(mask[...] > 0.0)
		indices_tumor_dim1 = indices_tumor[0]
		indices_tumor_dim2 = indices_tumor[1]
		indices_tumor_dim3 = indices_tumor[2]
				
		list_of_random_places = random.sample(range(0, len(indices_tumor_dim1)), num_voxels_per_subject)

		for __ in range(num_voxels_per_subject):

			central_points = [indices_tumor_dim1[list_of_random_places[__]],
				indices_tumor_dim2[list_of_random_places[__]], indices_tumor_dim3[list_of_random_places[__]]]
			print(central_points)

			plm = check_mask(mask, central_points, semi_block_size_output, semi_block_size_output2)
			print(plm.shape)	
			current_input_padded, central_points = check_and_add_zero_padding_regression(current_input,central_points,
				semi_block_size_input, semi_block_size_input2)
			list_blocks_segmentation.append(np.ones((block_size_output,block_size_output,block_size_output,1))*current_output)
			cropped_input_block = crop_3D_block(current_input_padded, central_points, semi_block_size_input, semi_block_size_input2)
			print(cropped_input_block.shape)
			gender_3d_block = np.ones((block_size_input,
				block_size_input, block_size_input, 1)) * np.float(current_gender)
			cropped_input_block = np.concatenate((cropped_input_block,gender_3d_block),axis=-1)
			
			list_blocks_input.append(cropped_input_block)
			list_block_masks.append(plm > 0.0)
			
	list_blocks_input = np.stack(list_blocks_input)
	list_blocks_segmentation = np.stack(list_blocks_segmentation)
	list_block_masks = np.stack(list_block_masks)

	return list_blocks_input, list_blocks_segmentation, list_block_masks


def extract_3d_blocks_training_regression_roi(inputul, outputul, iteration, block_size_input, block_size_output, mask,
	num_subjects, num_voxels_per_subject, gender, indices_mni_structural):
	### gender(num_batch, 1)
	### size of brain_scan (121, 145, 121)
	### mask -- (121, 145, 121) -- if using both GM and WM
	### inputul -- shape (num_batch, width, height, depth, num_imaging_modalities)
	### outputul -- shape (num_batch, 1)
	### current_shape = mask.shape 

	lista = np.arange(len(inputul.keys()))
	np.random.seed(iteration)
	np.random.shuffle(lista)
	current_index = lista[:num_subjects]
	semi_block_size_input = int(block_size_input//2)
	semi_block_size_input2 = block_size_input - semi_block_size_input
	semi_block_size_output = int(block_size_output//2)
	semi_block_size_output2 = block_size_output - semi_block_size_output
	list_blocks_input = []
	list_blocks_segmentation = []
	list_block_masks = []

	for _ in current_index:

		##### iterating over brain scans #####
		### pad current input and output scan to avoid problems ####
		current_input = inputul[_]
		current_output = outputul[_]
		current_gender = gender[_]

		#### shape of current scan ####
		current_shape = inputul[_].shape

		#indices_tumor = np.where(mask[...] > 0.0
		indices_tumor_dim1 = indices_mni_structural[0]
		indices_tumor_dim2 = indices_mni_structural[1]
		indices_tumor_dim3 = indices_mni_structural[2]
				
		list_of_random_places = random.sample(range(0, len(indices_tumor_dim1)), num_voxels_per_subject)

		for __ in range(num_voxels_per_subject):

			central_points = [indices_tumor_dim1[list_of_random_places[__]],
				indices_tumor_dim2[list_of_random_places[__]], indices_tumor_dim3[list_of_random_places[__]]]
			print(central_points)

			plm = check_mask(mask, central_points, semi_block_size_output, semi_block_size_output2)
			print(plm.shape)	
			current_input_padded, central_points = check_and_add_zero_padding_regression(current_input,central_points,
				semi_block_size_input, semi_block_size_input2)
			list_blocks_segmentation.append(np.ones((block_size_output,block_size_output,block_size_output,1))*current_output)
			cropped_input_block = crop_3D_block(current_input_padded, central_points, semi_block_size_input, semi_block_size_input2)
			print(cropped_input_block.shape)
			gender_3d_block = np.ones((block_size_input,
				block_size_input, block_size_input, 1)) * np.float(current_gender)
			cropped_input_block = np.concatenate((cropped_input_block,gender_3d_block),axis=-1)
			
			list_blocks_input.append(cropped_input_block)
			list_block_masks.append(plm > 0.0)
			
	list_blocks_input = np.stack(list_blocks_input)
	list_blocks_segmentation = np.stack(list_blocks_segmentation)
	list_block_masks = np.stack(list_block_masks)

	return list_blocks_input, list_blocks_segmentation, list_block_masks



def dice_score_multiclass(predicted_labels, labels, num_classes, type_unet):

	#### Dice Score for at least 3 classes #####

	### predicted_labels -- shape (num_batch, height, width, depth, num_classes)
	### labels -- shape (num_batch, height, width, depth, num_classes)

	print('shape of predicted labels')
	print(predicted_labels)
	print('shape of actual labels')
	print(labels)

	shape_of_data = labels.get_shape().as_list()
	if type_unet=='3D':

		indices_predictions = tf.argmax(tf.reshape(predicted_labels, [-1 , shape_of_data[4]]),axis=-1)
		indices_predictions = tf.reshape(indices_predictions,[-1 , shape_of_data[1] * shape_of_data[2] * shape_of_data[3] * 1])		

		indices_labels = tf.argmax(tf.reshape(labels, [-1 , shape_of_data[4]]),axis=-1)
		indices_labels = tf.reshape(indices_labels,[-1 , shape_of_data[1] * shape_of_data[2] * shape_of_data[3] * 1])		
	else:

		indices_predictions = tf.argmax(tf.reshape(predicted_labels, [-1 , shape_of_data[3]]),axis=-1)
		indices_predictions = tf.reshape(indices_predictions,[-1 , shape_of_data[1] * shape_of_data[2] * 1])		

		indices_labels = tf.argmax(tf.reshape(labels, [-1 , shape_of_data[3]]),axis=-1)
		indices_labels = tf.reshape(indices_labels,[-1 , shape_of_data[1] * shape_of_data[2]  * 1])		

	print('after transformation')
	print(indices_predictions)
	print(indices_labels)

	dice_score = defaultdict()
	for _ in range(num_classes):

		shared_bool = tf.logical_and( tf.equal(tf.cast(indices_predictions,tf.float32),tf.ones_like(indices_predictions,dtype=tf.float32)* tf.cast(_,tf.float32)) ,
			tf.equal(tf.cast(indices_labels,tf.float32),tf.ones_like(indices_predictions,dtype=tf.float32)*tf.cast(_,tf.float32)))
		area_shared = tf.reduce_sum(tf.cast(shared_bool,tf.float32),1)

		predictions_bool = tf.equal(tf.cast(indices_predictions,tf.float32),tf.ones_like(indices_predictions,dtype=tf.float32)* tf.cast(_,tf.float32))
		area_predictions = tf.reduce_sum(tf.cast(predictions_bool,tf.float32),1)
		
		labels_bool = tf.equal(tf.cast(indices_labels,tf.float32),tf.ones_like(indices_predictions,dtype=tf.float32)* tf.cast(_,tf.float32))
		area_labels = tf.reduce_sum(tf.cast(labels_bool,tf.float32),1)

		dice_score[_] = tf.reduce_mean( (2.0 * area_shared + 1e-6) / (area_predictions + area_labels + 1e-6))

	return dice_score


def dice_score(predicted_labels, labels, dim_output, type_unet):

	####### Dice score for binary classification #######

	### predicted_labels -- shape (num_batch, height, width)
	### labels -- shape (num_batch, height, width)
	print('shape of predicted labels')
	print(predicted_labels)
	print('shape of actual labels')
	print(labels)

	shape_of_data = labels.get_shape().as_list()
	indices_predictions = tf.round(tf.reshape(predicted_labels, [-1 , dim_output]))
	if type_unet=='3D':

		indices_predictions = tf.reshape(indices_predictions,[-1 , shape_of_data[1] * shape_of_data[2] * shape_of_data[3] * 1])		
	else:
		indices_predictions = tf.reshape(indices_predictions,[-1 , shape_of_data[1] * shape_of_data[2]  * 1])		

	indices_labels = tf.round(tf.reshape(labels, [-1 , dim_output]))

	if type_unet=='3D':

		indices_labels = tf.reshape(indices_labels,[-1 , shape_of_data[1] * shape_of_data[2] * shape_of_data[3] * 1])		
	else:
		indices_labels = tf.reshape(indices_labels,[-1 , shape_of_data[1] * shape_of_data[2]  * 1])		

	print('after transofrmation')
	print(indices_predictions)
	print(indices_labels)

	dice_score = defaultdict()
	for _ in range(2):
		shared_bool = tf.logical_and( tf.equal(tf.cast(indices_predictions,tf.float32),tf.ones_like(indices_predictions,dtype=tf.float32)* tf.cast(_,tf.float32)) ,
			tf.equal(tf.cast(indices_labels,tf.float32),tf.ones_like(indices_predictions,dtype=tf.float32)*tf.cast(_,tf.float32)))
		area_shared = tf.reduce_sum(tf.cast(shared_bool,tf.float32),1)

		predictions_bool = tf.equal(tf.cast(indices_predictions,tf.float32),tf.ones_like(indices_predictions,dtype=tf.float32)* tf.cast(_,tf.float32))
		area_predictions = tf.reduce_sum(tf.cast(predictions_bool,tf.float32),1)
		
		labels_bool = tf.equal(tf.cast(indices_labels,tf.float32),tf.ones_like(indices_predictions,dtype=tf.float32)* tf.cast(_,tf.float32))
		area_labels = tf.reduce_sum(tf.cast(labels_bool,tf.float32),1)

		dice_score[_] = tf.reduce_mean( (2.0 * area_shared+1e-6) / (area_predictions + area_labels + 1e-6))

	return dice_score


######  Extract non-overlapping 3D cubes in segmentation space #############
###### also extracts the overlapping bigger 3D cubes in raw input space ####

def extract_3D_cubes_input_seg(input_image, output_image, semi_block_size_input1, semi_block_size_output1,
	semi_block_size_input2, semi_block_size_output2, dim_output):

	block_size_output = semi_block_size_output1 + semi_block_size_output2
	block_size_input = semi_block_size_input1 + semi_block_size_input2

	shape_of_input_data = input_image.shape

	num_cubes_dim1 = np.int(shape_of_input_data[0] // block_size_output)
	num_cubes_dim2 = np.int(shape_of_input_data[1] // block_size_output)
	num_cubes_dim3 = np.int(shape_of_input_data[2] // block_size_output)

	list_input_cubes = []
	list_output_cubes = []
	min_value_image = np.min(input_image)

	diff_semi_block1 = semi_block_size_input1 - semi_block_size_output1
	diff_semi_block2 = semi_block_size_input2 - semi_block_size_output2
	print('size of input image going in padding operation')
	print(input_image.shape)
	print(diff_semi_block1)
	print(diff_semi_block2)

	input_image_padded = np.pad(input_image, ((diff_semi_block1, diff_semi_block2),
		(diff_semi_block1, diff_semi_block2), (diff_semi_block1, diff_semi_block2),
		(0,0)), mode='constant', constant_values = min_value_image)

	for i in range(num_cubes_dim1):
		for j in range(num_cubes_dim2):
			for k in range(num_cubes_dim3):

				### extract segmentation space 3D cube ###
				list_output_cubes.append(output_image[block_size_output*i:block_size_output*(i+1),
					block_size_output*j:block_size_output*(j+1),
					block_size_output*k:block_size_output*(k+1)])
				print(list_output_cubes[-1].shape)

				### extract raw input space 3D cube ###
				list_input_cubes.append(input_image_padded[block_size_output*i:(block_size_output*i+block_size_input),
					block_size_output*j:(block_size_output*j+block_size_input),
					block_size_output*k:(block_size_output*k+block_size_input),:])
				print(list_input_cubes[-1].shape)

	list_output_cubes =  np.stack(list_output_cubes)
	shape_of_seg_output = list_output_cubes.shape
	list_output_cubes = output_transformation(list_output_cubes)
	list_output_cubes = list_output_cubes.reshape((-1,1))
	enc = preprocessing.OneHotEncoder()
	enc.fit(list_output_cubes)
	list_output_cubes = enc.transform(list_output_cubes).toarray()
	list_output_cubes =  list_output_cubes.reshape((shape_of_seg_output[0],shape_of_seg_output[1],shape_of_seg_output[2],shape_of_seg_output[3], dim_output))
	list_input_cubes = np.stack(list_input_cubes)

	return list_input_cubes, list_output_cubes


######  Extract non-overlapping 2D patches in segmentation space #############
###### also extracts the overlapping bigger 2D patches in raw input space ####

def extract_2D_cubes_input_seg(input_image, output_image, semi_block_size_input1, semi_block_size_output1,
	semi_block_size_input2, semi_block_size_output2, dim_output):

	#### input_image -- shape (height, width, num_raw_modalities)
	#### output_image -- shape (height, width)

	block_size_output = semi_block_size_output1 + semi_block_size_output2
	block_size_input = semi_block_size_input1 + semi_block_size_input2

	shape_of_input_data = input_image.shape

	num_cubes_dim1 = np.int(shape_of_input_data[0] // block_size_output)
	num_cubes_dim2 = np.int(shape_of_input_data[1] // block_size_output)

	list_input_cubes = []
	list_output_cubes = []
	min_value_image = np.min(input_image)

	diff_semi_block1 = semi_block_size_input1 - semi_block_size_output1
	diff_semi_block2 = semi_block_size_input2 - semi_block_size_output2
	print('size of input image going in padding operation')
	print(input_image.shape)
	print(diff_semi_block1)
	print(diff_semi_block2)

	input_image_padded = np.pad(input_image, ((diff_semi_block1, diff_semi_block2),
		(diff_semi_block1, diff_semi_block2), (0,0)), mode='constant', constant_values = min_value_image)

	for i in range(num_cubes_dim1):
		for j in range(num_cubes_dim2):

			### extract segmentation space 3D cube ###
			list_output_cubes.append(output_image[block_size_output*i:block_size_output*(i+1),
				block_size_output*j:block_size_output*(j+1)])
			print(list_output_cubes[-1].shape)

			### extract raw input space 3D cube ###
			list_input_cubes.append(input_image_padded[block_size_output*i:(block_size_output*i+block_size_input),
				block_size_output*j:(block_size_output*j+block_size_input),:])
			print(list_input_cubes[-1].shape)

	list_output_cubes =  np.stack(list_output_cubes)
	list_output_cubes = output_transformation(list_output_cubes)
	shape_of_seg_output = list_output_cubes.shape
	list_output_cubes = list_output_cubes.reshape((-1,1))
	enc = preprocessing.OneHotEncoder()
	enc.fit(list_output_cubes)
	list_output_cubes = enc.transform(list_output_cubes).toarray()
	list_output_cubes =  list_output_cubes.reshape((shape_of_seg_output[0],shape_of_seg_output[1],shape_of_seg_output[2], dim_output))	
	list_input_cubes = np.stack(list_input_cubes)

	return list_input_cubes, list_output_cubes


######  Extract non-overlapping 3D cubes in Regression space ###############
###### also extracts the overlapping bigger 3D cubes in raw input space ####

def extract_3D_cubes_input_seg_regression(input_image, output_image, semi_block_size_input1, semi_block_size_output1,
	semi_block_size_input2, semi_block_size_output2, dim_output):

	block_size_output = semi_block_size_output1 + semi_block_size_output2
	block_size_input = semi_block_size_input1 + semi_block_size_input2

	shape_of_input_data = input_image.shape

	num_cubes_dim1 = np.int(shape_of_input_data[0] // block_size_output)
	num_cubes_dim2 = np.int(shape_of_input_data[1] // block_size_output)
	num_cubes_dim3 = np.int(shape_of_input_data[2] // block_size_output)

	list_input_cubes = []
	list_output_cubes = []
	min_value_image = np.min(input_image)

	diff_semi_block1 = semi_block_size_input1 - semi_block_size_output1
	diff_semi_block2 = semi_block_size_input2 - semi_block_size_output2
	print('size of input image going in padding operation')
	print(input_image.shape)
	print(diff_semi_block1)
	print(diff_semi_block2)

	input_image_padded = np.pad(input_image, ((diff_semi_block1, diff_semi_block2),
		(diff_semi_block1, diff_semi_block2), (diff_semi_block1, diff_semi_block2),
		(0,0)), mode='constant', constant_values = min_value_image)

	for i in range(num_cubes_dim1):
		for j in range(num_cubes_dim2):
			for k in range(num_cubes_dim3):

				### extract segmentation space 3D cube ###
				list_output_cubes.append(np.ones((block_size_output,block_size_output,block_size_output,1))*output_image)
				print(list_output_cubes[-1].shape)

				### extract raw input space 3D cube ###
				list_input_cubes.append(input_image_padded[block_size_output*i:(block_size_output*i+block_size_input),
					block_size_output*j:(block_size_output*j+block_size_input),
					block_size_output*k:(block_size_output*k+block_size_input),:])
				print(list_input_cubes[-1].shape)

	list_output_cubes =  np.stack(list_output_cubes)
	list_input_cubes = np.stack(list_input_cubes)

	return list_input_cubes, list_output_cubes




def extract_3d_blocks_training_regression_roi_context_enhanced(inputul, outputul, iteration, block_size_input, block_size_output, mask,
	num_subjects, num_voxels_per_subject, gender, indices_mni_structural):
	### gender(num_batch, 1)
	### size of brain_scan (121, 145, 121)
	### mask -- (121, 145, 121) -- if using both GM and WM
	### inputul -- shape (num_batch, width, height, depth, num_imaging_modalities)
	### outputul -- shape (num_batch, 1)
	### current_shape = mask.shape 

	lista = np.arange(len(inputul.keys()))
	np.random.seed(iteration)
	np.random.shuffle(lista)
	current_index = lista[:num_subjects]
	semi_block_size_input = int(block_size_input//2)
	semi_block_size_input2 = block_size_input - semi_block_size_input
	semi_block_size_output = int(block_size_output//2)
	semi_block_size_output2 = block_size_output - semi_block_size_output
	list_blocks_input = []
	list_blocks_segmentation = []
	list_block_masks = []
	list_ages = []

	for _ in current_index:

		##### iterating over brain scans #####
		### pad current input and output scan to avoid problems ####
		current_input = inputul[_]
		current_output = outputul[_]
		current_gender = gender[_]

		#### shape of current scan ####
		current_shape = inputul[_].shape

		#indices_tumor = np.where(mask[...] > 0.0
		indices_tumor_dim1 = indices_mni_structural[0]
		indices_tumor_dim2 = indices_mni_structural[1]
		indices_tumor_dim3 = indices_mni_structural[2]
				
		list_of_random_places = random.sample(range(0, len(indices_tumor_dim1)), num_voxels_per_subject)

		for __ in range(num_voxels_per_subject):

			central_points = [indices_tumor_dim1[list_of_random_places[__]],
				indices_tumor_dim2[list_of_random_places[__]], indices_tumor_dim3[list_of_random_places[__]]]
			print(central_points)

			plm = check_mask(mask, central_points, semi_block_size_output, semi_block_size_output2)
			print(plm.shape)	
			current_input_padded, central_points = check_and_add_zero_padding_regression(current_input,central_points,
				semi_block_size_input, semi_block_size_input2)
			list_blocks_segmentation.append(np.ones((block_size_output,block_size_output,block_size_output,1))*current_output)
			cropped_input_block = crop_3D_block(current_input_padded, central_points, semi_block_size_input, semi_block_size_input2)
			print(cropped_input_block.shape)
			gender_3d_block = np.ones((block_size_input,
				block_size_input, block_size_input, 1)) * np.float(current_gender)
			cropped_input_block = np.concatenate((cropped_input_block,gender_3d_block),axis=-1)
			
			list_blocks_input.append(cropped_input_block)
			list_block_masks.append(plm > 0.0)
			list_ages.append(current_output)

	list_blocks_input = np.stack(list_blocks_input)
	list_blocks_segmentation = np.stack(list_blocks_segmentation)
	list_block_masks = np.stack(list_block_masks)
	list_ages = np.stack(list_ages)
	list_ages = np.reshape(list_ages,(-1,1))


	return list_blocks_input, list_blocks_segmentation, list_block_masks, list_ages

