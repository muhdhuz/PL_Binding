import torch
import torch.utils.data as data
import numpy as np
import os
import os.path
import re
import random
import math


def read_pdb(filename):
	"""reads pdb file in training data, return corrds and atomtype feature as a numpy array"""
	with open(filename, 'r') as file:
		strline_L = file.readlines()
		# print(strline_L)

	X_list = list()
	Y_list = list()
	Z_list = list()
	atomtype_list = list()
	for strline in strline_L:
		# removes all whitespace at the start and end, including spaces, tabs, newlines and carriage returns
		stripped_line = strline.strip()

		line_length = len(stripped_line)
		# print("Line length:{}".format(line_length))
		if line_length < 78:
			print("ERROR: line length is different. Expected>=78, current={}".format(line_length))
		
		X_list.append(float(stripped_line[30:38].strip()))
		Y_list.append(float(stripped_line[38:46].strip()))
		Z_list.append(float(stripped_line[46:54].strip()))

		atomtype = stripped_line[76:78].strip()
		if atomtype == 'C':
			atomtype_list.append(1) # 'h' means hydrophobic
		else:
			atomtype_list.append(0) # 'p' means polar
	
	pdb = np.array([X_list, Y_list, Z_list, atomtype_list])
	return pdb
	
def read_pdb_test(filename):
	"""reads pdb file in test data, return corrds and atomtype feature as a numpy array"""
	with open(filename, 'r') as file:
		strline_L = file.readlines()
		# print(strline_L)

	X_list = list()
	Y_list = list()
	Z_list = list()
	atomtype_list = list()
	for strline in strline_L:
		# removes all whitespace at the start and end, including spaces, tabs, newlines and carriage returns
		stripped_line = strline.strip()
		# print(stripped_line)

		splitted_line = stripped_line.split('\t')
		
		X_list.append(float(splitted_line[0]))
		Y_list.append(float(splitted_line[1]))
		Z_list.append(float(splitted_line[2]))
		atomtype_list.append(str(splitted_line[3]))
		
	atomtype_list_int = [1 if atom is 'h' else 0 for atom in atomtype_list ]
	pdb = np.array([X_list, Y_list, Z_list, atomtype_list_int])
	return pdb

def listDirectory(directory, fileExtList,regex=None):										 
	"""returns a list of file info objects in directory that contains extension in the list fileExtList (include the . in your extension string)
	and regex if specified
	fileList - fullpath from working directory to files in directory
	fnameList - basenames in directory (including extension)
	regex - a substring in the filename, if unspecified will list all files in directory"""	
	if regex is not None:
		fnameList = [os.path.normcase(f)
				for f in os.listdir(directory)
					if (not(f.startswith('.')) and (regex in f))] 
	else:
		fnameList = [os.path.normcase(f)
				for f in os.listdir(directory)
					if (not(f.startswith('.')))] 
	
	fileList = [os.path.join(directory, f) 
			   for f in fnameList
				if os.path.splitext(f)[1] in fileExtList]  
	return fileList , fnameList
	
def min_max_length(filelist):
	"""returns the min and max lengths of proteins/ligands loaded from files in filelist"""
	min_length = 1000000
	max_length = 0
	for file in filelist: 
		pdb = read_pdb(file)
		length = pdb.shape[1] 
		if length > max_length:
			max_length = length
			max_file = file
		if length < min_length:
			min_length = length
			min_file = file
		
	return (min_length,max_length), min_file, max_file

def create_target(pro_file,lig_file,format="int"):
	"""Checks from file the label of the protein and ligand.
	eg. 0001_lig_cg.pdb -> return 0001
	If label of protein and ligand are the same, set target as 1, else set as 0 ie. binary classes"""
	P,L = os.path.basename(pro_file),os.path.basename(lig_file)
	pro_code = P[:4] #first 4 digits indicate the code
	lig_code = L[:4]
	if format == "array":
		if pro_code == lig_code:
			return np.array([1])
		else:
			return np.array([0])
	elif format == "int":
		if pro_code == lig_code:
			return 1
		else:
			return 0
	elif format == "one-hot":
		if pro_code == lig_code:
			return np.array([1,0])
		else:
			return np.array([0,1])
	else:
		raise ValueError("can take only int, array or one-hot as output format")
	
def make_dataset(pro_fileList,lig_fileList):
	"""make a dataset containing all possible combinations of proteins and ligands
	len(all_pairs) = len(pro_fileList)C1 x len(lig_fileList)C1"""
	all_pairs = [(x,y) for x in pro_fileList for y in lig_fileList]
	return all_pairs
	
def dataset_randomsplit(positive_dataset, negative_dataset, pos_ratio=0.1, neg_ratio=0.1):
	"""
	Randomly split a dataset into non-overlapping new datasets of given lengths. 
	Handles positive and negative samples separately to take care of unbalanced datasets.
	Arguments:
		dataset (Dataset): Dataset to be split
		ratio (sequence): ratio of dataset (if <1) or number of samples (>1) to be used as validation
	"""
	assert pos_ratio <= len(positive_dataset)
	assert neg_ratio <= len(negative_dataset) 
	random.shuffle(positive_dataset)
	random.shuffle(negative_dataset)

	if neg_ratio <= 1:
		neg_split = int(len(negative_dataset) * neg_ratio)
		neg_split -= neg_split % 1
	else:
		neg_split = neg_ratio
	if pos_ratio <= 1:
		pos_split = int(len(positive_dataset) * pos_ratio)
		pos_split -= pos_split % 1
	else:
		pos_split = pos_ratio

	positive_train = positive_dataset[pos_split:]
	negative_train = negative_dataset[neg_split:]
	positive_test = positive_dataset[:pos_split]
	negative_test = negative_dataset[:neg_split]

	test_set = negative_test + positive_test
	return test_set, negative_train, positive_train

def create_minidataset(negative_train, positive_train, length, epoch):
	"""creates a new 'mini' dataset every epoch with new negative examples"""
	training_set = positive_train + negative_train[epoch*length:(epoch+1)*length]
	return training_set


#*****************************
#utils
def centroid(array):
	"""takes in 3d array with rows for each dimension"""
	return np.mean(array[:3,:], axis=1, keepdims=True)	


#*****************************
#transforms

class create_voxel1():
	"""create voxel spanning max/min coords of protein (hence variable size with protein). ligands used the same voxel
	4 features: -protein histogram
				-ligand histogram
				-protein hydrophobic atoms
				-ligand hydrophobic atoms"""
	
	def __init__(self, bins=20):
		self.bins = bins
	
	def __call__(self, sample):
		protein, ligand = sample[0], sample[1]
		P_his = protein.T[:,:3]
		L_his = ligand.T[:,:3]
		HP_his = protein.T[:,3]
		HL_his = ligand.T[:,3]
		
		amin = np.amin(protein[:3,:],axis=1,keepdims=True)	 
		amax= np.amax(protein[:3,:],axis=1,keepdims=True)

		HP, edgesP = np.histogramdd(P_his, bins = self.bins)
		HL, edgesL = np.histogramdd(L_his, bins = self.bins, range=((amin[0],amax[0]),(amin[1],amax[1]),(amin[2],amax[2])))
		HHP, edgesHHP = np.histogramdd(P_his, bins = self.bins, range=((amin[0],amax[0]),(amin[1],amax[1]),(amin[2],amax[2])), weights=HP_his)
		HHL, edgesHHL = np.histogramdd(L_his, bins = self.bins, range=((amin[0],amax[0]),(amin[1],amax[1]),(amin[2],amax[2])), weights=HL_his)
		
		Hfull = np.stack((HP,HL,HHP,HHL),axis=0)
	
		return Hfull
		
class create_voxel2():
	"""create voxel with ligand centroid as origin, with span=hrange*2 in each dimension (hence fixed size). proteins use same voxel.
	3 features: -protein + ligand combined histogram (protein=1, ligand=-1)
				-protein hydrophobic atoms (if present=1)
				-ligand hydrophobic atoms (if present=-1)"""
	def __init__(self, bins=30, hrange=30):
		self.bins = bins
		self.hrange = hrange
	
	def __call__(self, sample):
		protein, ligand = sample[0], sample[1]
		cen_L = centroid(ligand)
		
		Lweights = np.ones(ligand.shape[1])*-1
		Lrange = ((cen_L[0]-self.hrange,cen_L[0]+self.hrange),(cen_L[1]-self.hrange,cen_L[1]+self.hrange),(cen_L[2]-self.hrange,cen_L[2]+self.hrange))
		Lhis = ligand.T[:,:3]
		Phis = protein.T[:,:3]
		Ph = protein.T[:,3]
		Lh = ligand.T[:,3]*-1

		HL, edgesL = np.histogramdd(Lhis, bins = self.bins, range=Lrange, weights=Lweights)
		HP, edgesP = np.histogramdd(Phis, bins = self.bins, range=Lrange)
		HPh, edgesPh = np.histogramdd(Phis, bins = self.bins, range=Lrange, weights=Ph)
		HLh, edgesLh = np.histogramdd(Lhis, bins = self.bins, range=Lrange, weights=Lh)		
		
		Hfull = np.stack((HP+HL,HPh,HLh),axis=0)
	
		return Hfull
		
class array2tensor():
	"""Convert ndarrays in sample to Tensors. Samples are assumed to be python dics"""
	def __init__(self,dtype):
		self.dtype = dtype
	
	def __call__(self, sample):
		return torch.from_numpy(sample).type(self.dtype)
		
class rotation3D():
	def __init__(self,axis=None):
		self.axis = axis
		
	def __call__(self, sample):
		protein, ligand = sample[0], sample[1]
		if self.axis is None:
			self.axis = np.random.uniform(size=(3,))
		else:
			self.axis = np.asarray(self.axis)
		self.axis = self.axis / math.sqrt(np.dot(self.axis, self.axis))
		theta = np.random.rand()* 2 * math.pi 
		
		a = math.cos(theta / 2.0)
		b, c, d = -self.axis * math.sin(theta / 2.0)
		aa, bb, cc, dd = a * a, b * b, c * c, d * d
		bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
		
		rotMat = np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
						 [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
						 [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
						 
		rot_P = np.dot(rotMat, protein[:3,:])
		rot_L = np.dot(rotMat, ligand[:3,:])
		return (np.vstack((rot_P,protein[3,:])),np.vstack((rot_L,ligand[3,:])))

		
#*****************************
#dataset prep

class ProLigDataset():
	"""Prepares the dataset by combining all possible paired combinations
	and splits them into training and test (validation) sets specified by parameters: split, pos_ratio, neg_ratio
	"""

	def __init__(self, root, split=True, pos_ratio=0.1, neg_ratio=0.1):
		self.root = root
		self.pos_ratio = pos_ratio
		self.neg_ratio = neg_ratio

		self.lig_fullpaths, self.lig_fnameList = listDirectory(root, '.pdb','lig') 
		self.pro_fullpaths, self.pro_fnameList = listDirectory(root, '.pdb','pro') 
		#self.lig_maxmin, self.lig_min_file, self.lig_max_file = min_max_length(self.lig_fullpaths)
		#self.pro_maxmin, self.pro_min_file, self.pro_max_file = min_max_length(self.pro_fullpaths)

		self.all_pairs = make_dataset(self.pro_fullpaths,self.lig_fullpaths)
		if len(self.all_pairs) == 0:
			raise(RuntimeError("Found 0 datafiles in subfolders of: " + root + "\n"))

		if split:
			self.positive = []
			self.negative = []		
			for path in self.all_pairs:
				if create_target(path[0],path[1])==1:
					self.positive.append(path)
				elif create_target(path[0],path[1])==0:
					self.negative.append(path)
					
			self.test_set, self.negative_train, self.positive_train = dataset_randomsplit(self.positive, self.negative, self.pos_ratio, self.neg_ratio)
		
	def __len__(self):
		return len(self.all_pairs)
		
		
class miniDataset(data.Dataset):
	"""A data loader for protein and ligand coords .pdb files:
	mode - 'training' or 'test' - uses the appropriate read_pdb function
	format - format of target - 'array', 'one-hot' or 'int' - see create_target function for more info
	"""

	def __init__(self, dataset, mode='training', format='array', transform=None, target_transform=None):
		
		self.dataset = dataset
		
		self.transform = transform
		self.target_transform = target_transform
		self.mode = mode
		self.format = format

	def __getitem__(self, index):
		"""
		Args:
			index (int): Index

		Returns:
			input, target is a binary class of whether the protein and ligand in input will bind.
		"""
		pro_path, lig_path = self.dataset[index][0], self.dataset[index][1]
		if self.mode == 'training':
			protein, ligand = read_pdb(pro_path), read_pdb(lig_path)
		else:
			protein, ligand = read_pdb_test(pro_path), read_pdb_test(lig_path)
		P_len, L_len = protein.shape[1], ligand.shape[1]
		
		target = create_target(pro_path,lig_path,format=self.format)
		
		if self.transform is not None:
			input = self.transform((protein,ligand))
		if self.target_transform is not None:
			target = self.target_transform(target)

		#input = torch.unsqueeze(torch.from_numpy(input).type(torch.FloatTensor),dim=0)
		#input = torch.from_numpy(input).type(torch.FloatTensor)
		#target = torch.from_numpy(target).type(torch.LongTensor)
		return input, target

	def __len__(self):
		return len(self.dataset)

"""
#To test:
adataset = ProLigDataset(root='training_data')

for i in range(len(adataset)):
	inp,target = adataset[i]
	print(inp)
	print(target)
	
	if i == 2:
		break 
"""

