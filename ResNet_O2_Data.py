#Code to analyze the images using a Neural Network
#Authors: Savvas Nesseris and Gonzalo Morrás Gutiérrez
#E-mail: gonzalo.morras@estudiante.uam.es

# Silence annoying TF crap printed on the screen
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=""  # or even "-1"  

# TensorFlow,  tf.keras and matplotlib stuff
import tensorflow as tf
import matplotlib.pyplot as plt

# Helper libraries
import time
import numpy as np
from numpy import asarray
from os import listdir
from pathlib import Path
import PIL
import PIL.Image
import shutil
import re
import time
start_runtime = time.time() 

#Hide GPU from visible devices
tf.config.set_visible_devices([], 'GPU')

# To find out which devices your operations and tensors are assigned to
#tf.debugging.set_log_device_placement(True)
print(tf.config.list_physical_devices('CPU'))

def get_images(path, get_SNR = False):
	
	imgs=list(Path(path).glob('./*.png'))
	images=list()
	if get_SNR: SNR = list()
	
	# Loop over the images  
	for filename in imgs:
			
		#get the images already normalized
		img_data = asarray(PIL.Image.open(str(filename)))/255.0

		images.append(img_data)
		
		if get_SNR:
			#get the SNR from the file name 
			#format (Shifted_H1%s%.f_V1%s%.f_%.3f_%.1f.png)%(p/m, |shift_H1|, p/m, |shift_V1|, t_trigger, SNR)
			head, tail = os.path.split(filename)
			SNR.append(float(re.findall("\d+\.\d+", tail)[-1]))
			
	if get_SNR: return asarray(images), asarray(SNR)
	else: return asarray(images)

# Here it starts   

# These are the names of the classes
# We only have two classes: 0=noise, 1=GW event
class_names = ['0', '1']

#minimum value of the probability to be considered gravitational wave
disc = 0.9

#number of epochs to train NN
epoch_num = 12

# load the model or compute it and save it
load_model = True
model_name = 'ResNet_v1.h5'

#Show the training curves for the validation images
no_train_curves = False

# Set the root path of the data
datapath = '/lustre/gmorras/samples4TriggerNN/samples50x50/'
test_to_exp =  1/18.0

# Set the test paths
test_path_GWs=datapath+'Validation_General_Injected_together/'
test_path_noise=datapath+'Validation_CoincidenceBackground_together/'

# Set the training paths
train_path_GWs=datapath+'General_Injected_together/'
train_path_noise=datapath+'CoincidenceBackground_together/'

# Set the O2 data path
O2_data_path = datapath+'O2_Data_together/'

# Set up the model and do the training
if load_model:
	print('Now loading model...')
	model = tf.keras.models.load_model(model_name)
else:
	# Read the GW training images
	print('Now reading the GWs train images...')
	train_images_GWs=get_images(train_path_GWs)
	train_labels_GWs=np.repeat(1,len(train_images_GWs))
	print(train_images_GWs.shape)

	# Read the noise training images
	print('Now reading the noise train images...')
	train_images_noise=get_images(train_path_noise)
	train_labels_noise=np.repeat(0,len(train_images_noise))
	print(train_images_noise.shape)

	#Gather all the train images
	train_images=np.concatenate((train_images_GWs,train_images_noise))

	#Gather all the train labels
	train_labels=np.concatenate((train_labels_GWs,train_labels_noise))

	#liberate unused memory
	del train_images_GWs, train_images_noise, train_labels_GWs, train_labels_noise

	baseModel= tf.keras.applications.ResNet50V2(weights=None, include_top=False, input_shape=(150,50,1), pooling="avg") #v1 v2

	out=baseModel.layers[-1].output 
	output = tf.keras.layers.Dense(1, activation='sigmoid')(out)

	model = tf.keras.models.Model(inputs=baseModel.input, outputs=output)

	model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy',  metrics=['accuracy'])
	
	
	if no_train_curves:
		
		print('Now starting the training!')
		
		#if no training curves, no need to load validation data
		model.fit(train_images, train_labels, epochs=epoch_num, batch_size = 32, verbose=2) 

		print('Now saving model...')
		model.save(model_name)
	
		#liberate memory
		del train_images, train_labels, baseModel, out, output
	else:
		#if training curves, need to load validation data
		# Read the GW test images
		print('Now reading the GW test images...')
		test_images_GWs=get_images(test_path_GWs)
		test_labels_GWs=np.repeat(1,len(test_images_GWs))
		print(test_images_GWs.shape)

		# Read the noise test images
		print('Now reading the noise test images...')
		test_images_noise=get_images(test_path_noise)
		test_labels_noise=np.repeat(0,len(test_images_noise))
		print(test_images_noise.shape)

		#Merge the two
		test_images= np.concatenate((test_images_GWs,test_images_noise))
		test_labels= np.concatenate((test_labels_GWs,test_labels_noise))
		del test_images_GWs, test_images_noise, test_labels_GWs, test_labels_noise
		
		print('Now starting the training!')
		#fit the model
		history = model.fit(train_images, train_labels, epochs=epoch_num, batch_size = 32, verbose=2, validation_data = (test_images, test_labels)) 

		print('Now saving model...')
		model.save(model_name)
	
		#liberate memory
		del train_images, train_labels, baseModel, out, output, test_images, test_labels
		
		#generate the plots with the training history

		# summarize history for accuracy
		plt.figure(6,figsize = (15,10))
		plt.rcParams.update({'font.size': 25})
		plt.plot(np.arange(epoch_num)+1, history.history['accuracy'], label = 'Training', linewidth = 2)
		plt.plot(np.arange(epoch_num)+1, history.history['val_accuracy'], label = 'Validation', linewidth = 2)
		plt.ylabel(r'Accuracy')
		plt.xlabel(r'Epoch')
		plt.xlim(1,epoch_num)
		plt.legend()
		plt.grid()
		plt.savefig("Output_ResNet/Accuracy.png", bbox_inches="tight")
		
		# summarize history for loss
		plt.figure(7,figsize = (15,10))
		plt.rcParams.update({'font.size': 25})
		plt.plot(np.arange(epoch_num)+1, history.history['loss'], label = 'Training', linewidth = 2)
		plt.plot(np.arange(epoch_num)+1, history.history['val_loss'], label = 'Validation', linewidth = 2)
		plt.ylabel(r'Loss')
		plt.xlabel(r'Epoch')
		plt.xlim(1,epoch_num)
		plt.legend()
		plt.grid()
		plt.savefig("Output_ResNet/Loss.png", bbox_inches="tight")
		
		#save training history
		np.savetxt("history.txt", np.transpose([history.history['loss'], history.history['val_loss'], history.history['accuracy'], history.history['val_accuracy']]))
		del history		

		
# Read the GW test images
print('Now reading the GW test images...')
test_images_GWs=get_images(test_path_GWs)
test_labels_GWs=np.repeat(1,len(test_images_GWs))
print(test_images_GWs.shape)

# Read the noise test images
print('Now reading the noise test images...')
test_images_noise=get_images(test_path_noise)
test_labels_noise=np.repeat(0,len(test_images_noise))
print(test_images_noise.shape)

#Merge the two
test_images= np.concatenate((test_images_GWs,test_images_noise))
test_labels= np.concatenate((test_labels_GWs,test_labels_noise))
del test_images_GWs, test_images_noise, test_labels_GWs, test_labels_noise

#test validation
test_loss, test_acc = model.evaluate(test_images,test_labels,verbose=0)

print("test_loss = ", test_loss, "        test_acc = ", test_acc)

# Get the probabilities and save the training
predictions = asarray(model.predict(test_images))[:,0]	

#Assign the classes
real_vals = asarray(test_labels)
del test_images, test_labels

#Generate the ROC curve of the NN
tROC = np.linspace(0,1,101)
false_positive_frac_t = np.array([])
true_positive_frac_t = np.array([])

for t_ROC_c in tROC:

	#Assign predicted class
	predclass_t = np.where(predictions>t_ROC_c, 1, 0)

	#truth table
	true_positives_t = np.sum(np.where(np.logical_and(predclass_t == 1, real_vals == predclass_t), 1, 0))	
	false_positives_t = np.sum(np.where(np.logical_and(predclass_t == 1, real_vals != predclass_t), 1, 0))	
	true_negatives_t = np.sum(np.where(np.logical_and(predclass_t == 0, real_vals == predclass_t), 1, 0))	
	false_negatives_t = np.sum(np.where(np.logical_and(predclass_t == 0, real_vals != predclass_t), 1, 0))

	#accuracy of the neural network
	false_positive_frac_t = np.append(false_positive_frac_t, false_positives_t/(false_positives_t + true_negatives_t))
	true_positive_frac_t = np.append(true_positive_frac_t, true_positives_t/(false_negatives_t + true_positives_t))

#Plot the ROC curve
plt.figure(1,figsize = (15,10))
plt.rcParams.update({'font.size': 25})
plt.plot(100*false_positive_frac_t, 100*true_positive_frac_t, linewidth = 2)
plt.xlabel(r'False Positive Rate (%)')
plt.ylabel(r'True Positive Rate (%)')
plt.grid()
plt.savefig("Output_ResNet/ROC.png", bbox_inches="tight")

#Assign predicted class
predclass = np.where(predictions>disc, 1, 0)

#truth table
true_positives = np.sum(np.where(np.logical_and(predclass == 1, real_vals == predclass), 1, 0))	
false_positives = np.sum(np.where(np.logical_and(predclass == 1, real_vals != predclass), 1, 0))	
true_negatives = np.sum(np.where(np.logical_and(predclass == 0, real_vals == predclass), 1, 0))	
false_negatives = np.sum(np.where(np.logical_and(predclass == 0, real_vals != predclass), 1, 0))
print("\nTrue Positive = %.f      False Positive = %.f"%(true_positives,false_positives))
print("True Negative = %.f      False Negative = %.f"%(true_negatives,false_negatives))

#accuracy of the neural network
false_positive_frac = false_positives/(false_positives + true_negatives)
false_negative_frac = false_negatives/(false_negatives + true_positives)

print("\nfalse_positive_frac = %.4f    false_negative_frac = %.4f\n"%(false_positive_frac, false_negative_frac))

# Read the data images
print('Now reading the O2 data images...')
data_images = get_images(O2_data_path)
data_images_names = list(Path(O2_data_path).glob('./*.png'))
print(data_images.shape)

#Run the neural network on the O2 data
O2_predictions = asarray(model.predict(data_images))[:,0]
del data_images

#find the hyperbolic candidates and save them in a folder
hyperbolic_candidates = list()
hyperbolic_candidates_path = 'Output_ResNet/hyperbolic_candidates/'
try:
	shutil.rmtree(hyperbolic_candidates_path)
	os.mkdir(hyperbolic_candidates_path)
except:
	os.mkdir(hyperbolic_candidates_path)


print("\nO2 candidates with a probability of being hyperbolic events greater than %.2f:"%(disc))

for ii in range(len(data_images_names)):
	if O2_predictions[ii]>disc: 
		hyperbolic_candidates.append(data_images_names[ii])
		print(O2_predictions[ii]," : ",data_images_names[ii])
		hyperbolic_candidates_head, hyperbolic_candidates_tail = os.path.split(data_images_names[ii])
		shutil.copyfile(data_images_names[ii], hyperbolic_candidates_path+hyperbolic_candidates_tail)

print("\nNumber of hyperbolic event candidates expected: %.1f +- %.1f"%(len(data_images_names)*false_positive_frac, np.sqrt(len(data_images_names)*false_positive_frac)))
print("Number of hyperbolic event candidates found: %.f "%(len(hyperbolic_candidates)))

#make a histogram of the probabilities for the training sample and data
prob_GW_test = (predictions[np.argwhere(real_vals == 0)[:,0]])
prob_GW_O2 = O2_predictions

N_test = np.size(prob_GW_test)
N_O2 = np.size(prob_GW_O2)

#Probability range 
prob_GW_0 = 0
prob_GW_f = 1

#number of bins
prob_Nbins = 10

#histogram of probabilities
hprob_GW_test, prob_bin_edges = np.histogram(prob_GW_test, bins=prob_Nbins, range=(prob_GW_0,prob_GW_f))
hprob_GW_O2, _ = np.histogram(prob_GW_O2, bins=prob_Nbins, range=(prob_GW_0,prob_GW_f))
prob_bins = 0.5*(prob_bin_edges[1:prob_Nbins+1]+prob_bin_edges[0:prob_Nbins])
prob_ebins= 0.5*(prob_bin_edges[1:prob_Nbins+1]-prob_bin_edges[0:prob_Nbins])

#plot histograms
plt.figure(2,figsize = (15,10))
plt.rcParams.update({'font.size': 25})
plt.errorbar(prob_bins, hprob_GW_O2, xerr=prob_ebins, fmt='o', capsize = 3.5, label = r"Observed Events")
plt.errorbar(prob_bins, test_to_exp*hprob_GW_test, yerr = np.sqrt(test_to_exp*hprob_GW_test),xerr=prob_ebins, fmt='o', capsize = 3.5, label = r"Expected Noise Events")
plt.xlabel(r"CHE Probability")
plt.ylabel(r"Number of events")
plt.xlim(prob_GW_0, prob_GW_f)
plt.yscale("log")
plt.grid()
plt.legend()
plt.savefig("Output_ResNet/hprob_GW.png", bbox_inches="tight")

#find trigger efficiency as a function of SNR
#SNR range to study
SNR_0 = 4
SNR_f = 40
SNR_Nbins = 36

#get the images with their respective SNR
test_images_GWs, SNR = get_images(test_path_GWs, get_SNR = True)

#Run the neural on the GW images
test_GWs_predictions = asarray(model.predict(test_images_GWs))
del test_images_GWs

#SNR of the injections that are detected
SNR_det = (SNR[np.argwhere(test_GWs_predictions > disc)])[:,0]

#histogram of injected SNR
hSNR_all, SNR_bin_edges = np.histogram(SNR, bins=SNR_Nbins, range=(SNR_0,SNR_f))
SNR_bins = 0.5*(SNR_bin_edges[1:SNR_Nbins+1]+SNR_bin_edges[0:SNR_Nbins])
SNR_ebins= 0.5*(SNR_bin_edges[1:SNR_Nbins+1]-SNR_bin_edges[0:SNR_Nbins])

#histogram of detected SNR
hSNR_det, _ = np.histogram(SNR_det, bins=SNR_Nbins, range=(SNR_0,SNR_f))

print("\nNumber of events injected: %.f"%(np.sum(hSNR_all)))
print("Number of events recovered: %.f"%(np.sum(hSNR_det)))

#plot histogram of injected SNR
plt.figure(3,figsize = (15,10))
plt.rcParams.update({'font.size': 25})
plt.errorbar(SNR_bins, hSNR_all, xerr=SNR_ebins, fmt='o', capsize = 3.5, label = r"Total CHE test images")
plt.errorbar(SNR_bins, hSNR_det, xerr=SNR_ebins, fmt='o', capsize = 3.5, label = r"Correctly classified CHE test images")
plt.xlabel(r"Signal to Noise Ratio")
plt.ylabel(r"Number of Events")
plt.xlim(SNR_0, SNR_f)
plt.ylim(0,3000)
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("Output_ResNet/ML_SNR_injected.png", bbox_inches="tight")

#plot efficiency
Trig_eff = hSNR_det/hSNR_all
eTrig_eff = np.sqrt(Trig_eff*(1-Trig_eff)/hSNR_all)
plt.figure(4,figsize = (15,10))
plt.rcParams.update({'font.size': 25})
plt.errorbar(SNR_bins, Trig_eff, yerr=eTrig_eff, xerr=SNR_ebins , fmt='o', capsize = 3.5)
plt.xlabel(r"Signal to Noise Ratio")
plt.ylabel(r"Neural Network Trigger Efficiency")
plt.xlim(SNR_0, SNR_f)
plt.ylim(0,1)
plt.grid()
plt.tight_layout()
plt.savefig("Output_ResNet/ML_Trigger_Efficiency.png", bbox_inches="tight")

#plot probability to be a GW as a function of SNR
prob_GW_GW = test_GWs_predictions[:,0]
injection_weight_SNR = 1/np.interp(SNR, SNR_bins, hSNR_all)

plt.figure(5,figsize = (15,10))
plt.rcParams.update({'font.size': 25})
plt.xlabel(r"Signal to Noise Ratio")
plt.ylabel(r"CHE Probability")
plt.hist2d(SNR,prob_GW_GW, bins = (36,36), range = [[SNR_0, SNR_f], [0,1.0]], weights = injection_weight_SNR, density = True)
plt.xlim(SNR_0, SNR_f)
plt.ylim(0,1.0)
plt.colorbar(label = r"Density of events (arbitrary units)")
plt.grid()
plt.savefig("Output_ResNet/ML_probGW_SNR.png", bbox_inches="tight")


#Runtime	
print("\nRuntime: %s seconds" % (time.time() - start_runtime))

plt.show()

