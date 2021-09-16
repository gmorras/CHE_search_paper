import os
os.environ.update(
    OMP_NUM_THREADS = '1',
    OPENBLAS_NUM_THREADS = '1',
    NUMEXPR_NUM_THREADS = '1',
    MKL_NUM_THREADS = '1',
)

import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from pycbc.detector import Detector
from pycbc.types import TimeSeries
from pycbc.frame import query_and_read_frame
from pycbc import frame
from pycbc import catalog
import pycbc.noise
import pycbc.psd
pi = np.pi
import time
from PIL import Image, ImageOps

start_runtime = time.time() 

#load the data which we are going to analyze
All_DATA = np.loadtxt("/lustre/gmorras/StrainData/GWOSC/O2/DetectorStatus/All_DATA.txt")
begins  = All_DATA[:,0]
ends = All_DATA[:,1]

file_size = 4096
fold_size = 4096*256
min_window_duration = 500

#include all plus minus allshift combinations from tshiftmin until tshiftmax (both included)
tshiftmin = 1
tshiftmax = 7

Output_path = "/lustre/gmorras/samples4TriggerNN/samples50x50/CoincidenceBackground_together/"

#Number of pixels in the final plots
Npixels = 50

#dynamic range
cmin = 1
cmax = 9	

for i_wind in range(begins.size):
	#GPS of the data to plot
	start_data = begins[i_wind]
	end_data = ends[i_wind]
	window_duration = end_data - start_data
	if window_duration < min_window_duration: continue
	
	print("\n\nstart_data = %.f    end_data = %.f    duration = %.f"%(start_data,end_data,window_duration))

	if start_data < 1187609754-1: continue

	#filter characteristics
	flow_bp = 20 #Hz
	fhigh_bp = 800 #Hz
	order = 512
	flow_noise = 10
	
	#get the experimental data
	t0 = start_data
	while t0 < end_data:
		#load the data
		#initial part of the path
		preff_str = '/lustre/gmorras/StrainData/GWOSC/O2/'
		#find the folder in which the file is at (only valid for times greater than 1180696576)
		t0_fold = (t0//fold_size)*fold_size
		fold_str = '%.f'%(t0_fold)
		
		#find the file in which it is located
		t0_file = (t0//file_size)*file_size
		tf = min(t0_file+file_size, end_data)	
		suff_str = '_GWOSC_O2_4KHZ_R1-%.f-4096.gwf'%(t0_file)
		
		#read the files
		read_hdet_exp_series_L1 = frame.read_frame(preff_str+'L1/'+ fold_str +'/L-L1'+suff_str, 'L1:GWOSC-4KHZ_R1_STRAIN',start_time = t0, end_time = tf)
		read_hdet_exp_series_H1 = frame.read_frame(preff_str+'H1/'+ fold_str +'/H-H1'+suff_str, 'H1:GWOSC-4KHZ_R1_STRAIN',start_time = t0, end_time = tf)
		read_hdet_exp_series_V1 = frame.read_frame(preff_str+'V1/'+ fold_str +'/V-V1'+suff_str, 'V1:GWOSC-4KHZ_R1_STRAIN',start_time = t0, end_time = tf)
		
		#append the data to the timeseries
		if t0 == start_data:
			hdet_exp_series_L1 = read_hdet_exp_series_L1
			hdet_exp_series_H1 = read_hdet_exp_series_H1
			hdet_exp_series_V1 = read_hdet_exp_series_V1
			chunk_epoch = hdet_exp_series_L1.start_time
			chunk_delta_t = hdet_exp_series_L1.delta_t
		else:
			hdet_exp_series_L1 = TimeSeries(np.append(hdet_exp_series_L1,read_hdet_exp_series_L1), delta_t=chunk_delta_t, epoch=chunk_epoch)
			hdet_exp_series_H1 = TimeSeries(np.append(hdet_exp_series_H1,read_hdet_exp_series_H1), delta_t=chunk_delta_t, epoch=chunk_epoch)
			hdet_exp_series_V1 = TimeSeries(np.append(hdet_exp_series_V1,read_hdet_exp_series_V1), delta_t=chunk_delta_t, epoch=chunk_epoch)
		
		#iterate
		t0 = t0_file + file_size

	print("Returned {}s of Livingston data at {}Hz".format(hdet_exp_series_L1.duration ,hdet_exp_series_L1.sample_rate))
	print("Returned {}s of Hanford data at {}Hz".format(hdet_exp_series_H1.duration, hdet_exp_series_H1.sample_rate))
	print("Returned {}s of Virgo data at {}Hz".format(hdet_exp_series_V1.duration ,hdet_exp_series_V1.sample_rate))

	#consider all cases
	for shift_str in range(tshiftmin, tshiftmax+1):
		for sign in range(2):
			
			if sign == 0:
				case = "Shifted_H1p%.f_V1m%.f"%(shift_str, shift_str)
				allshiftH1 = shift_str
				allshiftV1 = -shift_str
			if sign == 1:
				case = "Shifted_H1m%.f_V1p%.f"%(shift_str, shift_str)
				allshiftH1 = -shift_str
				allshiftV1 = shift_str

			print("\n"+case+"  dtH1 = %.1f    dtV1 = %.1f"%(allshiftH1,allshiftV1))

			#load images to be printed
			image_shiftH1_shiftV1_rL1H1_rL1V1 = np.loadtxt("/lustre/gmorras/Triggered/"+case+"/Text/image_shiftH1_shiftV1_rL1H1_rL1V1_%.2f_%.2f.txt"%(start_data,end_data))
		
			#if there are no images, continue
			if image_shiftH1_shiftV1_rL1H1_rL1V1.size == 0: continue
			
			#shift the Hanford time series by an amount allshiftH1:
			hdet_exp_series_H1 = TimeSeries(np.roll(np.array(hdet_exp_series_H1),int(allshiftH1/chunk_delta_t)), delta_t=chunk_delta_t, epoch=chunk_epoch)

			#shift the Virgo time series by an amount allshiftH1:
			hdet_exp_series_V1 = TimeSeries(np.roll(np.array(hdet_exp_series_V1),int(allshiftV1/chunk_delta_t)), delta_t=chunk_delta_t, epoch=chunk_epoch)

			#generate the images	
			#size and overlap of the final plots
			t_image_duration = 0.7
			t_image_overlap = 0.35
			t_image_step = t_image_duration - t_image_overlap
			qtimesteps = t_image_duration/Npixels
			
			#load images to be represented
			if image_shiftH1_shiftV1_rL1H1_rL1V1.size == 5:
				images = np.array([image_shiftH1_shiftV1_rL1H1_rL1V1[0]])
				shiftsH1 = np.array([image_shiftH1_shiftV1_rL1H1_rL1V1[1]])
				shiftsV1 = np.array([image_shiftH1_shiftV1_rL1H1_rL1V1[2]])
				rsL1H1 = np.array([image_shiftH1_shiftV1_rL1H1_rL1V1[3]])
				rsL1V1 = np.array([image_shiftH1_shiftV1_rL1H1_rL1V1[4]])
			else:
				images = image_shiftH1_shiftV1_rL1H1_rL1V1[:,0]
				shiftsH1 = image_shiftH1_shiftV1_rL1H1_rL1V1[:,1]
				shiftsV1 = image_shiftH1_shiftV1_rL1H1_rL1V1[:,2]
				rsL1H1 = image_shiftH1_shiftV1_rL1H1_rL1V1[:,3]
				rsL1V1 = image_shiftH1_shiftV1_rL1H1_rL1V1[:,4]

			#original discriminant formula
			f_r_L1_H1_V1 = 0.67*np.maximum(0,rsL1H1) + 0.33*np.maximum(0,rsL1V1)
				
			#size and overlap of the chunks to analyze
			t_det_duration = 80
			t_det_overlap = 16
			t_det_step = t_det_duration - t_det_overlap
			
			nimages = images.size

			for t_det_sample in np.arange(start_data+t_det_overlap/2,end_data-t_det_overlap/2-t_det_step,t_det_step):
				#limits of the chunk to be analyzed
				t_det0 = t_det_sample - t_det_overlap/2
				t_detf = t_det_sample + t_det_step + t_det_overlap/2
				#print("t_det0 = %.f     t_detf= %.f"%(t_det0-start_data,t_detf-start_data))
				
				#decide if this chunk has to be analyzed and find the images contained within
				dont_analyze_chunk = True
				i_sample = []
				for i_decide in range(nimages):
					if images[i_decide] >= t_det_sample and images[i_decide] < t_det_sample+t_det_step: 
						dont_analyze_chunk = False
						i_sample = np.append(i_sample,i_decide)
						print("%.f/%.f"%(i_decide+1,nimages))
					
				if dont_analyze_chunk: continue
				
				
				#cut the chunk of strain to analyze
				hdet_series_L1 = hdet_exp_series_L1.time_slice(t_det0,t_detf)
				hdet_series_H1 = hdet_exp_series_H1.time_slice(t_det0,t_detf)
				hdet_series_V1 = hdet_exp_series_V1.time_slice(t_det0,t_detf)
				
				#remove very low frequencies
				hdet_noise_series_L1 = hdet_series_L1.highpass_fir(flow_noise,order)
				hdet_noise_series_H1 = hdet_series_H1.highpass_fir(flow_noise,order)
				hdet_noise_series_V1 = hdet_series_V1.highpass_fir(flow_noise,order)
				
				#whiten signal
				whiten_hdet_series_L1, psd_whiten_L1 = hdet_series_L1.whiten(4,4, return_psd=True, low_frequency_cutoff=flow_noise)
				whiten_hdet_series_H1, psd_whiten_H1 = hdet_series_H1.whiten(4,4, return_psd=True, low_frequency_cutoff=flow_noise)
				whiten_hdet_series_V1, psd_whiten_V1 = hdet_series_V1.whiten(4,4, return_psd=True, low_frequency_cutoff=flow_noise)
				
				#filtered signal
				bp_hdet_series_L1 = whiten_hdet_series_L1.highpass_fir(flow_bp,order).lowpass_fir(fhigh_bp,order)
				bp_hdet_series_H1 = whiten_hdet_series_H1.highpass_fir(flow_bp,order).lowpass_fir(fhigh_bp,order)
				bp_hdet_series_V1 = whiten_hdet_series_V1.highpass_fir(flow_bp,order).lowpass_fir(fhigh_bp,order)

				#Q transform
				qtimes_L1, qfreqs_L1, qpower_L1 = bp_hdet_series_L1.qtransform(delta_t = qtimesteps, logfsteps = Npixels, qrange=(8,8), frange=(flow_bp,fhigh_bp))
				qtimes_H1, qfreqs_H1, qpower_H1 = bp_hdet_series_H1.qtransform(delta_t = qtimesteps, logfsteps = Npixels, qrange=(8,8), frange=(flow_bp,fhigh_bp))
				qtimes_V1, qfreqs_V1, qpower_V1 = bp_hdet_series_V1.qtransform(delta_t = qtimesteps, logfsteps = Npixels, qrange=(8,8), frange=(flow_bp,fhigh_bp))

				#generate the images
				for i_image in i_sample:
				
					#limits of the L1 image
					t_image0_L1 = images[int(i_image)] - t_image_duration/2
					t_imagef_L1 = images[int(i_image)] + t_image_duration/2	
					#plotting indices
					i0plot_L1 = np.argmin(np.abs(qtimes_L1-t_image0_L1))-1
					ifplot_L1 = i0plot_L1 + Npixels
					#quatity to plot
					lnqpower_plot_L1 = np.log(qpower_L1[:,i0plot_L1:ifplot_L1])
					#array to finally plot
					arr_plot_L1 = np.where(lnqpower_plot_L1 < cmin, cmin, lnqpower_plot_L1)
					arr_plot_L1 = np.where(arr_plot_L1 > cmax, cmax, arr_plot_L1)				
					#transform to uint8
					arr_plot_L1 = np.uint8(255*(arr_plot_L1-cmin)/(cmax-cmin))			
					
					#limits of the H1 image
					t_image0_H1 = t_image0_L1 + shiftsH1[int(i_image)]
					t_imagef_H1 = t_imagef_L1 + shiftsH1[int(i_image)]		
					#plotting indices
					i0plot_H1 = np.argmin(np.abs(qtimes_H1-t_image0_H1))-1
					ifplot_H1 = i0plot_H1 + Npixels
					#quatity to plot
					lnqpower_plot_H1 = np.log(qpower_H1[:,i0plot_H1:ifplot_H1])
					#array to finally plot
					arr_plot_H1 = np.where(lnqpower_plot_H1 < cmin, cmin, lnqpower_plot_H1)
					arr_plot_H1 = np.where(arr_plot_H1 > cmax, cmax, arr_plot_H1)				
					#transform to uint8
					arr_plot_H1 = np.uint8(255*(arr_plot_H1-cmin)/(cmax-cmin))			
									
					#limits of the V1 image
					t_image0_V1 = t_image0_L1 + shiftsV1[int(i_image)]
					t_imagef_V1 = t_imagef_L1 + shiftsV1[int(i_image)]	
					#plotting indices
					i0plot_V1 = np.argmin(np.abs(qtimes_V1-t_image0_V1))-1
					ifplot_V1 = i0plot_V1 + Npixels
					#quatity to plot
					lnqpower_plot_V1 = np.log(qpower_V1[:,i0plot_V1:ifplot_V1])
					#array to finally plot
					arr_plot_V1 = np.where(lnqpower_plot_V1 < cmin, cmin, lnqpower_plot_V1)
					arr_plot_V1 = np.where(arr_plot_V1 > cmax, cmax, arr_plot_V1)				
					#transform to uint8
					arr_plot_V1 = np.uint8(255*(arr_plot_V1-cmin)/(cmax-cmin))			

					#put all in a single image
					arr_plot = np.append(np.flip(arr_plot_L1, axis=0), np.flip(arr_plot_H1, axis=0), axis=0)
					arr_plot = np.append(arr_plot, np.flip(arr_plot_V1,axis=0), axis=0)
					im_plot = Image.fromarray(arr_plot)
					im_plot.save(Output_path+case+"_%.3f_%.3f.png"%(images[int(i_image)],f_r_L1_H1_V1[int(i_image)]))			

			#liberate some memory
			del hdet_series_L1, hdet_series_H1, hdet_series_V1, hdet_noise_series_L1, hdet_noise_series_H1, hdet_noise_series_V1
			del bp_hdet_series_L1, bp_hdet_series_H1, bp_hdet_series_V1
			del whiten_hdet_series_L1, psd_whiten_L1, whiten_hdet_series_H1, psd_whiten_H1, whiten_hdet_series_V1, psd_whiten_V1
			del qtimes_L1, qfreqs_L1, qpower_L1, qtimes_H1, qfreqs_H1, qpower_H1, qtimes_V1, qfreqs_V1, qpower_V1
			del lnqpower_plot_L1, arr_plot_L1, lnqpower_plot_H1, arr_plot_H1, lnqpower_plot_V1, arr_plot_V1, arr_plot, im_plot
				
			#unshift the Hanford time
			hdet_exp_series_H1 = TimeSeries(np.roll(np.array(hdet_exp_series_H1),-int(allshiftH1/chunk_delta_t)), delta_t=chunk_delta_t, epoch=chunk_epoch)

			#unshift the Virgo time:
			hdet_exp_series_V1 = TimeSeries(np.roll(np.array(hdet_exp_series_V1),-int(allshiftV1/chunk_delta_t)), delta_t=chunk_delta_t, epoch=chunk_epoch)

#Runtime
print("Runtime: %s seconds" % (time.time() - start_runtime))

