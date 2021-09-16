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

start_runtime = time.time() 

#load the data which we are going to analyze
All_DATA = np.loadtxt("/lustre/gmorras/StrainData/GWOSC/O2/DetectorStatus/All_DATA.txt")
begins  = All_DATA[:,0]
ends = All_DATA[:,1]

file_size = 4096
fold_size = 4096*256

#only analize the chunk if it sufficiently long 
min_window_duration = 500

#Node parallelization
TotProcessNum = 5
ProcessNum = 4

for i_wind in range(begins.size):

	#only analize chunk if it belongs to the current process
	if i_wind % TotProcessNum != ProcessNum: continue

	#GPS of the data to plot
	start_data = begins[i_wind]
	end_data = ends[i_wind]
	window_duration = end_data - start_data
	if window_duration < min_window_duration: continue
	
	print("\n \n start_data = %.f    end_data = %.f    duration = %.f"%(start_data,end_data,window_duration))

	#filter characteristics
	flow_bp = 20 #Hz
	fhigh_bp = 800 #Hz
	order = 512
	flow_noise = 10

	#size and overlap of the final plots
	t_image_duration = 0.3
	t_image_overlap = t_image_duration/2
	t_image_step = t_image_duration - t_image_overlap

	#Number of pixels in the final plots
	Npixels = 200
	qtimesteps = t_image_duration/Npixels	

	#value of the cut for the correlation coefficients
	weight_r_L1_H1 = 0.67
	f_r_L1_H1_V1_cut = 0.3

	weight_r_L1_V1 = 1 - weight_r_L1_H1
	r_L1_H1_cut = max(0.2,(f_r_L1_H1_V1_cut-weight_r_L1_V1)/weight_r_L1_H1)

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


	#get light travel time between detectors
	dt_L1_H1 = Detector('L1').light_travel_time_to_detector(Detector('H1')) + 2/hdet_exp_series_L1.sample_rate
	dt_L1_V1 = Detector('L1').light_travel_time_to_detector(Detector('V1')) + 2/hdet_exp_series_H1.sample_rate
	dt_H1_V1 = Detector('H1').light_travel_time_to_detector(Detector('V1')) + 2/hdet_exp_series_V1.sample_rate

	#shift the Hanford time series by an amount allshiftH1:
	allshiftH1 = -3
	hdet_exp_series_H1 = TimeSeries(np.roll(np.array(hdet_exp_series_H1),int(allshiftH1/chunk_delta_t)), delta_t=chunk_delta_t, epoch=chunk_epoch)

	#shift the Virgo time series by an amount allshiftH1:
	allshiftV1 = 3
	hdet_exp_series_V1 = TimeSeries(np.roll(np.array(hdet_exp_series_V1),int(allshiftV1/chunk_delta_t)), delta_t=chunk_delta_t, epoch=chunk_epoch)

	#size and overlap of the chunks to analyze
	t_det_duration = 80
	t_det_overlap = 16
	t_det_step = t_det_duration - t_det_overlap

	#matrix with the values of the results
	t_image = np.array([])

	r_L1_H1 = np.array([])
	p_L1_H1 = np.array([])
	true_shifts_L1_H1 = np.array([])

	r_L1_V1 = np.array([])
	p_L1_V1 = np.array([])
	true_shifts_L1_V1 = np.array([])

	#trigger

	for t_det_sample in np.arange(start_data+t_det_overlap/2,end_data-t_det_overlap/2-t_det_step,t_det_step):
		#limits of the chunk to be analyzed
		t_det0 = t_det_sample - t_det_overlap/2
		t_detf = t_det_sample + t_det_step + t_det_overlap/2
		
		
		#cut the chunk of strain to analyze
		hdet_series_L1 = hdet_exp_series_L1.time_slice(t_det0,t_detf)
		hdet_series_H1 = hdet_exp_series_H1.time_slice(t_det0,t_detf)
		hdet_series_V1 = hdet_exp_series_V1.time_slice(t_det0,t_detf)
		
		#check whether there are any empty values
		skip_by_nan = False
		if np.isnan(np.sum(np.array(hdet_series_L1))) == True:
			print("Nan encountered in L1")
			skip_by_nan = True
		if np.isnan(np.sum(np.array(hdet_series_H1))) == True:
			print("Nan encountered in H1")
			skip_by_nan = True
		if np.isnan(np.sum(np.array(hdet_series_V1))) == True:
			print("Nan encountered in V1")
			skip_by_nan = True
			
		if skip_by_nan == True: continue
		
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

		#see correlation between L1 and H1
		for t_image_sample in np.arange(t_det_sample+t_image_overlap/2,t_det_sample+t_det_step-t_image_overlap/2-t_image_step,t_image_step):
			#window limits
			t_image0 = t_image_sample - t_image_overlap/2
			t_imagef = t_image_sample + t_image_step + t_image_overlap/2
			
			#############    L1-H1 correlation  #############  
			#reference plotting indices
			i0_L1_ref = np.argmin(np.abs(qtimes_L1-t_image0))-1
			if_L1_ref = np.argmin(np.abs(qtimes_L1-t_imagef))+1
			N_L1_ref = if_L1_ref-i0_L1_ref+1
			
			#get the reference qpower
			q_L1_ref = qpower_L1[:,i0_L1_ref:(if_L1_ref+1)]
			t_L1_ref = qtimes_L1[i0_L1_ref:(if_L1_ref+1)]
			
			#get the shifting plotting indices
			i0_H1_shift = np.argmin(np.abs(qtimes_H1-(t_image0-dt_L1_H1)))-1
			if_H1_shift = np.argmin(np.abs(qtimes_H1-(t_imagef+dt_L1_H1)))+1
			N_H1_shift = if_H1_shift-i0_H1_shift+1
			
			#shifting qpower
			q_H1_shift = qpower_H1[:,i0_H1_shift:(if_H1_shift+1)]
			
			#consider all possible shifts and find the one that maximizes the correlation
			Nshifts_L1_H1 = N_H1_shift-N_L1_ref
			r_L1_H1_max = -2
			for i_shift in range(Nshifts_L1_H1):
				#compute the Pearson correlation coefficient
				r, p = stats.pearsonr(q_L1_ref.flatten(), q_H1_shift[:,i_shift:(i_shift+N_L1_ref)].flatten())
				
				if r>r_L1_H1_max:
					r_L1_H1_max = r
					p_L1_H1_max = p
					shift_L1_H1 = qtimes_H1[i0_H1_shift+i_shift]-qtimes_L1[i0_L1_ref]
					i_shift_L1_H1 = i_shift
			
			#if the correlation betwwen L1-H1 is not big enough, discard the event
			if r_L1_H1_max < r_L1_H1_cut: continue
			
			#############    L1-V1 correlation  ############# 
			#get the shifting plotting indices
			i0_V1_shift = np.argmin(np.abs(qtimes_V1-(t_image0-dt_L1_V1)))-1
			if_V1_shift = np.argmin(np.abs(qtimes_V1-(t_imagef+dt_L1_V1)))+1
			N_V1_shift = if_V1_shift-i0_V1_shift+1
			
			#shifting qpower
			q_V1_shift = qpower_V1[:,i0_V1_shift:(if_V1_shift+1)]
			
			#consider all possible shifts and find the one that maximizes the correlation
			Nshifts_L1_V1 = N_V1_shift-N_L1_ref
			r_L1_V1_max = -2
			for i_shift in range(Nshifts_L1_V1):
				#compute the Pearson correlation coefficient
				r, p = stats.pearsonr(q_L1_ref.flatten(), q_V1_shift[:,i_shift:(i_shift+N_L1_ref)].flatten())
				
				if r>r_L1_V1_max:
					r_L1_V1_max = r
					p_L1_V1_max = p
					shift_L1_V1 = qtimes_V1[i0_V1_shift+i_shift]-qtimes_L1[i0_L1_ref]
					i_shift_L1_V1 = i_shift
					
			#if the correlation betwwen L1-V1 is not big enough, discard the event
			f_r_L1_H1_V1_max = weight_r_L1_H1*max(0,r_L1_H1_max) + weight_r_L1_V1*max(0,r_L1_V1_max)
			if f_r_L1_H1_V1_max < f_r_L1_H1_V1_cut: continue
				
			#save values
			#define the time of the event as the center of mass time of a sum of the detector signals
			mass_distr = r_L1_H1_max*(q_L1_ref+q_H1_shift[:,i_shift_L1_H1:(i_shift_L1_H1+N_L1_ref)]) + r_L1_V1_max*(q_L1_ref+q_V1_shift[:,i_shift_L1_V1:(i_shift_L1_V1+N_L1_ref)])
			mass_distr = np.where(mass_distr>np.mean(mass_distr),mass_distr,0)
			t_image_max = np.sum(np.sum(mass_distr,axis=0)*t_L1_ref)/np.sum(mass_distr)
			
			#see that the event has not already triggered
			if np.size(t_image) != 0:
				if np.amin(np.abs(t_image-t_image_max))<t_image_duration:
					#choose the one that maximizes f_r_L1_H1_V1
					i_compare = np.argmin(np.abs(t_image-t_image_max))
					f_r_L1_H1_V1_old = weight_r_L1_H1*max(0,r_L1_H1[i_compare]) + weight_r_L1_V1*max(0,r_L1_V1[i_compare])
					if f_r_L1_H1_V1_old < f_r_L1_H1_V1_max:
						#save new values
						t_image[i_compare] = t_image_max
					
						r_L1_H1[i_compare] = r_L1_H1_max
						p_L1_H1[i_compare] = p_L1_H1_max
						true_shifts_L1_H1[i_compare] = shift_L1_H1
			
						r_L1_V1[i_compare] = r_L1_V1_max
						p_L1_V1[i_compare] = p_L1_V1_max
						true_shifts_L1_V1[i_compare] = shift_L1_V1
					continue
			
			#if the event has not already triggered, save it	
			t_image = np.append(t_image,t_image_max)
					
			r_L1_H1 = np.append(r_L1_H1,r_L1_H1_max)
			p_L1_H1 = np.append(p_L1_H1,p_L1_H1_max)
			true_shifts_L1_H1 = np.append(true_shifts_L1_H1,shift_L1_H1)
			
			r_L1_V1 = np.append(r_L1_V1,r_L1_V1_max)
			p_L1_V1 = np.append(p_L1_V1,p_L1_V1_max)
			true_shifts_L1_V1 = np.append(true_shifts_L1_V1,shift_L1_V1)
			
			print("rL1H1 = %.3f     rL1V1 = %.3f     frL1H1V1=%.3f     t = %.2f"%(r_L1_H1_max,r_L1_V1_max,f_r_L1_H1_V1_max,t_image_max))
			
	#save the data required to reconstruct the correlated images
	image_shiftH1_shiftV1_rL1H1_rL1V1 = np.transpose([t_image,true_shifts_L1_H1,true_shifts_L1_V1,r_L1_H1,r_L1_V1])
	np.savetxt("/lustre/gmorras/Triggered/Shifted_H1m3_V1p3/Text/image_shiftH1_shiftV1_rL1H1_rL1V1_%.2f_%.2f.txt"%(start_data,end_data),image_shiftH1_shiftV1_rL1H1_rL1V1)

	#generate the images
	#if there are no images, continue
	if t_image.size == 0: continue

	#size and overlap of the final plots
	t_image_duration = 0.7
	t_image_overlap = 0.35
	t_image_step = t_image_duration - t_image_overlap

	#Number of pixels in the final plots
	Npixels = 400
	qtimesteps = t_image_duration/Npixels	

        #load images to be represented
	images = t_image
	shiftsH1 = true_shifts_L1_H1
	shiftsV1 = true_shifts_L1_V1
	rsL1H1 = r_L1_H1
	rsL1V1 = r_L1_V1

	for t_det_sample in np.arange(start_data+t_det_overlap/2,end_data-t_det_overlap/2-t_det_step,t_det_step):
		#limits of the chunk to be analyzed
		t_det0 = t_det_sample - t_det_overlap/2
		t_detf = t_det_sample + t_det_step + t_det_overlap/2
		
		#decide if this chunk has to be analyzed and find the images contained within
		dont_analyze_chunk = True
		i_sample = []
		for i_decide in range(images.size):
			if images[i_decide] >= t_det_sample and images[i_decide] < t_det_sample+t_det_step: 
				dont_analyze_chunk = False
				i_sample = np.append(i_sample,i_decide)
			
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
			ifplot_L1 = np.argmin(np.abs(qtimes_L1-t_imagef_L1))+1
			#get the q power
			q_L1 = qpower_L1[:,i0plot_L1:(ifplot_L1+1)]
			t_L1 = qtimes_L1[i0plot_L1:(ifplot_L1+1)]
			N_L1 = ifplot_L1-i0plot_L1
			
			#limits of the H1 image
			t_image0_H1 = t_image0_L1 + shiftsH1[int(i_image)]
			t_imagef_H1 = t_imagef_L1 + shiftsH1[int(i_image)]
			
			#plotting indices
			i0plot_H1 = np.argmin(np.abs(qtimes_H1-t_image0_H1))-1
			ifplot_H1 = i0plot_H1 + N_L1
			#get the q power
			q_H1 = qpower_H1[:,i0plot_H1:(ifplot_H1+1)]
			t_H1 = qtimes_H1[i0plot_H1:(ifplot_H1+1)]
					
			#limits of the V1 image
			t_image0_V1 = t_image0_L1 + shiftsV1[int(i_image)]
			t_imagef_V1 = t_imagef_L1 + shiftsV1[int(i_image)]	
			#plotting indices
			i0plot_V1 = np.argmin(np.abs(qtimes_V1-t_image0_V1))-1
			ifplot_V1 = i0plot_V1 + N_L1
			#get the q power
			q_V1 = qpower_V1[:,i0plot_V1:(ifplot_V1+1)]
			t_V1 = qtimes_V1[i0plot_V1:(ifplot_V1+1)]
					
			#Find the correlated part of the signal
			#corr_qs = rsL1H1[int(i_image)]*(q_L1+q_H1)+ rsL1V1[int(i_image)]*(q_L1+q_V1)		
			normq_L1m = (q_L1-np.mean(q_L1))/np.linalg.norm(q_L1-np.mean(q_L1))
			normq_H1m = (q_H1-np.mean(q_H1))/np.linalg.norm(q_H1-np.mean(q_H1))
			normq_V1m = (q_V1-np.mean(q_V1))/np.linalg.norm(q_V1-np.mean(q_V1))

			corr_qs = rsL1H1[int(i_image)]*(normq_L1m*normq_H1m)+ rsL1V1[int(i_image)]*(normq_L1m*normq_V1m)
			corr_qs_cut = np.mean(corr_qs) + np.std(corr_qs)

			q_L1_signal = np.where(corr_qs>corr_qs_cut,q_L1,1)
			q_H1_signal = np.where(corr_qs>corr_qs_cut,q_H1,1)
			q_V1_signal = np.where(corr_qs>corr_qs_cut,q_V1,1)
			
			#Signal to noise ratio
			dt_L1 = t_L1[1]-t_L1[0]
			dlnf_L1 = np.log(qfreqs_L1[1])-np.log(qfreqs_L1[0])
			SNR_L1 = 2*np.sqrt(max(np.sum(np.sum(q_L1_signal-1,axis=1)*qfreqs_L1)*dt_L1*dlnf_L1,0))
			
			dt_H1 = t_H1[1]-t_H1[0]
			dlnf_H1 = np.log(qfreqs_H1[1])-np.log(qfreqs_H1[0])
			SNR_H1 = 2*np.sqrt(max(np.sum(np.sum(q_H1_signal-1,axis=1)*qfreqs_H1)*dt_H1*dlnf_H1,0))
			
			dt_V1 = t_V1[1]-t_V1[0]
			dlnf_V1 = np.log(qfreqs_V1[1])-np.log(qfreqs_V1[0])
			SNR_V1 = 2*np.sqrt(max(np.sum(np.sum(q_V1_signal-1,axis=1)*qfreqs_V1)*dt_V1*dlnf_V1,0))
			
			#plot
			fig, ((ax1,ax4), (ax2,ax5), (ax3,ax6)) = plt.subplots(3,2,num = 1, figsize = (16,16), clear = True)

			#first image
			ax1.set_xlim(t_image0_L1, t_imagef_L1)
			ax1.set_ylim(flow_bp,fhigh_bp)
			ax1.set_yscale('log')
			ax1.pcolormesh(t_L1, qfreqs_L1, np.sqrt(q_L1), cmap = "gray", vmin = 1.5, vmax = np.amax(np.sqrt(q_L1)), shading='auto')
			ax1.set_xlabel(r"t [s]")
			ax1.set_ylabel(r"f [Hz]")
			ax1.set_title(r"Ligo Livingston")
			
			#second image
			ax2.set_xlim(t_image0_H1, t_imagef_H1)
			ax2.set_ylim(flow_bp,fhigh_bp)
			ax2.set_yscale('log')
			ax2.pcolormesh(t_H1, qfreqs_H1, np.sqrt(q_H1), cmap = "gray", vmin = 1.5, vmax = np.amax(np.sqrt(q_H1)), shading='auto')
			ax2.set_xlabel(r"t [s]")
			ax2.set_ylabel(r"f [Hz]")
			ax2.set_title(r"Ligo Hanford   Pearson $r = %.3f$   $t_{H1}-t_{L1} = %.2f$ms"%(rsL1H1[int(i_image)],1000*shiftsH1[int(i_image)]))
			
			#third image
			ax3.set_xlim(t_image0_V1, t_imagef_V1)
			ax3.set_ylim(flow_bp,fhigh_bp)
			ax3.set_yscale('log')
			ax3.pcolormesh(t_V1, qfreqs_V1, np.sqrt(q_V1), cmap = "gray", vmin = 1.5, vmax = np.amax(np.sqrt(q_V1))+1.5, shading='auto')
			ax3.set_xlabel(r"t [s]")
			ax3.set_ylabel(r"f [Hz]")
			ax3.set_title(r"Virgo   Pearson $r = %.3f$   $t_{V1}-t_{L1} = %.2f$ms"%(rsL1V1[int(i_image)],1000*shiftsV1[int(i_image)]))
			
			#first image
			ax4.set_xlim(t_image0_L1, t_imagef_L1)
			ax4.set_ylim(flow_bp,fhigh_bp)
			ax4.set_yscale('log')
			ax4.pcolormesh(t_L1, qfreqs_L1, np.sqrt(q_L1_signal), cmap = "gray", vmin = 1.5, vmax = np.amax(np.sqrt(q_L1)), shading='auto')
			ax4.set_xlabel(r"t [s]")
			ax4.set_ylabel(r"f [Hz]")
			ax4.set_title(r"Ligo Livingston Correlated:   Estimated SNR = %.1f"%(SNR_L1))
			
			#second image
			ax5.set_xlim(t_image0_H1, t_imagef_H1)
			ax5.set_ylim(flow_bp,fhigh_bp)
			ax5.set_yscale('log')
			ax5.pcolormesh(t_H1, qfreqs_H1, np.sqrt(q_H1_signal), cmap = "gray", vmin = 1.5, vmax = np.amax(np.sqrt(q_H1)), shading='auto')
			ax5.set_xlabel(r"t [s]")
			ax5.set_ylabel(r"f [Hz]")
			ax5.set_title(r"Ligo Hanford Correlated:   Estimated SNR = %.1f"%(SNR_H1))
			
			#third image
			ax6.set_xlim(t_image0_V1, t_imagef_V1)
			ax6.set_ylim(flow_bp,fhigh_bp)
			ax6.set_yscale('log')
			ax6.pcolormesh(t_V1, qfreqs_V1, np.sqrt(q_V1_signal), cmap = "gray", vmin = 1.5, vmax = np.amax(np.sqrt(q_V1)), shading='auto')
			ax6.set_xlabel(r"t [s]")
			ax6.set_ylabel(r"f [Hz]")
			ax6.set_title(r"Virgo Correlated:   Estimated SNR = %.1f"%(SNR_V1))
			
			#save figure
			plt.savefig("/lustre/gmorras/Triggered/Shifted_H1m3_V1p3/Images/L1_H1_V1_%.3f_%.3f.png"%(t_image0_L1,t_imagef_L1), bbox_inches="tight")	

#Runtime
print("Runtime: %s seconds" % (time.time() - start_runtime))


