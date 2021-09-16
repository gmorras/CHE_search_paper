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
from hphc_func import hphc
from hphc_func_15PN import hphc_15PN
import time
from PIL import Image, ImageOps
start_runtime = time.time() 

#Unit conversion
G = 4.30091e-3 #pc*(km/s)^2/M_s
c = 299792.458 #km/s
pc = 3.08567758128e+16 #m	
G_SI = 6.67430e-11 #m*(m/s)^2/kg
c_SI = 299792458 #m/s

#load the data which we are going to analyze
All_DATA = np.loadtxt("/lustre/gmorras/StrainData/GWOSC/O2/DetectorStatus/All_DATA.txt")
begins  = All_DATA[:,0]
ends = All_DATA[:,1]

file_size = 4096
fold_size = 4096*256
min_window_duration = 1000

#shift Cases (repeated cases will generate more images for that case)
AllCases = ["Shifted_H1p2_V1m1"]
Output_path = "/lustre/gmorras/samples4TriggerNN/samples50x50/General_Injected_together/"
SNRtot_rL1H1_rL1V1_path = "/lustre/gmorras/samples4TriggerNN/samples50x50/TriggerInfo/SNRtot_rL1H1_rL1V1_2_cont.txt"

#Final plots specifications
Npixels_plot = 50
t_image_duration_plot = 0.7 
qtimesteps_plot = t_image_duration_plot/Npixels_plot	

#dynamic range
cmin = 1
cmax = 9

#injection specifications	
#SNR range 
SNR_0 = 4
SNR_f = 40

#padding
padding = 5

#filter characteristics
flow_bp = 20 #Hz
fhigh_bp = 800 #Hz
order = 512
flow_noise = 10

#ranges of random parameters
#Mass of the black holes (in solar masses)
m_0 = 0.3
m_f = 50

#eccentricity j = sqrt(et^2-1)
j0_0 = 0.25
j0_f = 5

#maximum velocity
vmax_0 = 0.1
vmax_f = 0.4

#distance to gravitational source
R_pc_0 = 0.5e6 #pc
R_pc_f = 50e6 #pc

#spin magnitude
chi_0 = 0
chi_f = 0.7

#size and overlap of the chunks to analyze
t_det_duration = 80
t_det_overlap = 16
t_det_step = t_det_duration - t_det_overlap

#size and overlap of the correlating plots
t_image_duration = 0.3
t_image_overlap = t_image_duration/2
t_image_step = t_image_duration - t_image_overlap
Npixels = 200
qtimesteps = t_image_duration/Npixels	

#value of the cut for the correlation coefficients
weight_r_L1_H1 = 0.67
f_r_L1_H1_V1_cut = 0.3

weight_r_L1_V1 = 1 - weight_r_L1_H1
r_L1_H1_cut = max(0.2,(f_r_L1_H1_V1_cut-weight_r_L1_V1)/weight_r_L1_H1)

#array into which store injection data
SNRtot_rL1H1_rL1V1 = np.empty((0,3), dtype=float)
print(SNRtot_rL1H1_rL1V1)

#begin the injection and reconstruction
for i_wind in range(begins.size):
	#GPS of the data to plot
	start_data = begins[i_wind]
	end_data = ends[i_wind]
	window_duration = end_data - start_data
	if window_duration < min_window_duration: continue
	
	print("\n\n\nstart_data = %.f    end_data = %.f    duration = %.f"%(start_data,end_data,window_duration))

	if start_data < 1186124882-1: continue
	if start_data > 1187366373-1: continue

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
	
	#consider all cases
	for case in AllCases:
		
		#get the shifts from the case
		if case == "Shifted_H1p2_V1m1":
			allshiftH1 = 2
			allshiftV1 = -1
		if case == "Shifted_H1p1_V1m2":
			allshiftH1 = 1
			allshiftV1 = -2
		if case == "Shifted_H1m2_V1p1":
			allshiftH1 = -2
			allshiftV1 = 1
		if case == "Shifted_H1m1_V1p2":
			allshiftH1 = -1
			allshiftV1 = 2


		print("\n"+case+"  dtH1 = %.1f    dtV1 = %.1f"%(allshiftH1,allshiftV1))
		
		#shift the Hanford time series by an amount allshiftH1:
		hdet_exp_series_H1 = TimeSeries(np.roll(np.array(hdet_exp_series_H1),int(allshiftH1/chunk_delta_t)), delta_t=chunk_delta_t, epoch=chunk_epoch)

		#shift the Virgo time series by an amount allshiftH1:
		hdet_exp_series_V1 = TimeSeries(np.roll(np.array(hdet_exp_series_V1),int(allshiftV1/chunk_delta_t)), delta_t=chunk_delta_t, epoch=chunk_epoch)


		#trigger
		for t_det_sample in np.arange(start_data+t_det_overlap/2,end_data-t_det_overlap/2-t_det_step,t_det_step):
			#matrix with the values of the results
			t_image = -1
			
			#limits of the chunk to be analyzed
			t_det0 = t_det_sample - t_det_overlap/2
			t_detf = t_det_sample + t_det_step + t_det_overlap/2

                        #if the Hanford shift is in this chunk do not analyze it
			if allshiftH1 > 0:
				if start_data+allshiftH1>t_det0 and start_data+allshiftH1<t_detf: continue
			if allshiftH1 < 0:
				if end_data+allshiftH1>t_det0 and end_data+allshiftH1<t_detf: continue

			#if the Virgo shift is in this chunk do not analyze it
			if allshiftV1 > 0:
				if start_data+allshiftV1>t_det0 and start_data+allshiftV1<t_detf: continue
			if allshiftV1 < 0:
				if end_data+allshiftV1>t_det0 and end_data+allshiftV1<t_detf: continue
						
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
			
			#whiten signal and get PSD to estimate SNR of the de injection
			whiten_hdet_series_L1, psd_whiten_L1 = hdet_series_L1.whiten(4,4, return_psd=True, low_frequency_cutoff=flow_noise)
			whiten_hdet_series_H1, psd_whiten_H1 = hdet_series_H1.whiten(4,4, return_psd=True, low_frequency_cutoff=flow_noise)
			whiten_hdet_series_V1, psd_whiten_V1 = hdet_series_V1.whiten(4,4, return_psd=True, low_frequency_cutoff=flow_noise)
			
			#make injection
			injectime = np.random.uniform(low=t_det_sample+padding, high=t_det_sample + t_det_step-padding)
			keep_going = True
			while(keep_going):
				#random parameters
				#internal
				m1 = np.random.uniform(low=m_0, high=m_f)
				m2 = np.random.uniform(low=m_0, high=m_f)
				if m2 > m1: m1, m2 = m2, m1
				j0 = 10**np.random.uniform(low=np.log10(j0_0), high=np.log10(j0_f))
				vmax = np.random.uniform(low=vmax_0, high=vmax_f)
				
				#spins
				chi1 = np.random.uniform(low=chi_0, high=chi_f)
				chi2 = np.random.uniform(low=chi_0, high=chi_f)
				theta1i = np.random.uniform(low=0, high= pi)
				theta2i = np.random.uniform(low=0, high= pi)
				phi1i = np.random.uniform(low=0, high=2*pi)
				phi2i = np.random.uniform(low=0, high=2*pi)
				
				#orientation and location
				#R_pc = np.random.uniform(low=R_pc_0**3, high=R_pc_f**3)**(1/3) #constant volume density
				R_pc = np.random.uniform(low=R_pc_0, high=R_pc_f) #contant radial density
				Phi0 = np.random.uniform(low=0, high=2*pi)
				Theta = np.random.uniform(low=0, high= pi)
				ra = np.random.uniform(low=0, high=2*pi)
				dec = np.random.uniform(low=-0.5*pi, high= 0.5*pi)
				pol = np.random.uniform(low=0, high=2*pi)
				
				#go from sampling parameters to simulation parameters
				et0 = np.sqrt(1+j0**2)
				xi0 = (np.sqrt((et0-1)/(et0+1))*vmax)**3
				
				#check that the spin is in approximation
				S1 = chi1*(m1/m2)
				S2 = chi2*(m2/m1)
				kx0 = -(xi0**(1/3.0))*(S1*np.sin(theta1i)*np.cos(phi1i)+S2*np.sin(theta2i)*np.cos(phi2i))/np.sqrt(et0**2-1)
				ky0 = -(xi0**(1/3.0))*(S1*np.sin(theta1i)*np.sin(phi1i)+S2*np.sin(theta2i)*np.sin(phi2i))/np.sqrt(et0**2-1)
				if kx0**2+ky0**2 > 0.8: continue
					
				#simulation time interval in seconds
				t0_s = -30
				tf_s = 30
				sample_rate = hdet_series_L1.sample_rate

				#simulation time interval array
				dt_eval_s = 1/sample_rate
				t_eval_s = dt_eval_s*np.arange(int(t0_s*sample_rate),int(tf_s*sample_rate))
				N_eval = t_eval_s.size

				#simulation time interval
				t_eval = t_eval_s/((G*(m1+m2)/(c**3))*(pc/1000))
				t0 = t_eval[0]
				tf = t_eval[N_eval-1]

				#distance to gravitational source
				R = ((c**2)/(G*(m1+m2)))*R_pc # GM=c=1 units
				
				#compute gravitational waves
				hplus, hcross  = hphc_15PN(m1,m2,chi1,theta1i,phi1i,chi2,theta2i,phi2i,xi0,et0,Phi0,t0,tf,N_eval,Theta,R)
					
				#fraction to inject
				t_inj0 = injectime - 20
				t_injf = injectime + 20
					
				#proyecting gravitational waves into the detector
				hp_series = TimeSeries(hplus, delta_t = dt_eval_s, epoch = injectime + t_eval_s[0])
				hc_series = TimeSeries(hcross, delta_t = dt_eval_s, epoch = injectime + t_eval_s[0])
				hinj_series_L1 = (Detector("L1").project_wave(hp_series, hc_series, ra, dec, pol, method = "lal")).time_slice(t_inj0,t_injf)
				hinj_series_H1 = (Detector("H1").project_wave(hp_series, hc_series, ra, dec, pol, method = "lal")).time_slice(t_inj0,t_injf)	
				hinj_series_V1 = (Detector("V1").project_wave(hp_series, hc_series, ra, dec, pol, method = "lal")).time_slice(t_inj0,t_injf)	
				
				#zoom to compute SNR
				t_zoom0 = injectime - 0.35
				t_zoomf = injectime + 0.35
					
				#Signal to noise ratio
				h_zoom_L1 = hinj_series_L1.time_slice(t_zoom0, t_zoomf).highpass_fir(flow_bp,order).lowpass_fir(fhigh_bp,order)
				dSNR_L1 = (np.abs(h_zoom_L1.to_frequencyseries())**2)/pycbc.psd.interpolate(psd_whiten_L1, h_zoom_L1.delta_f)
				SNR_L1 = 2*np.sqrt(np.sum(np.array(dSNR_L1))*h_zoom_L1.delta_f)			
				
				h_zoom_H1 = hinj_series_H1.time_slice(t_zoom0, t_zoomf).highpass_fir(flow_bp,order).lowpass_fir(fhigh_bp,order)
				dSNR_H1 = (np.abs(h_zoom_H1.to_frequencyseries())**2)/pycbc.psd.interpolate(psd_whiten_H1, h_zoom_H1.delta_f)
				SNR_H1 = 2*np.sqrt(np.sum(np.array(dSNR_H1))*h_zoom_H1.delta_f)			

				h_zoom_V1 = hinj_series_V1.time_slice(t_zoom0, t_zoomf).highpass_fir(flow_bp,order).lowpass_fir(fhigh_bp,order)
				dSNR_V1 = (np.abs(h_zoom_V1.to_frequencyseries())**2)/pycbc.psd.interpolate(psd_whiten_V1, h_zoom_V1.delta_f)
				SNR_V1 = 2*np.sqrt(np.sum(np.array(dSNR_V1))*h_zoom_V1.delta_f)
					
				#total Signal to noise ratio
				SNR_tot = np.sqrt(SNR_L1**2 + SNR_H1**2 + SNR_V1**2)			

				#Discard if out of SNR range
				if SNR_tot<SNR_0 or SNR_tot>=SNR_f or np.isnan(SNR_tot): continue
				
				#keep the event that satisfies the cuts
				keep_going = False
				print("\nm1 = %.3f    m2 = %.3f    j0 = %.3f    vmax = %.2f    R = %.2f    chi1 = %.2g    chi2 = %.2g    SNR_L1 = %.1f    SNR_H1 = %.1f    SNR_V1 = %.1f    SNR_tot = %.1f" % (m1, m2, j0, vmax, R_pc*1e-6, chi1, chi2, SNR_L1, SNR_H1, SNR_V1,SNR_tot))
			
			#inject the signal into the experimental strain			
			hdet_series_L1 = hdet_series_L1.inject(hinj_series_L1.highpass_fir(flow_noise,order))
			hdet_series_H1 = hdet_series_H1.inject(hinj_series_H1.highpass_fir(flow_noise,order))
			hdet_series_V1 = hdet_series_V1.inject(hinj_series_V1.highpass_fir(flow_noise,order))

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

			#Run the trigger correlation
			for t_image_sample in np.arange(t_det_sample+t_image_overlap/2,t_det_sample+t_det_step-t_image_overlap/2-t_image_step,t_image_step):
				#window limits
				t_image0 = t_image_sample - t_image_overlap/2
				t_imagef = t_image_sample + t_image_step + t_image_overlap/2
				
				#only analize images in which the injection can be seen				
				if np.abs(0.5*(t_image0+t_imagef)-injectime) > 3*t_image_duration: continue
				
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
				
				#to consider an event, impose that the injection is in the image
				if np.abs(t_image_max-injectime) > t_image_duration/2: continue
			
				#see that the event has not already triggered
				if t_image != -1:
					if np.abs(t_image-t_image_max)<t_image_duration:
						#choose the one that maximizes f_r_L1_H1_V1
						f_r_L1_H1_V1_old = weight_r_L1_H1*max(0,r_L1_H1) + weight_r_L1_V1*max(0,r_L1_V1)
						if f_r_L1_H1_V1_old < f_r_L1_H1_V1_max:
							#save new values
							t_image = t_image_max
						
							r_L1_H1 = r_L1_H1_max
							p_L1_H1 = p_L1_H1_max
							true_shifts_L1_H1 = shift_L1_H1
				
							r_L1_V1 = r_L1_V1_max
							p_L1_V1 = p_L1_V1_max
							true_shifts_L1_V1 = shift_L1_V1
						continue
				
				#save the event					
				t_image = t_image_max
						
				r_L1_H1 = r_L1_H1_max
				p_L1_H1 = p_L1_H1_max
				true_shifts_L1_H1 = shift_L1_H1
				
				r_L1_V1 = r_L1_V1_max
				p_L1_V1 = p_L1_V1_max
				true_shifts_L1_V1 = shift_L1_V1
				
				print("rL1H1 = %.3f     rL1V1 = %.3f     frL1H1V1=%.3f     t = %.2f"%(r_L1_H1_max,r_L1_V1_max,f_r_L1_H1_V1_max,t_image_max))
				
			#generate the image of the injection
			#if there are no images, store SNR and continue
			if t_image == -1:
				SNRtot_rL1H1_rL1V1 = np.append(SNRtot_rL1H1_rL1V1, [[SNR_tot, 0, 0]], axis=0) 
				print("Injection not detected :(") 
				continue

			#Store trigger performance
			SNRtot_rL1H1_rL1V1 = np.append(SNRtot_rL1H1_rL1V1, [[SNR_tot, r_L1_H1, r_L1_V1]], axis=0) 
			
			#do the Q transform in the size of the final plots
			qtimes_L1, qfreqs_L1, qpower_L1 = bp_hdet_series_L1.qtransform(delta_t = qtimesteps_plot, logfsteps = Npixels_plot, qrange=(8,8), frange=(flow_bp,fhigh_bp))
			qtimes_H1, qfreqs_H1, qpower_H1 = bp_hdet_series_H1.qtransform(delta_t = qtimesteps_plot, logfsteps = Npixels_plot, qrange=(8,8), frange=(flow_bp,fhigh_bp))
			qtimes_V1, qfreqs_V1, qpower_V1 = bp_hdet_series_V1.qtransform(delta_t = qtimesteps_plot, logfsteps = Npixels_plot, qrange=(8,8), frange=(flow_bp,fhigh_bp))

			#limits of the L1 image
			t_image0_L1 = t_image - t_image_duration_plot/2
			t_imagef_L1 = t_image + t_image_duration_plot/2	
			#plotting indices
			i0plot_L1 = np.argmin(np.abs(qtimes_L1-t_image0_L1))-1
			ifplot_L1 = i0plot_L1 + Npixels_plot
			#quatity to plot
			lnqpower_plot_L1 = np.log(qpower_L1[:,i0plot_L1:ifplot_L1])
			#array to finally plot
			arr_plot_L1 = np.where(lnqpower_plot_L1 < cmin, cmin, lnqpower_plot_L1)
			arr_plot_L1 = np.where(arr_plot_L1 > cmax, cmax, arr_plot_L1)				
			#transform to uint8
			arr_plot_L1 = np.uint8(255*(arr_plot_L1-cmin)/(cmax-cmin))			
			
			#limits of the H1 image
			t_image0_H1 = t_image0_L1 + true_shifts_L1_H1
			t_imagef_H1 = t_imagef_L1 + true_shifts_L1_H1		
			#plotting indices
			i0plot_H1 = np.argmin(np.abs(qtimes_H1-t_image0_H1))-1
			ifplot_H1 = i0plot_H1 + Npixels_plot
			#quatity to plot
			lnqpower_plot_H1 = np.log(qpower_H1[:,i0plot_H1:ifplot_H1])
			#array to finally plot
			arr_plot_H1 = np.where(lnqpower_plot_H1 < cmin, cmin, lnqpower_plot_H1)
			arr_plot_H1 = np.where(arr_plot_H1 > cmax, cmax, arr_plot_H1)				
			#transform to uint8
			arr_plot_H1 = np.uint8(255*(arr_plot_H1-cmin)/(cmax-cmin))		
								
			#limits of the V1 image
			t_image0_V1 = t_image0_L1 + true_shifts_L1_V1
			t_imagef_V1 = t_imagef_L1 + true_shifts_L1_V1	
			#plotting indices
			i0plot_V1 = np.argmin(np.abs(qtimes_V1-t_image0_V1))-1
			ifplot_V1 = i0plot_V1 + Npixels_plot
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
			im_plot.save(Output_path+case+"_%.3f_%.1f.png"%(injectime,SNR_tot))
		
		#unshift the Hanford time
		hdet_exp_series_H1 = TimeSeries(np.roll(np.array(hdet_exp_series_H1),-int(allshiftH1/chunk_delta_t)), delta_t=chunk_delta_t, epoch=chunk_epoch)

		#unshift the Virgo time:
		hdet_exp_series_V1 = TimeSeries(np.roll(np.array(hdet_exp_series_V1),-int(allshiftV1/chunk_delta_t)), delta_t=chunk_delta_t, epoch=chunk_epoch)
	
	np.savetxt(SNRtot_rL1H1_rL1V1_path,SNRtot_rL1H1_rL1V1)

#Runtime
print("Runtime: %s seconds" % (time.time() - start_runtime))


