# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 11:19:22 2018
frb_mclc.py
script to do Monte Carlo simulation of FRB 1D light curves and test 
statistical parameters that might be fed into the decision tree for identifying
candidates in the RealFast pipeline
Want to look at rms, ratio of original rms to clipped rms, skew, and kurtosis
Add in smoothness as a statistical measure
@author: jdlin
"""

import numpy as np
from scipy import signal
from scipy import stats
import matplotlib.pyplot as plt
from astropy.stats import median_absolute_deviation as MAD

#some tests from playing with the rms in rms_play.py
def calc_rms(in_arr):
    out_rms = np.sqrt(1.0/len(in_arr)*np.sum(in_arr**2))
    return out_rms

def smooth_calc(in_arr):
    #smooth_measure = np.std(np.diff(in_arr)) / np.abs(np.mean(np.diff(in_arr)))
    smooth_measure= np.std(np.diff(in_arr))
    return smooth_measure

def get_avgs(in_arr1,in_arr2,in_arr3,in_arr4,in_arr5,in_arr6,in_arr7,in_arr8):
    #find mean, median, std, max, and min of the input arrays
    #input arrays should be:
    #in_arr1 = rms, in_arr2 = rms ratio, in_arr3 = skew, in_arr4 = kurtosis, 
    #in_arr5 = smoothness, in_arr6 = mean to clippped rms ratio
    #in_arr7 = peak/rms, in_arr8 = peak/median_absolute_deviation
    out_arr = np.zeros((8,5))
    #rms
    out_arr[0,0]=np.mean(in_arr1)
    out_arr[0,1]=np.median(in_arr1)
    out_arr[0,2]=np.std(in_arr1)
    out_arr[0,3]=np.max(in_arr1)
    out_arr[0,4]=np.min(in_arr1)
    #rms ratio
    out_arr[1,0]=np.mean(in_arr2)
    out_arr[1,1]=np.median(in_arr2)
    out_arr[1,2]=np.std(in_arr2)
    out_arr[1,3]=np.max(in_arr2)
    out_arr[1,4]=np.min(in_arr2)
    #skew
    out_arr[2,0]=np.mean(in_arr3)
    out_arr[2,1]=np.median(in_arr3)
    out_arr[2,2]=np.std(in_arr3)
    out_arr[2,3]=np.max(in_arr3)
    out_arr[2,4]=np.min(in_arr3)
    #kurtosis
    out_arr[3,0]=np.mean(in_arr4)
    out_arr[3,1]=np.median(in_arr4)
    out_arr[3,2]=np.std(in_arr4)
    out_arr[3,3]=np.max(in_arr4)
    out_arr[3,4]=np.min(in_arr4)
    #smoothness
    out_arr[4,0]=np.mean(in_arr5)
    out_arr[4,1]=np.median(in_arr5)
    out_arr[4,2]=np.std(in_arr5)
    out_arr[4,3]=np.max(in_arr5)
    out_arr[4,4]=np.min(in_arr5)
    #mean to clipped rms ratio
    out_arr[5,0]=np.mean(in_arr6)
    out_arr[5,1]=np.median(in_arr6)
    out_arr[5,2]=np.std(in_arr6)
    out_arr[5,3]=np.max(in_arr6)
    out_arr[5,4]=np.min(in_arr6)
    #peak to rms
    out_arr[6,0]=np.mean(in_arr7)
    out_arr[6,1]=np.median(in_arr7)
    out_arr[6,2]=np.std(in_arr7)
    out_arr[6,3]=np.max(in_arr7)
    out_arr[6,4]=np.min(in_arr7)
    #peak to meadian absolute deviation
    out_arr[7,0]=np.mean(in_arr8)
    out_arr[7,1]=np.median(in_arr8)
    out_arr[7,2]=np.std(in_arr8)
    out_arr[7,3]=np.max(in_arr8)
    out_arr[7,4]=np.min(in_arr8)
    return out_arr

#set rms threshold for clipping light curves
rms_threshold = 2.0

#vanilla FRB: fairly bright single element pulse at center of time series
vanilla_rms = []
vanilla_rmsratio = []
vanilla_skew = []
vanilla_kurtosis = []
vanilla_smooth = []
vanilla_meancliprms = []
vanilla_peaktorms = []
vanilla_peaktomad = []

for i in range(0,1000):
    frb_vanilla = np.random.randn(31)/100.0
    #set middle of time serues as FRB detection
    frb_vanilla[15] = 0.1+abs(np.random.randn(1)/20.0)
    vanilla_rms.append(calc_rms(frb_vanilla))
    vanilla_rmsratio.append(vanilla_rms[i]/calc_rms(frb_vanilla[np.where(frb_vanilla<rms_threshold*vanilla_rms[i])]))
    vanilla_skew.append(stats.skew(frb_vanilla))
    vanilla_kurtosis.append(stats.kurtosis(frb_vanilla))
    vanilla_smooth.append(smooth_calc(frb_vanilla))
    vanilla_meancliprms.append(np.mean(frb_vanilla)/calc_rms(frb_vanilla[np.where(frb_vanilla<rms_threshold*vanilla_rms[i])]))
    vanilla_peaktorms.append(max(frb_vanilla)/calc_rms(frb_vanilla[np.where(frb_vanilla<max(frb_vanilla))]))
    vanilla_peaktomad.append(max(frb_vanilla)/MAD(frb_vanilla[np.where(frb_vanilla<max(frb_vanilla))]))

#plot ressults
plt.figure(1,(7.5,8.5))
plt.subplot(5,2,1)
plt.plot(vanilla_rms,'.',markersize=2.0)
plt.title('Vanilla FRB')
plt.ylabel('RMS')
plt.subplot(5,2,2)
plt.plot(vanilla_rmsratio,'.',markersize=2.0)
plt.ylabel('RMS Ratio')
plt.subplot(5,2,3)
plt.plot(vanilla_skew,'.',markersize=2.0)
plt.ylabel('Skewness')
plt.subplot(5,2,4)
plt.plot(vanilla_kurtosis,'.',markersize=2.0)
plt.ylabel('Kurtosis')
plt.subplot(5,2,5)
plt.plot(vanilla_smooth,'.',markersize=2.0)
plt.ylabel('Smoothness')
plt.subplot(5,2,6)
plt.plot(vanilla_meancliprms,'.',markersize=2.0)
plt.ylabel('Mean/Clipped RMS')
plt.subplot(5,2,7)
plt.plot(vanilla_peaktorms,'.',markersize=2.0)
plt.ylabel('Peak/RMS')
plt.subplot(5,2,8)
plt.plot(vanilla_peaktomad,'.',markersize=2.0)
plt.ylabel('Peak/MAD')
plt.subplot(5,1,5)
plt.plot(frb_vanilla)
plt.savefig('/home/jlinford/FRBs/FRB_MCLC/FRB_Vanilla.png')
plt.savefig('/home/jlinford/FRBs/FRB_MCLC/FRB_Vanilla.eps',format='eps',dpi=200)
plt.close()

#get some average values and stds
#vanilla_rms_mean = np.mean(vanilla_rms)
#vanilla_rms_median = np.median(vanilla_rms)
#vanilla_rms_std = np.std(vanilla_rms)
#vanilla_rmsratio_mean = np.mean(vanilla_rmsratio)
#vanilla_rmsratio_median = np.median(vanilla_rmsratio)
#vanilla_rmsratio_std = np.std(vanilla_rmsratio)
#vanilla_skew_mean = np.mean(vanilla_skew)
#vanilla_skew_median = np.median(vanilla_skew)
#vanilla_skew_std = np.std(vanilla_skew)
#vanilla_kurtosis_mean = np.mean(vanilla_kurtosis)
#vanilla_kurtosis_median = np.median(vanilla_kurtosis)
#vanilla_kurtosis_std = np.std(vanilla_kurtosis)

vanilla_avgs = get_avgs(vanilla_rms,vanilla_rmsratio,vanilla_skew,vanilla_kurtosis,vanilla_smooth,vanilla_meancliprms,vanilla_peaktorms,vanilla_peaktomad)


#early vanilla FRB
early_rms = []
early_rmsratio = []
early_skew = []
early_kurtosis = []
early_smooth = []
early_meancliprms = []
early_peaktorms = []
early_peaktomad = []

for i in range(0,500):
    frb_early = np.random.randn(31)/100.0
    #inject FRB pulse early in the time series - simulates FRB showing up in first segment
    frb_early[np.random.randint(2,10,size=1)] = 0.1+abs(np.random.randn(1)/20.0)
    early_rms.append(calc_rms(frb_early))
    early_rmsratio.append(early_rms[i]/calc_rms(frb_early[np.where(frb_early<rms_threshold*early_rms[i])]))
    early_skew.append(stats.skew(frb_early))
    early_kurtosis.append(stats.kurtosis(frb_early))
    early_smooth.append(smooth_calc(frb_early))
    early_meancliprms.append(np.mean(frb_early)/calc_rms(frb_early[np.where(frb_early<rms_threshold*early_rms[i])]))
    early_peaktorms.append(max(frb_early)/calc_rms(frb_early[np.where(frb_early<max(frb_early))]))
    early_peaktomad.append(max(frb_early)/MAD(frb_early[np.where(frb_early<max(frb_early))]))

#plot ressults
plt.figure(2,(7.5,8.5))
plt.subplot(5,2,1)
plt.plot(early_rms,'.',markersize=2.0)
plt.title('Early FRB')
plt.ylabel('RMS')
plt.subplot(5,2,2)
plt.plot(early_rmsratio,'.',markersize=2.0)
plt.ylabel('RMS Ratio')
plt.subplot(5,2,3)
plt.plot(early_skew,'.',markersize=2.0)
plt.ylabel('Skewness')
plt.subplot(5,2,4)
plt.plot(early_kurtosis,'.',markersize=2.0)
plt.ylabel('Kurtosis')
plt.subplot(5,2,5)
plt.plot(early_smooth,'.',markersize=2.0)
plt.ylabel('Smoothness')
plt.subplot(5,2,6)
plt.plot(early_meancliprms,'.',markersize=2.0)
plt.ylabel('Mean/Clipped RMS')
plt.subplot(5,2,7)
plt.plot(early_peaktorms,'.',markersize=2.0)
plt.ylabel('Peak/RMS')
plt.subplot(5,2,8)
plt.plot(early_peaktomad,'.',markersize=2.0)
plt.ylabel('Peak/MAD')
plt.subplot(5,1,5)
plt.plot(frb_early)
plt.savefig('/home/jlinford/FRBs/FRB_MCLC/FRB_Early.png')
plt.savefig('/home/jlinford/FRBs/FRB_MCLC/FRB_Early.eps',format='eps',dpi=200)
plt.close()

early_frb_avgs = get_avgs(early_rms,early_rmsratio,early_skew,early_kurtosis,early_smooth,early_meancliprms,early_peaktorms,early_peaktomad)

#late vanilla FRB
late_rms = []
late_rmsratio = []
late_skew = []
late_kurtosis = []
late_smooth = []
late_meancliprms = []
late_peaktorms = []
late_peaktomad = []

for i in range(0,500):
    frb_late = np.random.randn(31)/100.0
    #inject FRB pulse early in the time series - simulates FRB showing up in last segment
    frb_late[np.random.randint(21,29,size=1)] = 0.1+abs(np.random.randn(1)/20.0)
    late_rms.append(calc_rms(frb_late))
    late_rmsratio.append(late_rms[i]/calc_rms(frb_late[np.where(frb_late<rms_threshold*late_rms[i])]))
    late_skew.append(stats.skew(frb_late))
    late_kurtosis.append(stats.kurtosis(frb_late))
    late_smooth.append(smooth_calc(frb_late))
    late_meancliprms.append(np.mean(frb_late)/calc_rms(frb_late[np.where(frb_late<rms_threshold*late_rms[i])]))
    late_peaktorms.append(max(frb_late)/calc_rms(frb_late[np.where(frb_late<max(frb_late))]))
    late_peaktomad.append(max(frb_late)/MAD(frb_late[np.where(frb_late<max(frb_late))]))

#plot ressults
plt.figure(3,(7.5,8.5))
plt.subplot(5,2,1)
plt.plot(late_rms,'.',markersize=2.0)
plt.title('Late FRB')
plt.ylabel('RMS')
plt.subplot(5,2,2)
plt.plot(late_rmsratio,'.',markersize=2.0)
plt.ylabel('RMS Ratio')
plt.subplot(5,2,3)
plt.plot(late_skew,'.',markersize=2.0)
plt.ylabel('Skewness')
plt.subplot(5,2,4)
plt.plot(late_kurtosis,'.',markersize=2.0)
plt.ylabel('Kurtosis')
plt.subplot(5,2,5)
plt.plot(late_smooth,'.',markersize=2.0)
plt.ylabel('Smoothness')
plt.subplot(5,2,6)
plt.plot(late_meancliprms,'.',markersize=2.0)
plt.ylabel('Mean/Clipped RMS')
plt.subplot(5,2,7)
plt.plot(late_peaktorms,'.',markersize=2.0)
plt.ylabel('Peak/RMS')
plt.subplot(5,2,8)
plt.plot(late_peaktomad,'.',markersize=2.0)
plt.ylabel('Peak/MAD')
plt.subplot(5,1,5)
plt.plot(frb_late)
plt.savefig('/home/jlinford/FRBs/FRB_MCLC/FRB_Late.png')
plt.savefig('/home/jlinford/FRBs/FRB_MCLC/FRB_Late.eps',format='eps',dpi=200)
plt.close()

late_frb_avgs = get_avgs(late_rms,late_rmsratio,late_skew,late_kurtosis,late_smooth,late_meancliprms,late_peaktorms,late_peaktomad)

#fast rise, linear decay
lineardecay_rms = []
lineardecay_rmsratio = []
lineardecay_skew = []
lineardecay_kurtosis = []
lineardecay_smooth = []
lineardecay_meancliprms = []
lineardecay_peaktorms = []
lineardecay_peaktomad = []

for i in range(0,1000):
    frb_lineardecay = np.random.randn(31)/100.0
    frb_lineardecay[15]= 0.1+abs(np.random.randn(1)/20.0)+frb_lineardecay[15]
    lin_width = np.random.randint(8,15,size=1)
    lin_decay = frb_lineardecay[15]/float(lin_width) * (15 + lin_width - np.arange(16,15+lin_width))
    frb_lineardecay[16:15+int(lin_width)]=frb_lineardecay[16:15+int(lin_width)]+lin_decay
    lineardecay_rms.append(calc_rms(frb_lineardecay))
    lineardecay_rmsratio.append(lineardecay_rms[i]/calc_rms(frb_lineardecay[np.where(frb_lineardecay<rms_threshold*lineardecay_rms[i])]))
    lineardecay_skew.append(stats.skew(frb_lineardecay))
    lineardecay_kurtosis.append(stats.kurtosis(frb_lineardecay))
    lineardecay_smooth.append(smooth_calc(frb_lineardecay))
    lineardecay_meancliprms.append(np.mean(frb_lineardecay)/calc_rms(frb_lineardecay[np.where(frb_lineardecay<rms_threshold*lineardecay_rms[i])]))
    lineardecay_peaktorms.append(max(frb_lineardecay)/calc_rms(frb_lineardecay[np.where(frb_lineardecay<max(frb_lineardecay))]))
    lineardecay_peaktomad.append(max(frb_lineardecay)/MAD(frb_lineardecay[np.where(frb_lineardecay<max(frb_lineardecay))]))

#plot results
plt.figure(4,(7.5,8.5))
plt.subplot(5,2,1)
plt.plot(lineardecay_rms,'.',markersize=2.0)
plt.title('Linear Decay')
plt.ylabel('RMS')
plt.subplot(5,2,2)
plt.plot(lineardecay_rmsratio,'.',markersize=2.0)
plt.ylabel('RMS Ratio')
plt.subplot(5,2,3)
plt.plot(lineardecay_skew,'.',markersize=2.0)
plt.ylabel('Skewness')
plt.subplot(5,2,4)
plt.plot(lineardecay_kurtosis,'.',markersize=2.0)
plt.ylabel('Kurtosis')
plt.subplot(5,2,5)
plt.plot(lineardecay_smooth,'.',markersize=2.0)
plt.ylabel('Smoothness')
plt.subplot(5,2,6)
plt.plot(lineardecay_meancliprms,'.',markersize=2.0)
plt.ylabel('Mean/Clipped RMS')
plt.subplot(5,2,7)
plt.plot(lineardecay_peaktorms,'.',markersize=2.0)
plt.ylabel('Peak/RMS')
plt.subplot(5,2,8)
plt.plot(lineardecay_peaktomad,'.',markersize=2.0)
plt.subplot(5,1,5)
plt.plot(frb_lineardecay)
plt.savefig('/home/jlinford/FRBs/FRB_MCLC/FRB_LinearDecay.png')
plt.savefig('/home/jlinford/FRBs/FRB_MCLC/FRB_LinearDecay.eps',format='eps',dpi=200)
plt.close()

lineardecay_avgs = get_avgs(lineardecay_rms,lineardecay_rmsratio,lineardecay_skew,lineardecay_kurtosis,lineardecay_smooth,lineardecay_meancliprms,lineardecay_peaktorms,lineardecay_peaktomad)

#fast rise, exponential decay
expdecay_rms = []
expdecay_rmsratio = []
expdecay_skew= []
expdecay_kurtosis = []
expdecay_smooth= []
expdecay_meancliprms = []
expdecay_peaktorms = []
expdecay_peaktomad = []

for i in range(0,1000):
    frb_expdecay = np.random.randn(31)/100.0
    frb_expdecay[15]= 0.1+abs(np.random.randn(1)/20.0)
    exp_decay_width = np.random.randint(8,15,size=1)
    exp_decay_alpha = (1.0/float(exp_decay_width))*(np.log(frb_expdecay[15])-np.log(0.01))
    exp_decay_A = frb_expdecay[15]*np.exp(exp_decay_alpha*15.0)
    frb_expdecay[16:15+int(exp_decay_width)] = frb_expdecay[16:15+int(exp_decay_width)] + exp_decay_A*np.exp(-1.0*exp_decay_alpha*np.arange(16,15+exp_decay_width))
    expdecay_rms.append(calc_rms(frb_expdecay))
    expdecay_rmsratio.append(expdecay_rms[i]/calc_rms(frb_expdecay[np.where(frb_expdecay<rms_threshold*expdecay_rms[i])]))
    expdecay_skew.append(stats.skew(frb_expdecay))
    expdecay_kurtosis.append(stats.kurtosis(frb_expdecay))
    expdecay_smooth.append(smooth_calc(frb_expdecay))
    expdecay_meancliprms.append(np.mean(frb_expdecay)/calc_rms(frb_expdecay[np.where(frb_expdecay<rms_threshold*expdecay_rms[i])]))
    expdecay_peaktorms.append(max(frb_expdecay)/calc_rms(frb_expdecay[np.where(frb_expdecay<max(frb_expdecay))]))
    expdecay_peaktomad.append(max(frb_expdecay)/MAD(frb_expdecay[np.where(frb_expdecay<max(frb_expdecay))]))

#plot results
plt.figure(5,(7.5,8.5))
plt.subplot(5,2,1)
plt.plot(expdecay_rms,'.',markersize=2.0)
plt.title('Exponential Decay')
plt.ylabel('RMS')
plt.subplot(5,2,2)
plt.plot(expdecay_rmsratio,'.',markersize=2.0)
plt.ylabel('RMS Ratio')
plt.subplot(5,2,3)
plt.plot(expdecay_skew,'.',markersize=2.0)
plt.ylabel('Skewness')
plt.subplot(5,2,4)
plt.plot(expdecay_kurtosis,'.',markersize=2.0)
plt.ylabel('Kurtosis')
plt.subplot(5,2,5)
plt.plot(expdecay_smooth,'.',markersize=2.0)
plt.ylabel('Smoothness')
plt.subplot(5,2,6)
plt.plot(expdecay_meancliprms,'.',markersize=2.0)
plt.ylabel('Mean/Clipped RMS')
plt.subplot(5,2,7)
plt.plot(expdecay_peaktorms,'.',markersize=2.0)
plt.ylabel('Prak/RMS')
plt.subplot(5,2,8)
plt.plot(expdecay_peaktomad,'.',markersize=2.0)
plt.subplot(5,1,5)
plt.plot(frb_expdecay)
plt.savefig('/home/jlinford/FRBs/FRB_MCLC/FRB_ExpDecay.png')
plt.savefig('/home/jlinford/FRBs/FRB_MCLC/FRB_ExpDecay.eps',formtat='eps',dpi=200)
plt.close()

expdecay_avgs = get_avgs(expdecay_rms,expdecay_rmsratio,expdecay_skew,expdecay_kurtosis,expdecay_smooth,expdecay_meancliprms,expdecay_peaktorms,expdecay_peaktomad)

#narrow Gaussian
narrowGauss_rms = []
narrowGauss_rmsratio = []
narrowGauss_skew = []
narrowGauss_kurtosis = []
narrowGauss_smooth = []
narrowGauss_meancliprms = []
narrowGauss_peaktorms = []
narrowGauss_peaktomad = []

for i in range(0,1000):
    frb_narrowGauss = np.random.randn(31)/100.0
    g_pls = signal.gaussian(7,std=1.0)*(0.1+np.random.randn(1)/20.0)
    frb_narrowGauss[12:19] = frb_narrowGauss[12:19]+g_pls
    narrowGauss_rms.append(calc_rms(frb_narrowGauss))
    narrowGauss_rmsratio.append(narrowGauss_rms[i]/calc_rms(frb_narrowGauss[np.where(frb_narrowGauss<rms_threshold*narrowGauss_rms[i])]))
    narrowGauss_skew.append(stats.skew(frb_narrowGauss))
    narrowGauss_kurtosis.append(stats.kurtosis(frb_narrowGauss))
    narrowGauss_smooth.append(smooth_calc(frb_narrowGauss))
    narrowGauss_meancliprms.append(np.mean(frb_narrowGauss)/calc_rms(frb_narrowGauss[np.where(frb_narrowGauss<rms_threshold*narrowGauss_rms[i])]))
    narrowGauss_peaktorms.append(max(frb_narrowGauss)/calc_rms(frb_narrowGauss[np.where(frb_narrowGauss<max(frb_narrowGauss))]))
    narrowGauss_peaktomad.append(max(frb_narrowGauss)/MAD(frb_narrowGauss[np.where(frb_narrowGauss<max(frb_narrowGauss))]))

#plot results
plt.figure(6,(7.5,8.5))
plt.subplot(5,2,1)
plt.plot(narrowGauss_rms,'.',markersize=2.0)
plt.title('Narrow Gaussian')
plt.ylabel('RMS')
plt.subplot(5,2,2)
plt.plot(narrowGauss_rmsratio,'.',markersize=2.0)
plt.ylabel('RMS Ratio')
plt.subplot(5,2,3)
plt.plot(narrowGauss_skew,'.',markersize=2.0)
plt.ylabel('Skewness')
plt.subplot(5,2,4)
plt.plot(narrowGauss_kurtosis,'.',markersize=2.0)
plt.ylabel('Kurtosis')
plt.subplot(5,2,5)
plt.plot(narrowGauss_smooth,'.',markersize=2.0)
plt.ylabel('Smoothness')
plt.subplot(5,2,6)
plt.plot(narrowGauss_meancliprms,'.',markersize=2.0)
plt.ylabel('Mean/Cipped RMS')
plt.subplot(5,2,7)
plt.plot(narrowGauss_peaktorms,'.',markersize=2.0)
plt.ylabel('Peak/RMS')
plt.subplot(5,2,8)
plt.plot(narrowGauss_peaktomad,'.',markersize=2.0)
plt.subplot(5,1,5)
plt.plot(frb_narrowGauss)
plt.savefig('/home/jlinford/FRBs/FRB_MCLC/FRB_NarrowGauss.png')
plt.savefig('/home/jlinford/FRBs/FRB_MCLC/FRB_NarrowGauss.eps',format='eps',dpi=200)
plt.close()

narrowGauss_avgs = get_avgs(narrowGauss_rms,narrowGauss_rmsratio,narrowGauss_skew,narrowGauss_kurtosis,narrowGauss_smooth,narrowGauss_meancliprms,narrowGauss_peaktorms,narrowGauss_peaktomad)

#broad Gaussian
broadGauss_rms = []
broadGauss_rmsratio = []
broadGauss_skew = []
broadGauss_kurtosis = []
broadGauss_smooth = []
broadGauss_meancliprms = []
broadGauss_peaktorms = []
broadGauss_peaktomad = []

for i in range(0,1000):
    frb_broadGauss = np.random.randn(31)/100.0
    b_pls = signal.gaussian(21,std=3.0)*(0.1+np.random.randn(1)/20.0)
    frb_broadGauss[5:26] = frb_broadGauss[5:26]+b_pls
    broadGauss_rms.append(calc_rms(frb_broadGauss))
    broadGauss_rmsratio.append(broadGauss_rms[i]/calc_rms(frb_broadGauss[np.where(frb_broadGauss<rms_threshold*broadGauss_rms[i])]))
    broadGauss_skew.append(stats.skew(frb_broadGauss))
    broadGauss_kurtosis.append(stats.kurtosis(frb_broadGauss))
    broadGauss_smooth.append(smooth_calc(frb_broadGauss))
    broadGauss_meancliprms.append(np.mean(frb_broadGauss)/calc_rms(frb_broadGauss[np.where(frb_broadGauss<rms_threshold*broadGauss_rms[i])]))
    broadGauss_peaktorms.append(max(frb_broadGauss)/calc_rms(frb_broadGauss[np.where(frb_broadGauss<max(frb_broadGauss))]))
    broadGauss_peaktomad.append(max(frb_broadGauss)/MAD(frb_broadGauss[np.where(frb_broadGauss<max(frb_broadGauss))]))

#plot results
plt.figure(7,(7.5,8.5))
plt.subplot(5,2,1)
plt.plot(broadGauss_rms,'.',markersize=2.0)
plt.title('Broad Gaussian')
plt.ylabel('RMS')
plt.subplot(5,2,2)
plt.plot(broadGauss_rmsratio,'.',markersize=2.0)
plt.ylabel('RMS Ratio')
plt.subplot(5,2,3)
plt.plot(broadGauss_skew,'.',markersize=2.0)
plt.ylabel('Skewness')
plt.subplot(5,2,4)
plt.plot(broadGauss_kurtosis,'.',markersize=2.0)
plt.ylabel('Kurtosis')
plt.subplot(5,2,5)
plt.plot(broadGauss_smooth,'.',markersize=2.0)
plt.ylabel('Smoothness')
plt.subplot(5,2,6)
plt.plot(broadGauss_meancliprms,'.',markersize=2.0)
plt.ylabel('Mean/Clipped RMS')
plt.subplot(5,2,7)
plt.plot(broadGauss_peaktorms,'.',markersize=2.0)
plt.ylabel('Peak/RMS')
plt.subplot(5,2,8)
plt.plot(broadGauss_peaktomad,'.',markersize=2.0)
plt.ylabel('Peak/MAD')
plt.subplot(5,1,5)
plt.plot(frb_broadGauss)
plt.savefig('/home/jlinford/FRBs/FRB_MCLC/FRB_BroadGauss.png')
plt.savefig('/home/jlinford/FRBs/FRB_MCLC/FRB_BroadGauss.eps',format='eps',dpi=200)
plt.close()

broadGauss_avgs = get_avgs(broadGauss_rms,broadGauss_rmsratio,broadGauss_skew,broadGauss_kurtosis,broadGauss_smooth,broadGauss_meancliprms,broadGauss_peaktorms,broadGauss_peaktomad)


#okay, that should be enough of the good FRB models
#now need to look at the bad data that may look like candidates

#periodic bad data
badperiodic_rms = []
badperiodic_rmsratio = []
badperiodic_skew = []
badperiodic_kurtosis = []
badperiodic_smooth = []
badperiodic_meancliprms = []
badperiodic_peaktorms = []
badperiodic_peaktomad = []

for i in range(0,1000):
    bad_periodic = np.sin(np.pi/15.0*np.arange(-15,16))*np.random.randn(1)/20.0 + np.random.randn(31)/100.0
    badperiodic_rms.append(calc_rms(bad_periodic))
    badperiodic_rmsratio.append(badperiodic_rms[i]/calc_rms(bad_periodic[np.where(bad_periodic<rms_threshold*badperiodic_rms[i])]))
    badperiodic_skew.append(stats.skew(bad_periodic))
    badperiodic_kurtosis.append(stats.kurtosis(bad_periodic))
    badperiodic_smooth.append(smooth_calc(bad_periodic))
    badperiodic_meancliprms.append(np.mean(bad_periodic)/calc_rms(bad_periodic[np.where(bad_periodic<rms_threshold*badperiodic_rms[i])]))
    badperiodic_peaktorms.append(max(bad_periodic)/calc_rms(bad_periodic[np.where(bad_periodic<max(bad_periodic))]))
    badperiodic_peaktomad.append(max(bad_periodic)/MAD(bad_periodic[np.where(bad_periodic<max(bad_periodic))]))
    
#plot results
plt.figure(8,(7.5,8.5))
plt.subplot(5,2,1)
plt.plot(badperiodic_rms,'.',markersize=2.0)
plt.title('Periodic Signal')
plt.ylabel('RMS')
plt.subplot(5,2,2)
plt.plot(badperiodic_rmsratio,'.',markersize=2.0)
plt.ylabel('RMS Ratio')
plt.subplot(5,2,3)
plt.plot(badperiodic_skew,'.',markersize=2.0)
plt.ylabel('Skewness')
plt.subplot(5,2,4)
plt.plot(badperiodic_kurtosis,'.',markersize=2.0)
plt.ylabel('Kurtosis')
plt.subplot(5,2,5)
plt.plot(badperiodic_smooth,'.',markersize=2.0)
plt.ylabel('Smoothness')
plt.subplot(5,2,6)
plt.plot(badperiodic_meancliprms,'.',markersize=2.0)
plt.ylabel('Mean/Clipped RMS')
plt.subplot(5,2,7)
plt.plot(badperiodic_peaktorms,'.',markersize=2.0)
plt.ylabel('Peak/RMS')
plt.subplot(5,2,8)
plt.plot(badperiodic_peaktomad,'.',markersize=2.0)
plt.subplot(5,1,5)
plt.plot(bad_periodic)
plt.savefig('/home/jlinford/FRBs/FRB_MCLC/Bad_Periodic.png')
plt.savefig('/home/jlinford/FRBs/FRB_MCLC/Bad_Periodic.eps',format='eps',dpi=200)
plt.close()

badperiodic_avgs = get_avgs(badperiodic_rms,badperiodic_rmsratio,badperiodic_skew,badperiodic_kurtosis,badperiodic_smooth,badperiodic_meancliprms,badperiodic_peaktorms,badperiodic_peaktomad)

#noisy data
badnoisy_rms = []
badnoisy_rmsratio = []
badnoisy_skew = []
badnoisy_kurtosis = []
badnoisy_smooth = []
badnoisy_meancliprms = []
badnoisy_peaktorms = []
badnoisy_peaktomad = []

for i in range(0,1000):
    bad_noisy = np.random.randn(31)/10.0
    badnoisy_rms.append(calc_rms(bad_noisy))
    badnoisy_rmsratio.append(badnoisy_rms[i]/calc_rms(bad_noisy[np.where(bad_noisy<rms_threshold*badnoisy_rms[i])]))
    badnoisy_skew.append(stats.skew(bad_noisy))
    badnoisy_kurtosis.append(stats.kurtosis(bad_noisy))
    badnoisy_smooth.append(smooth_calc(bad_noisy))
    badnoisy_meancliprms.append(np.mean(bad_noisy)/calc_rms(bad_noisy[np.where(bad_noisy<rms_threshold*badnoisy_rms[i])]))
    badnoisy_peaktorms.append(max(bad_noisy)/calc_rms(bad_noisy[np.where(bad_noisy<max(bad_noisy))]))
    badnoisy_peaktomad.append(max(bad_noisy)/MAD(bad_noisy[np.where(bad_noisy<max(bad_noisy))]))
    
#plot results
plt.figure(9,(7.5,8.5))
plt.subplot(5,2,1)
plt.plot(badnoisy_rms,'.',markersize=2.0)
plt.title('Noisy Signal')
plt.ylabel('RMS')
plt.subplot(5,2,2)
plt.plot(badnoisy_rmsratio,'.',markersize=2.0)
plt.ylabel('RMS Ratio')
plt.subplot(5,2,3)
plt.plot(badnoisy_skew,'.',markersize=2.0)
plt.ylabel('Skewness')
plt.subplot(5,2,4)
plt.plot(badnoisy_kurtosis,'.',markersize=2.0)
plt.ylabel('Kurtosis')
plt.subplot(5,2,5)
plt.plot(badnoisy_smooth,'.',markersize=2.0)
plt.ylabel('Smoothess')
plt.subplot(5,2,6)
plt.plot(badnoisy_meancliprms,'.',markersize=2.0)
plt.ylabel('Mean/Clipped RMS')
plt.subplot(5,2,7)
plt.plot(badnoisy_peaktorms,'.',markersize=2.0)
plt.ylabel('Peak/RMS')
plt.subplot(5,2,8)
plt.plot(badnoisy_peaktomad,'.',markersize=2.0)
plt.ylabel('Peak/MAD')
plt.subplot(5,1,5)
plt.plot(bad_noisy)
plt.savefig('/home/jlinford/FRBs/FRB_MCLC/Bad_Noisy.png')
plt.savefig('/home/jlinford/FRBs/FRB_MCLC/Bad_Noisy.eps',format='eps',dpi=200)
plt.close()

badnoisy_avgs = get_avgs(badnoisy_rms,badnoisy_rmsratio,badnoisy_skew,badnoisy_kurtosis,badnoisy_smooth,badnoisy_meancliprms,badnoisy_peaktorms,badnoisy_peaktomad)

#bad data, 2 levels
bad2levels_rms = []
bad2levels_rmsratio = []
bad2levels_skew = []
bad2levels_kurtosis = []
bad2levels_smooth = []
bad2levels_meancliprms = []
bad2levels_peaktorms = []
bad2levels_peaktomad = []

for i in range(0,1000):
    bad_2levels = np.random.randn(31)/100.0
    bad_2levels[15::] = bad_2levels[15::] + np.abs(np.random.randn(1)/20.0)
    bad2levels_rms.append(calc_rms(bad_2levels))
    bad2levels_rmsratio.append(bad2levels_rms[i]/calc_rms(bad_2levels[np.where(bad_2levels<rms_threshold*bad2levels_rms[i])]))
    bad2levels_skew.append(stats.skew(bad_2levels))
    bad2levels_kurtosis.append(stats.kurtosis(bad_2levels))
    bad2levels_smooth.append(smooth_calc(bad_2levels))
    bad2levels_meancliprms.append(np.mean(bad_2levels)/calc_rms(bad_2levels[np.where(bad_2levels<rms_threshold*bad2levels_rms[i])]))
    bad2levels_peaktorms.append(max(bad_2levels)/calc_rms(bad_2levels[np.where(bad_2levels<max(bad_2levels))]))
    bad2levels_peaktomad.append(max(bad_2levels)/MAD(bad_2levels[np.where(bad_2levels<max(bad_2levels))]))

#plot results
plt.figure(10,(7.5,8.5))
plt.subplot(5,2,1)
plt.plot(bad2levels_rms,'.',markersize=2.0)
plt.title('2 Levels')
plt.ylabel('RMS')
plt.subplot(5,2,2)
plt.plot(bad2levels_rmsratio,'.',markersize=2.0)
plt.ylabel('RMS Ratio')
plt.subplot(5,2,3)
plt.plot(bad2levels_skew,'.',markersize=2.0)
plt.ylabel('Skewness')
plt.subplot(5,2,4)
plt.plot(bad2levels_kurtosis,'.',markersize=2.0)
plt.ylabel('Kurtosis')
plt.subplot(5,2,5)
plt.plot(bad2levels_smooth,'.',markersize=2.0)
plt.ylabel('Smoothness')
plt.subplot(5,2,6)
plt.plot(bad2levels_meancliprms,'.',markersize=2.0)
plt.ylabel('Mean/Clipped RMS')
plt.subplot(5,2,7)
plt.plot(bad2levels_peaktorms,'.',markersize=2.0)
plt.ylabel('Peak/RMS')
plt.subplot(5,2,8)
plt.plot(bad2levels_peaktomad,'.',markersize=2.0)
plt.ylabel('Peak/MAD')
plt.subplot(5,1,5)
plt.plot(bad_2levels)
plt.savefig('/home/jlinford/FRBs/FRB_MCLC/Bad_2Levels.png')
plt.savefig('/home/jlinford/FRBs/FRB_MCLC/Bad_2Levels.eps',format='eps',dpi=200)
plt.close()

bad2levels_avgs = get_avgs(bad2levels_rms,bad2levels_rmsratio,bad2levels_skew,bad2levels_kurtosis,bad2levels_smooth,bad2levels_meancliprms,bad2levels_peaktorms,bad2levels_peaktomad)

#noise change
badnoisechange_rms = []
badnoisechange_rmsratio = []
badnoisechange_skew = []
badnoisechange_kurtosis = []
badnoisechange_smooth = []
badnoisechange_meancliprms = []
badnoisechange_peaktorms = []
badnoisechange_peaktomad = []

for i in range(0,1000):
    bad_noisechange = np.random.randn(31)/100.0
    change_index = np.random.randint(2,15,size=1)
    bad_noisechange[int(change_index):int(change_index)+15] = np.random.randn(15)/10.0
    badnoisechange_rms.append(calc_rms(bad_noisechange))
    badnoisechange_rmsratio.append(badnoisechange_rms[i]/calc_rms(bad_noisechange[np.where(bad_noisechange<rms_threshold*badnoisechange_rms[i])]))
    badnoisechange_skew.append(stats.skew(bad_noisechange))
    badnoisechange_kurtosis.append(stats.kurtosis(bad_noisechange))
    badnoisechange_smooth.append(smooth_calc(bad_noisechange))
    badnoisechange_meancliprms.append(np.mean(bad_noisechange)/calc_rms(bad_noisechange[np.where(bad_noisechange<rms_threshold*badnoisechange_rms[i])]))
    badnoisechange_peaktorms.append(max(bad_noisechange)/calc_rms(bad_noisechange[np.where(bad_noisechange<max(bad_noisechange))]))
    badnoisechange_peaktomad.append(max(bad_noisechange)/MAD(bad_noisechange[np.where(bad_noisechange<max(bad_noisechange))]))
    
#plot results
plt.figure(11,(7.5,8.5))
plt.subplot(5,2,1)
plt.plot(badnoisechange_rms,'.',markersize=2.0)
plt.title('Noise Change')
plt.ylabel('RMS')
plt.subplot(5,2,2)
plt.plot(badnoisechange_rmsratio,'.',markersize=2.0)
plt.ylabel('RMS Ratio')
plt.subplot(5,2,3)
plt.plot(badnoisechange_skew,'.',markersize=2.0)
plt.ylabel('Skewness')
plt.subplot(5,2,4)
plt.plot(badnoisechange_kurtosis,'.',markersize=2.0)
plt.ylabel('Kurtosis')
plt.subplot(5,2,5)
plt.plot(badnoisechange_smooth,'.',markersize=2.0)
plt.ylabel('Smoothness')
plt.subplot(5,2,6)
plt.plot(badnoisechange_meancliprms,'.',markersize=2.0)
plt.ylabel('Mean/Clipped RMS')
plt.subplot(5,2,7)
plt.plot(badnoisechange_peaktorms,'.',markersize=2.0)
plt.ylabel('Peak/RMS')
plt.subplot(5,2,8)
plt.plot(badnoisechange_peaktomad,'.',markersize=2.0)
plt.ylabel('Peak/MAD')
plt.subplot(5,1,5)
plt.plot(bad_noisechange)
plt.savefig('/home/jlinford/FRBs/FRB_MCLC/Bad_NoiseChange.png')
plt.savefig('/home/jlinford/FRBs/FRB_MCLC/Bad_NoiseChange.eps',format='eps',dpi=200)
plt.close()

badnoisechange_avgs = get_avgs(badnoisechange_rms,badnoisechange_rmsratio,badnoisechange_skew,badnoisechange_kurtosis,badnoisechange_smooth,badnoisechange_meancliprms,badnoisechange_peaktorms,badnoisechange_peaktomad)

#sawtooth signal
badsawtooth_rms = []
badsawtooth_rmsratio = []
badsawtooth_skew = []
badsawtooth_kurtosis = []
badsawtooth_smooth= []
badsawtooth_meancliprms = []
badsawtooth_peaktorms = []
badsawtooth_peaktomad= []

for i in range(0,1000):
    bad_sawtooth = np.random.randn(31)/100.0
    saw_fall1 = 0.0-0.07/5.0*np.arange(0,6)
    saw_rise1 = saw_fall1[-1]+0.15/15.0*(np.arange(6,15)-5.0)
    saw_fall2 = saw_rise1[-1] - 0.055/5.0*(np.arange(15,26)-20)
    saw_rise2 = saw_fall2[-1]+0.05/5.0*(np.arange(26,31)-26)
    bad_sawtooth[0:6] = bad_sawtooth[0:6] + saw_fall1
    bad_sawtooth[6:15] = bad_sawtooth[6:15] + saw_rise1
    bad_sawtooth[15:26] = bad_sawtooth[15:26] + saw_fall2
    bad_sawtooth[26::] = bad_sawtooth[26::] + saw_rise2
    badsawtooth_rms.append(calc_rms(bad_sawtooth))
    badsawtooth_rmsratio.append(badsawtooth_rms[i]/calc_rms(bad_sawtooth[np.where(bad_sawtooth<rms_threshold*badsawtooth_rms[i])]))
    badsawtooth_skew.append(stats.skew(bad_sawtooth))
    badsawtooth_kurtosis.append(stats.kurtosis(bad_sawtooth))
    badsawtooth_smooth.append(smooth_calc(bad_sawtooth))
    badsawtooth_meancliprms.append(np.mean(bad_sawtooth)/calc_rms(bad_sawtooth[np.where(bad_sawtooth<rms_threshold*badsawtooth_rms[i])]))
    badsawtooth_peaktorms.append(max(bad_sawtooth)/calc_rms(bad_sawtooth[np.where(bad_sawtooth<max(bad_sawtooth))]))
    badsawtooth_peaktomad.append(max(bad_sawtooth)/MAD(bad_sawtooth[np.where(bad_sawtooth<max(bad_sawtooth))]))

#plot results
plt.figure(12,(7.5,8.5))
plt.subplot(5,2,1)
plt.plot(badsawtooth_rms,'.',markersize=2.0)
plt.title('Sawtooth Signal')
plt.ylabel('RMS')
plt.subplot(5,2,2)
plt.plot(badsawtooth_rmsratio,'.',markersize=2.0)
plt.ylabel('RMS Ratio')
plt.subplot(5,2,3)
plt.plot(badsawtooth_skew,'.',markersize=2.0)
plt.ylabel('Skewness')
plt.subplot(5,2,4)
plt.plot(badsawtooth_kurtosis,'.',markersize=2.0)
plt.ylabel('Kurtosis')
plt.subplot(5,2,5)
plt.plot(badsawtooth_smooth,'.',markersize=2.0)
plt.ylabel('Smoothness')
plt.subplot(5,2,6)
plt.plot(badsawtooth_meancliprms,'.',markersize=2.0)
plt.ylabel('Mean/Clipped RMS')
plt.subplot(5,2,7)
plt.plot(badsawtooth_peaktorms,'.',markersize=2.0)
plt.ylabel('Peak/RMS')
plt.subplot(5,2,8)
plt.plot(badsawtooth_peaktomad,'.',markersize=2.0)
plt.ylabel('Peak/MAD')
plt.subplot(5,1,5)
plt.plot(bad_sawtooth)
plt.savefig('/home/jlinford/FRBs/FRB_MCLC/Bad_Sawtooth.png')
plt.savefig('/home/jlinford/FRBs/FRB_MCLC/Bad_Sawtooth.eps',format='eps',dpi=200)
plt.close()

badsawtooth_avgs = get_avgs(badsawtooth_rms,badsawtooth_rmsratio,badsawtooth_skew,badsawtooth_kurtosis,badsawtooth_smooth,badsawtooth_meancliprms,badsawtooth_peaktorms,badsawtooth_peaktomad)

#==============================================================================
#make plots of the average values for each measuremnent in the simulated light curves
plt.figure(13,(8,7))
plt.subplot(4,2,1)
#RMS
plt.errorbar([1],vanilla_avgs[0,0],yerr=vanilla_avgs[0,2],color='b',marker='o')
plt.errorbar([2],early_frb_avgs[0,0],yerr=early_frb_avgs[0,2],color='b',marker='o')
plt.errorbar([3],late_frb_avgs[0,0],yerr=late_frb_avgs[0,2],color='b',marker='o')
plt.errorbar([4],lineardecay_avgs[0,0],yerr=lineardecay_avgs[0,2],color='b',marker='o')
plt.errorbar([5],expdecay_avgs[0,0],yerr=expdecay_avgs[0,2],color='b',marker='o')
plt.errorbar([6],narrowGauss_avgs[0,0],yerr=narrowGauss_avgs[0,2],color='b',marker='o')
plt.errorbar([7],broadGauss_avgs[0,0],yerr=broadGauss_avgs[0,2],color='b',marker='o')
plt.errorbar([8],badperiodic_avgs[0,0],yerr=badperiodic_avgs[0,2],color='r',marker='s')
plt.errorbar([9],badnoisy_avgs[0,0],yerr=badnoisy_avgs[0,2],color='r',marker='s')
plt.errorbar([10],bad2levels_avgs[0,0],yerr=bad2levels_avgs[0,2],color='r',marker='s')
plt.errorbar([11],badnoisechange_avgs[0,0],yerr=badnoisechange_avgs[0,2],color='r',marker='s')
plt.errorbar([12],badsawtooth_avgs[0,0],yerr=badsawtooth_avgs[0,2],color='r',marker='s')
plt.ylabel('RMS')
plt.subplot(4,2,2)
#RMS ratio
n=1
plt.errorbar([1],vanilla_avgs[n,0],yerr=vanilla_avgs[n,2],color='b',marker='o')
plt.errorbar([2],early_frb_avgs[n,0],yerr=early_frb_avgs[n,2],color='b',marker='o')
plt.errorbar([3],late_frb_avgs[n,0],yerr=late_frb_avgs[n,2],color='b',marker='o')
plt.errorbar([4],lineardecay_avgs[n,0],yerr=lineardecay_avgs[n,2],color='b',marker='o')
plt.errorbar([5],expdecay_avgs[n,0],yerr=expdecay_avgs[n,2],color='b',marker='o')
plt.errorbar([6],narrowGauss_avgs[n,0],yerr=narrowGauss_avgs[n,2],color='b',marker='o')
plt.errorbar([7],broadGauss_avgs[n,0],yerr=broadGauss_avgs[n,2],color='b',marker='o')
plt.errorbar([8],badperiodic_avgs[n,0],yerr=badperiodic_avgs[n,2],color='r',marker='s')
plt.errorbar([9],badnoisy_avgs[n,0],yerr=badnoisy_avgs[n,2],color='r',marker='s')
plt.errorbar([10],bad2levels_avgs[n,0],yerr=bad2levels_avgs[n,2],color='r',marker='s')
plt.errorbar([11],badnoisechange_avgs[n,0],yerr=badnoisechange_avgs[n,2],color='r',marker='s')
plt.errorbar([12],badsawtooth_avgs[n,0],yerr=badsawtooth_avgs[n,2],color='r',marker='s')
plt.ylabel('RMS Ratio')
plt.subplot(4,2,3)
#Skew
n=2
plt.errorbar([1],vanilla_avgs[n,0],yerr=vanilla_avgs[n,2],color='b',marker='o')
plt.errorbar([2],early_frb_avgs[n,0],yerr=early_frb_avgs[n,2],color='b',marker='o')
plt.errorbar([3],late_frb_avgs[n,0],yerr=late_frb_avgs[n,2],color='b',marker='o')
plt.errorbar([4],lineardecay_avgs[n,0],yerr=lineardecay_avgs[n,2],color='b',marker='o')
plt.errorbar([5],expdecay_avgs[n,0],yerr=expdecay_avgs[n,2],color='b',marker='o')
plt.errorbar([6],narrowGauss_avgs[n,0],yerr=narrowGauss_avgs[n,2],color='b',marker='o')
plt.errorbar([7],broadGauss_avgs[n,0],yerr=broadGauss_avgs[n,2],color='b',marker='o')
plt.errorbar([8],badperiodic_avgs[n,0],yerr=badperiodic_avgs[n,2],color='r',marker='s')
plt.errorbar([9],badnoisy_avgs[n,0],yerr=badnoisy_avgs[n,2],color='r',marker='s')
plt.errorbar([10],bad2levels_avgs[n,0],yerr=bad2levels_avgs[n,2],color='r',marker='s')
plt.errorbar([11],badnoisechange_avgs[n,0],yerr=badnoisechange_avgs[n,2],color='r',marker='s')
plt.errorbar([12],badsawtooth_avgs[n,0],yerr=badsawtooth_avgs[n,2],color='r',marker='s')
plt.ylabel('Skewness')
plt.subplot(4,2,4)
#Kurtosis
n=3
plt.errorbar([1],vanilla_avgs[n,0],yerr=vanilla_avgs[n,2],color='b',marker='o')
plt.errorbar([2],early_frb_avgs[n,0],yerr=early_frb_avgs[n,2],color='b',marker='o')
plt.errorbar([3],late_frb_avgs[n,0],yerr=late_frb_avgs[n,2],color='b',marker='o')
plt.errorbar([4],lineardecay_avgs[n,0],yerr=lineardecay_avgs[n,2],color='b',marker='o')
plt.errorbar([5],expdecay_avgs[n,0],yerr=expdecay_avgs[n,2],color='b',marker='o')
plt.errorbar([6],narrowGauss_avgs[n,0],yerr=narrowGauss_avgs[n,2],color='b',marker='o')
plt.errorbar([7],broadGauss_avgs[n,0],yerr=broadGauss_avgs[n,2],color='b',marker='o')
plt.errorbar([8],badperiodic_avgs[n,0],yerr=badperiodic_avgs[n,2],color='r',marker='s')
plt.errorbar([9],badnoisy_avgs[n,0],yerr=badnoisy_avgs[n,2],color='r',marker='s')
plt.errorbar([10],bad2levels_avgs[n,0],yerr=bad2levels_avgs[n,2],color='r',marker='s')
plt.errorbar([11],badnoisechange_avgs[n,0],yerr=badnoisechange_avgs[n,2],color='r',marker='s')
plt.errorbar([12],badsawtooth_avgs[n,0],yerr=badsawtooth_avgs[n,2],color='r',marker='s')
plt.ylabel('Kurtosis')
plt.subplot(4,2,5)
#Smoothness
n=4
plt.errorbar([1],vanilla_avgs[n,0],yerr=vanilla_avgs[n,2],color='b',marker='o')
plt.errorbar([2],early_frb_avgs[n,0],yerr=early_frb_avgs[n,2],color='b',marker='o')
plt.errorbar([3],late_frb_avgs[n,0],yerr=late_frb_avgs[n,2],color='b',marker='o')
plt.errorbar([4],lineardecay_avgs[n,0],yerr=lineardecay_avgs[n,2],color='b',marker='o')
plt.errorbar([5],expdecay_avgs[n,0],yerr=expdecay_avgs[n,2],color='b',marker='o')
plt.errorbar([6],narrowGauss_avgs[n,0],yerr=narrowGauss_avgs[n,2],color='b',marker='o')
plt.errorbar([7],broadGauss_avgs[n,0],yerr=broadGauss_avgs[n,2],color='b',marker='o')
plt.errorbar([8],badperiodic_avgs[n,0],yerr=badperiodic_avgs[n,2],color='r',marker='s')
plt.errorbar([9],badnoisy_avgs[n,0],yerr=badnoisy_avgs[n,2],color='r',marker='s')
plt.errorbar([10],bad2levels_avgs[n,0],yerr=bad2levels_avgs[n,2],color='r',marker='s')
plt.errorbar([11],badnoisechange_avgs[n,0],yerr=badnoisechange_avgs[n,2],color='r',marker='s')
plt.errorbar([12],badsawtooth_avgs[n,0],yerr=badsawtooth_avgs[n,2],color='r',marker='s')
plt.ylabel('Smoothness')
plt.subplot(4,2,6)
#mean to clipped ratio
n=5
plt.errorbar([1],vanilla_avgs[n,0],yerr=vanilla_avgs[n,2],color='b',marker='o')
plt.errorbar([2],early_frb_avgs[n,0],yerr=early_frb_avgs[n,2],color='b',marker='o')
plt.errorbar([3],late_frb_avgs[n,0],yerr=late_frb_avgs[n,2],color='b',marker='o')
plt.errorbar([4],lineardecay_avgs[n,0],yerr=lineardecay_avgs[n,2],color='b',marker='o')
plt.errorbar([5],expdecay_avgs[n,0],yerr=expdecay_avgs[n,2],color='b',marker='o')
plt.errorbar([6],narrowGauss_avgs[n,0],yerr=narrowGauss_avgs[n,2],color='b',marker='o')
plt.errorbar([7],broadGauss_avgs[n,0],yerr=broadGauss_avgs[n,2],color='b',marker='o')
plt.errorbar([8],badperiodic_avgs[n,0],yerr=badperiodic_avgs[n,2],color='r',marker='s')
plt.errorbar([9],badnoisy_avgs[n,0],yerr=badnoisy_avgs[n,2],color='r',marker='s')
plt.errorbar([10],bad2levels_avgs[n,0],yerr=bad2levels_avgs[n,2],color='r',marker='s')
plt.errorbar([11],badnoisechange_avgs[n,0],yerr=badnoisechange_avgs[n,2],color='r',marker='s')
plt.errorbar([12],badsawtooth_avgs[n,0],yerr=badsawtooth_avgs[n,2],color='r',marker='s')
plt.ylabel('Mean/ClipRMS')
plt.subplot(4,2,7)
#peak to rms
n=6
plt.errorbar([1],vanilla_avgs[n,0],yerr=vanilla_avgs[n,2],color='b',marker='o')
plt.errorbar([2],early_frb_avgs[n,0],yerr=early_frb_avgs[n,2],color='b',marker='o')
plt.errorbar([3],late_frb_avgs[n,0],yerr=late_frb_avgs[n,2],color='b',marker='o')
plt.errorbar([4],lineardecay_avgs[n,0],yerr=lineardecay_avgs[n,2],color='b',marker='o')
plt.errorbar([5],expdecay_avgs[n,0],yerr=expdecay_avgs[n,2],color='b',marker='o')
plt.errorbar([6],narrowGauss_avgs[n,0],yerr=narrowGauss_avgs[n,2],color='b',marker='o')
plt.errorbar([7],broadGauss_avgs[n,0],yerr=broadGauss_avgs[n,2],color='b',marker='o')
plt.errorbar([8],badperiodic_avgs[n,0],yerr=badperiodic_avgs[n,2],color='r',marker='s')
plt.errorbar([9],badnoisy_avgs[n,0],yerr=badnoisy_avgs[n,2],color='r',marker='s')
plt.errorbar([10],bad2levels_avgs[n,0],yerr=bad2levels_avgs[n,2],color='r',marker='s')
plt.errorbar([11],badnoisechange_avgs[n,0],yerr=badnoisechange_avgs[n,2],color='r',marker='s')
plt.errorbar([12],badsawtooth_avgs[n,0],yerr=badsawtooth_avgs[n,2],color='r',marker='s')
plt.ylabel('Peak/RMS')
plt.subplot(4,2,8)
#peak to MAD
n=7
plt.errorbar([1],vanilla_avgs[n,0],yerr=vanilla_avgs[n,2],color='b',marker='o')
plt.errorbar([2],early_frb_avgs[n,0],yerr=early_frb_avgs[n,2],color='b',marker='o')
plt.errorbar([3],late_frb_avgs[n,0],yerr=late_frb_avgs[n,2],color='b',marker='o')
plt.errorbar([4],lineardecay_avgs[n,0],yerr=lineardecay_avgs[n,2],color='b',marker='o')
plt.errorbar([5],expdecay_avgs[n,0],yerr=expdecay_avgs[n,2],color='b',marker='o')
plt.errorbar([6],narrowGauss_avgs[n,0],yerr=narrowGauss_avgs[n,2],color='b',marker='o')
plt.errorbar([7],broadGauss_avgs[n,0],yerr=broadGauss_avgs[n,2],color='b',marker='o')
plt.errorbar([8],badperiodic_avgs[n,0],yerr=badperiodic_avgs[n,2],color='r',marker='s')
plt.errorbar([9],badnoisy_avgs[n,0],yerr=badnoisy_avgs[n,2],color='r',marker='s')
plt.errorbar([10],bad2levels_avgs[n,0],yerr=bad2levels_avgs[n,2],color='r',marker='s')
plt.errorbar([11],badnoisechange_avgs[n,0],yerr=badnoisechange_avgs[n,2],color='r',marker='s')
plt.errorbar([12],badsawtooth_avgs[n,0],yerr=badsawtooth_avgs[n,2],color='r',marker='s')
plt.ylabel('Peak/MAD')
plt.savefig('/home/jlinford/FRBs/FRB_MCLC/FRB_LC_Stat_means.png')
plt.savefig('/home/jlinford/FRBs/FRB_MCLC/FRB_LC_Stat_means.eps',format='eps',dpi=200)

#try plotting skew vs rms ratio for all simulating light curves
ms=0.5
plt.figure(14)
#'good' frbs
plt.plot(vanilla_skew,vanilla_rmsratio,'b.',markersize=ms)
plt.plot(early_skew,early_rmsratio,'b.',markersize=ms)
plt.plot(late_skew,late_rmsratio,'b.',markersize=ms)
plt.plot(lineardecay_skew,lineardecay_rmsratio,'b.',markersize=ms)
plt.plot(expdecay_skew,expdecay_rmsratio,'b.',markersize=ms)
plt.plot(narrowGauss_skew,narrowGauss_rmsratio,'b.',markersize=ms)
plt.plot(broadGauss_skew,broadGauss_rmsratio,'b.',markersize=ms)
#'bad' light curves
plt.plot(badperiodic_skew,badperiodic_rmsratio,'r.',markersize=ms)
plt.plot(badnoisy_skew,badnoisy_rmsratio,'r.',markersize=ms)
plt.plot(bad2levels_skew,bad2levels_rmsratio,'r.',markersize=ms)
plt.plot(badnoisechange_skew,badnoisechange_rmsratio,'r.',markersize=ms)
plt.plot(badsawtooth_skew,badsawtooth_rmsratio,'r.',markersize=ms)
plt.xlabel('Skewness')
plt.ylabel('RMS Ratio')
plt.savefig('/home/jlinford/FRBs/FRB_MCLC/FRB_SkewVsRMSratio.png')
plt.savefig('/home/jlinford/FRBs/FRB_MCLC/FRB_SkewVsRMSratio.eps',format='eps',dpi=200)

#try plotting kurtosis vs rms ratio
plt.figure(15)
#'good' frbs
plt.plot(vanilla_kurtosis,vanilla_rmsratio,'b.',markersize=ms)
plt.plot(early_kurtosis,early_rmsratio,'b.',markersize=ms)
plt.plot(late_kurtosis,late_rmsratio,'b.',markersize=ms)
plt.plot(lineardecay_kurtosis,lineardecay_rmsratio,'b.',markersize=ms)
plt.plot(expdecay_kurtosis,expdecay_rmsratio,'b.',markersize=ms)
plt.plot(narrowGauss_kurtosis,narrowGauss_rmsratio,'b.',markersize=ms)
plt.plot(broadGauss_kurtosis,broadGauss_rmsratio,'b.',markersize=ms)
#'bad' light curves
plt.plot(badperiodic_kurtosis,badperiodic_rmsratio,'r.',markersize=ms)
plt.plot(badnoisy_kurtosis,badnoisy_rmsratio,'r.',markersize=ms)
plt.plot(bad2levels_kurtosis,bad2levels_rmsratio,'r.',markersize=ms)
plt.plot(badnoisechange_kurtosis,badnoisechange_rmsratio,'r.',markersize=ms)
plt.plot(badsawtooth_kurtosis,badsawtooth_rmsratio,'r.',markersize=ms)
plt.xlabel('Kurtosis')
plt.ylabel('RMS Ratio')
plt.savefig('/home/jlinford/FRBs/FRB_MCLC/FRB_KurtosisVsRMSratio.png')
plt.savefig('/home/jlinford/FRBs/FRB_MCLC/FRB_KurtosisVsRMSratio.eps',format='eps',dpi=200)

#try plotting skew vs kurtosis
plt.figure(16)
#'good' frbs
plt.plot(vanilla_kurtosis,vanilla_skew,'b.',markersize=ms)
plt.plot(early_kurtosis,early_skew,'b.',markersize=ms)
plt.plot(late_kurtosis,late_skew,'b.',markersize=ms)
plt.plot(lineardecay_kurtosis,lineardecay_skew,'b.',markersize=ms)
plt.plot(expdecay_kurtosis,expdecay_skew,'b.',markersize=ms)
plt.plot(narrowGauss_kurtosis,narrowGauss_skew,'b.',markersize=ms)
plt.plot(broadGauss_kurtosis,broadGauss_skew,'b.',markersize=ms)
#'bad' light curves
plt.plot(badperiodic_kurtosis,badperiodic_skew,'r.',markersize=ms)
plt.plot(badnoisy_kurtosis,badnoisy_skew,'r.',markersize=ms)
plt.plot(bad2levels_kurtosis,bad2levels_skew,'r.',markersize=ms)
plt.plot(badnoisechange_kurtosis,badnoisechange_skew,'r.',markersize=ms)
plt.plot(badsawtooth_kurtosis,badsawtooth_skew,'r.',markersize=ms)
plt.xlabel('Kurtosis')
plt.ylabel('Skewness')
plt.savefig('/home/jlinford/FRBs/FRB_MCLC/FRB_KurtosisVsSkew.png')
plt.savefig('/home/jlinford/FRBs/FRB_MCLC/FRB_KurtosisVsSkew.eps',format='eps',dpi=200)

#rms ratio vs peak/MAD
plt.figure(17)
#'good' frbs
plt.plot(vanilla_peaktomad,vanilla_rmsratio,'b.',markersize=ms)
plt.plot(early_peaktomad,early_rmsratio,'b.',markersize=ms)
plt.plot(late_peaktomad,late_rmsratio,'b.',markersize=ms)
plt.plot(lineardecay_peaktomad,lineardecay_rmsratio,'b.',markersize=ms)
plt.plot(expdecay_peaktomad,expdecay_rmsratio,'b.',markersize=ms)
plt.plot(narrowGauss_peaktomad,narrowGauss_rmsratio,'b.',markersize=ms)
plt.plot(broadGauss_peaktomad,broadGauss_rmsratio,'b.',markersize=ms)
#'bad' light curves
plt.plot(badperiodic_peaktomad,badperiodic_rmsratio,'r.',markersize=ms)
plt.plot(badnoisy_peaktomad,badnoisy_rmsratio,'r.',markersize=ms)
plt.plot(bad2levels_peaktomad,bad2levels_rmsratio,'r.',markersize=ms)
plt.plot(badnoisechange_peaktomad,badnoisechange_rmsratio,'r.',markersize=ms)
plt.plot(badsawtooth_peaktomad,badsawtooth_rmsratio,'r.',markersize=ms)
plt.xlabel('Peak/MAD')
plt.ylabel('RMS Ratio')
plt.savefig('/home/jlinford/FRBs/FRB_MCLC/FRB_PeakToMADVsRMSratio.png')
plt.savefig('/home/jlinford/FRBs/FRB_MCLC/FRB_PeakToMADVsRMSratio.eps',format='eps',dpi=200)

#skewness vs peak/MAD
plt.figure(18)
#'good' frbs
plt.plot(vanilla_peaktomad,vanilla_skew,'b.',markersize=ms)
plt.plot(early_peaktomad,early_skew,'b.',markersize=ms)
plt.plot(late_peaktomad,late_skew,'b.',markersize=ms)
plt.plot(lineardecay_peaktomad,lineardecay_skew,'b.',markersize=ms)
plt.plot(expdecay_peaktomad,expdecay_skew,'b.',markersize=ms)
plt.plot(narrowGauss_peaktomad,narrowGauss_skew,'b.',markersize=ms)
plt.plot(broadGauss_peaktomad,broadGauss_skew,'b.',markersize=ms)
#'bad' light curves
plt.plot(badperiodic_peaktomad,badperiodic_skew,'r.',markersize=ms)
plt.plot(badnoisy_peaktomad,badnoisy_skew,'r.',markersize=ms)
plt.plot(bad2levels_peaktomad,bad2levels_skew,'r.',markersize=ms)
plt.plot(badnoisechange_peaktomad,badnoisechange_skew,'r.',markersize=ms)
plt.plot(badsawtooth_peaktomad,badsawtooth_skew,'r.',markersize=ms)
plt.xlabel('Peak/MAD')
plt.ylabel('Skewness')
plt.savefig('/home/jlinford/FRBs/FRB_MCLC/FRB_PeakToMADVsSkew.png')
plt.savefig('/home/jlinford/FRBs/FRB_MCLC/FRB_PeakToMADVsSkew.eps',format='eps',dpi=200)

#kurtosis vs peak/MAD
plt.figure(19)
#'good' frbs
plt.plot(vanilla_peaktomad,vanilla_kurtosis,'b.',markersize=ms)
plt.plot(early_peaktomad,early_kurtosis,'b.',markersize=ms)
plt.plot(late_peaktomad,late_kurtosis,'b.',markersize=ms)
plt.plot(lineardecay_peaktomad,lineardecay_kurtosis,'b.',markersize=ms)
plt.plot(expdecay_peaktomad,expdecay_kurtosis,'b.',markersize=ms)
plt.plot(narrowGauss_peaktomad,narrowGauss_kurtosis,'b.',markersize=ms)
plt.plot(broadGauss_peaktomad,broadGauss_kurtosis,'b.',markersize=ms)
#'bad' light curves
plt.plot(badperiodic_peaktomad,badperiodic_kurtosis,'r.',markersize=ms)
plt.plot(badnoisy_peaktomad,badnoisy_kurtosis,'r.',markersize=ms)
plt.plot(bad2levels_peaktomad,bad2levels_kurtosis,'r.',markersize=ms)
plt.plot(badnoisechange_peaktomad,badnoisechange_kurtosis,'r.',markersize=ms)
plt.plot(badsawtooth_peaktomad,badsawtooth_kurtosis,'r.',markersize=ms)
plt.xlabel('Peak/MAD')
plt.ylabel('Kustosis')
plt.savefig('/home/jlinford/FRBs/FRB_MCLC/FRB_PeakToMADVsKurtosis.png')
plt.savefig('/home/jlinford/FRBs/FRB_MCLC/FRB_PeakToMADVsKurtosis.eps',format='eps',dpi=200)


#try plotting histograms to better show overlap
good_rmsratio = vanilla_rmsratio + early_rmsratio + late_rmsratio + lineardecay_rmsratio + expdecay_rmsratio + narrowGauss_rmsratio + broadGauss_rmsratio
good_skew = vanilla_skew + early_skew + late_skew + lineardecay_skew + expdecay_skew + narrowGauss_skew + broadGauss_skew
good_kurtosis = vanilla_kurtosis + early_kurtosis + late_kurtosis + lineardecay_kurtosis + expdecay_kurtosis + narrowGauss_kurtosis + broadGauss_kurtosis
good_peaktomad = vanilla_peaktomad + early_peaktomad + late_peaktomad + lineardecay_peaktomad + expdecay_peaktomad + narrowGauss_peaktomad + broadGauss_peaktomad
bad_rmsratio = badperiodic_rmsratio + badnoisy_rmsratio + bad2levels_rmsratio + badnoisechange_rmsratio + badsawtooth_rmsratio
bad_skew = badperiodic_skew + badnoisy_skew + bad2levels_skew + badnoisechange_skew + badsawtooth_skew
bad_kurtosis = badperiodic_kurtosis + badnoisy_kurtosis + bad2levels_kurtosis + badnoisechange_kurtosis + badsawtooth_kurtosis
bad_peaktomad = badperiodic_peaktomad + badnoisy_peaktomad + bad2levels_peaktomad + badnoisechange_peaktomad + badsawtooth_peaktomad

plt.figure(20,(7,6))
plt.subplot(2,2,1)
plt.hist([good_rmsratio,bad_rmsratio],bins=np.arange(1.0,5.5,0.1),stacked=True,color=["blue","red"])
plt.xlabel('RMS Ratio')
plt.ylim([0,1000])
plt.subplot(2,2,2)
plt.hist([good_skew,bad_skew],bins=np.arange(-3.0,5.2,0.1),stacked=True,color=["blue","red"])
plt.xlabel('Skewness')
plt.subplot(2,2,3)
plt.hist([good_kurtosis,bad_kurtosis],bins=np.arange(-3.0,25.0,0.5),stacked=True,color=["blue","red"])
plt.xlabel('Kurtosis')
plt.subplot(2,2,4)
plt.hist([good_peaktomad,bad_peaktomad],bins=np.arange(1.0,60.0,1.0),stacked=True,color=["blue","red"])
plt.xlabel('Peak/MAD')
plt.savefig('/home/jlinford/FRBs/FRB_MCLC/FRB_stat_histograms.eps',format='eps',dpi=200)


#make a single figure will all of the statistical relationships in subplots instead of one plot for each
plt.figure(21,(8,10))
plt.subplot(3,2,1)
#rms ratio vs skewness
#'good' frbs
plt.plot(vanilla_skew,vanilla_rmsratio,'b.',markersize=ms)
plt.plot(early_skew,early_rmsratio,'b.',markersize=ms)
plt.plot(late_skew,late_rmsratio,'b.',markersize=ms)
plt.plot(lineardecay_skew,lineardecay_rmsratio,'b.',markersize=ms)
plt.plot(expdecay_skew,expdecay_rmsratio,'b.',markersize=ms)
plt.plot(narrowGauss_skew,narrowGauss_rmsratio,'b.',markersize=ms)
plt.plot(broadGauss_skew,broadGauss_rmsratio,'b.',markersize=ms)
#'bad' light curves
plt.plot(badperiodic_skew,badperiodic_rmsratio,'r.',markersize=ms)
plt.plot(badnoisy_skew,badnoisy_rmsratio,'r.',markersize=ms)
plt.plot(bad2levels_skew,bad2levels_rmsratio,'r.',markersize=ms)
plt.plot(badnoisechange_skew,badnoisechange_rmsratio,'r.',markersize=ms)
plt.plot(badsawtooth_skew,badsawtooth_rmsratio,'r.',markersize=ms)
plt.xlabel('Skewness')
plt.ylabel('RMS Ratio')
plt.subplot(3,2,2)
#rms ratio vs kurtosis
#'good' frbs
plt.plot(vanilla_kurtosis,vanilla_rmsratio,'b.',markersize=ms)
plt.plot(early_kurtosis,early_rmsratio,'b.',markersize=ms)
plt.plot(late_kurtosis,late_rmsratio,'b.',markersize=ms)
plt.plot(lineardecay_kurtosis,lineardecay_rmsratio,'b.',markersize=ms)
plt.plot(expdecay_kurtosis,expdecay_rmsratio,'b.',markersize=ms)
plt.plot(narrowGauss_kurtosis,narrowGauss_rmsratio,'b.',markersize=ms)
plt.plot(broadGauss_kurtosis,broadGauss_rmsratio,'b.',markersize=ms)
#'bad' light curves
plt.plot(badperiodic_kurtosis,badperiodic_rmsratio,'r.',markersize=ms)
plt.plot(badnoisy_kurtosis,badnoisy_rmsratio,'r.',markersize=ms)
plt.plot(bad2levels_kurtosis,bad2levels_rmsratio,'r.',markersize=ms)
plt.plot(badnoisechange_kurtosis,badnoisechange_rmsratio,'r.',markersize=ms)
plt.plot(badsawtooth_kurtosis,badsawtooth_rmsratio,'r.',markersize=ms)
plt.xlabel('Kurtosis')
plt.ylabel('RMS Ratio')
plt.subplot(3,2,3)
#rms ratio vs peak/mad
#'good' frbs
plt.plot(vanilla_peaktomad,vanilla_rmsratio,'b.',markersize=ms)
plt.plot(early_peaktomad,early_rmsratio,'b.',markersize=ms)
plt.plot(late_peaktomad,late_rmsratio,'b.',markersize=ms)
plt.plot(lineardecay_peaktomad,lineardecay_rmsratio,'b.',markersize=ms)
plt.plot(expdecay_peaktomad,expdecay_rmsratio,'b.',markersize=ms)
plt.plot(narrowGauss_peaktomad,narrowGauss_rmsratio,'b.',markersize=ms)
plt.plot(broadGauss_peaktomad,broadGauss_rmsratio,'b.',markersize=ms)
#'bad' light curves
plt.plot(badperiodic_peaktomad,badperiodic_rmsratio,'r.',markersize=ms)
plt.plot(badnoisy_peaktomad,badnoisy_rmsratio,'r.',markersize=ms)
plt.plot(bad2levels_peaktomad,bad2levels_rmsratio,'r.',markersize=ms)
plt.plot(badnoisechange_peaktomad,badnoisechange_rmsratio,'r.',markersize=ms)
plt.plot(badsawtooth_peaktomad,badsawtooth_rmsratio,'r.',markersize=ms)
plt.xlabel('Peak/MAD')
plt.ylabel('RMS Ratio')
plt.subplot(3,2,4)
#skewness vs kurtosis
#'good' frbs
plt.plot(vanilla_kurtosis,vanilla_skew,'b.',markersize=ms)
plt.plot(early_kurtosis,early_skew,'b.',markersize=ms)
plt.plot(late_kurtosis,late_skew,'b.',markersize=ms)
plt.plot(lineardecay_kurtosis,lineardecay_skew,'b.',markersize=ms)
plt.plot(expdecay_kurtosis,expdecay_skew,'b.',markersize=ms)
plt.plot(narrowGauss_kurtosis,narrowGauss_skew,'b.',markersize=ms)
plt.plot(broadGauss_kurtosis,broadGauss_skew,'b.',markersize=ms)
#'bad' light curves
plt.plot(badperiodic_kurtosis,badperiodic_skew,'r.',markersize=ms)
plt.plot(badnoisy_kurtosis,badnoisy_skew,'r.',markersize=ms)
plt.plot(bad2levels_kurtosis,bad2levels_skew,'r.',markersize=ms)
plt.plot(badnoisechange_kurtosis,badnoisechange_skew,'r.',markersize=ms)
plt.plot(badsawtooth_kurtosis,badsawtooth_skew,'r.',markersize=ms)
plt.xlabel('Kurtosis')
plt.ylabel('Skewness')
plt.subplot(3,2,5)
#skewness vs peak/mad
#'good' frbs
plt.plot(vanilla_peaktomad,vanilla_skew,'b.',markersize=ms)
plt.plot(early_peaktomad,early_skew,'b.',markersize=ms)
plt.plot(late_peaktomad,late_skew,'b.',markersize=ms)
plt.plot(lineardecay_peaktomad,lineardecay_skew,'b.',markersize=ms)
plt.plot(expdecay_peaktomad,expdecay_skew,'b.',markersize=ms)
plt.plot(narrowGauss_peaktomad,narrowGauss_skew,'b.',markersize=ms)
plt.plot(broadGauss_peaktomad,broadGauss_skew,'b.',markersize=ms)
#'bad' light curves
plt.plot(badperiodic_peaktomad,badperiodic_skew,'r.',markersize=ms)
plt.plot(badnoisy_peaktomad,badnoisy_skew,'r.',markersize=ms)
plt.plot(bad2levels_peaktomad,bad2levels_skew,'r.',markersize=ms)
plt.plot(badnoisechange_peaktomad,badnoisechange_skew,'r.',markersize=ms)
plt.plot(badsawtooth_peaktomad,badsawtooth_skew,'r.',markersize=ms)
plt.xlabel('Peak/MAD')
plt.ylabel('Skewness')
plt.subplot(3,2,6)
#kurtosis vs peak/MAD
#'good' frbs
plt.plot(vanilla_peaktomad,vanilla_kurtosis,'b.',markersize=ms)
plt.plot(early_peaktomad,early_kurtosis,'b.',markersize=ms)
plt.plot(late_peaktomad,late_kurtosis,'b.',markersize=ms)
plt.plot(lineardecay_peaktomad,lineardecay_kurtosis,'b.',markersize=ms)
plt.plot(expdecay_peaktomad,expdecay_kurtosis,'b.',markersize=ms)
plt.plot(narrowGauss_peaktomad,narrowGauss_kurtosis,'b.',markersize=ms)
plt.plot(broadGauss_peaktomad,broadGauss_kurtosis,'b.',markersize=ms)
#'bad' light curves
plt.plot(badperiodic_peaktomad,badperiodic_kurtosis,'r.',markersize=ms)
plt.plot(badnoisy_peaktomad,badnoisy_kurtosis,'r.',markersize=ms)
plt.plot(bad2levels_peaktomad,bad2levels_kurtosis,'r.',markersize=ms)
plt.plot(badnoisechange_peaktomad,badnoisechange_kurtosis,'r.',markersize=ms)
plt.plot(badsawtooth_peaktomad,badsawtooth_kurtosis,'r.',markersize=ms)
plt.xlabel('Peak/MAD')
plt.ylabel('Kustosis')
plt.savefig('/home/jlinford/FRBs/FRB_MCLC/FRB_StatRelations.png')
plt.savefig('/home/jlinford/FRBs/FRB_MCLC/FRB_StatRelations.eps',format='eps',dpi=200)


#==============================================================================

#want to write out a file with all of the relavent statistics in it
#create a csv file to write data to
mclc_of = open('/home/jlinford/FRBs/FRB_MCLC/MCLC_Statistics.csv','w')
#header
mclc_of.write('Name , Measurement, Mean, Median, STD, Max, Min'+'\n')
#Vanilla FRB
mclc_of.write('Vanilla FRB' + ',' + 'RMS' + ',' + str(vanilla_avgs[0,0]) + ',' + str(vanilla_avgs[0,1]) + ',' + str(vanilla_avgs[0,2]) + ',' + str(vanilla_avgs[0,3]) + ',' + str(vanilla_avgs[0,4]) +'\n' )
mclc_of.write('' + ',' + 'RMS Ratio' + ',' + str(vanilla_avgs[1,0]) + ',' + str(vanilla_avgs[1,1]) + ',' + str(vanilla_avgs[1,2]) + ',' + str(vanilla_avgs[1,3]) + ',' + str(vanilla_avgs[1,4]) +'\n' )
mclc_of.write('' + ',' + 'Skewness' + ',' + str(vanilla_avgs[2,0]) + ',' + str(vanilla_avgs[2,1]) + ',' + str(vanilla_avgs[2,2]) + ',' + str(vanilla_avgs[2,3]) + ',' + str(vanilla_avgs[2,4]) +'\n' )
mclc_of.write('' + ',' + 'Kurtosis' + ',' + str(vanilla_avgs[3,0]) + ',' + str(vanilla_avgs[3,1]) + ',' + str(vanilla_avgs[3,2]) + ',' + str(vanilla_avgs[3,3]) + ',' + str(vanilla_avgs[3,4]) +'\n' )
mclc_of.write('' + ',' + 'Smoothness' + ',' + str(vanilla_avgs[4,0]) + ',' + str(vanilla_avgs[4,1]) + ',' + str(vanilla_avgs[4,2]) + ',' + str(vanilla_avgs[4,3]) + ',' + str(vanilla_avgs[4,4]) +'\n' )
mclc_of.write('' + ',' + 'Mean to Clipped RMS Ratio' + ',' + str(vanilla_avgs[5,0]) + ',' + str(vanilla_avgs[5,1]) + ',' + str(vanilla_avgs[5,2]) + ',' + str(vanilla_avgs[5,3]) + ',' + str(vanilla_avgs[5,4]) +'\n' )
mclc_of.write('' + ',' + 'Peak to RMS Ratio' + ',' + str(vanilla_avgs[6,0]) + ',' + str(vanilla_avgs[6,1]) + ',' + str(vanilla_avgs[6,2]) + ',' + str(vanilla_avgs[6,3]) + ',' + str(vanilla_avgs[6,4]) +'\n' )
mclc_of.write('' + ',' + 'Peak to Median Absolute Deviation' + ',' + str(vanilla_avgs[7,0]) + ',' + str(vanilla_avgs[7,1]) + ',' + str(vanilla_avgs[7,2]) + ',' + str(vanilla_avgs[7,3]) + ',' + str(vanilla_avgs[7,4]) +'\n' )
#early FRB
mclc_of.write('Early FRB' + ',' + 'RMS' + ',' + str(early_frb_avgs[0,0]) + ',' + str(early_frb_avgs[0,1]) + ',' + str(early_frb_avgs[0,2]) + ',' + str(early_frb_avgs[0,3]) + ',' + str(early_frb_avgs[0,4]) +'\n' )
mclc_of.write('' + ',' + 'RMS Ratio' + ',' + str(early_frb_avgs[1,0]) + ',' + str(early_frb_avgs[1,1]) + ',' + str(early_frb_avgs[1,2]) + ',' + str(early_frb_avgs[1,3]) + ',' + str(early_frb_avgs[1,4]) +'\n' )
mclc_of.write('' + ',' + 'Skewness' + ',' + str(early_frb_avgs[2,0]) + ',' + str(early_frb_avgs[2,1]) + ',' + str(early_frb_avgs[2,2]) + ',' + str(early_frb_avgs[2,3]) + ',' + str(early_frb_avgs[2,4]) +'\n' )
mclc_of.write('' + ',' + 'Kurtosis' + ',' + str(early_frb_avgs[3,0]) + ',' + str(early_frb_avgs[3,1]) + ',' + str(early_frb_avgs[3,2]) + ',' + str(early_frb_avgs[3,3]) + ',' + str(early_frb_avgs[3,4]) +'\n' )
mclc_of.write('' + ',' + 'Smoothness' + ',' + str(early_frb_avgs[4,0]) + ',' + str(early_frb_avgs[4,1]) + ',' + str(early_frb_avgs[4,2]) + ',' + str(early_frb_avgs[4,3]) + ',' + str(early_frb_avgs[4,4]) +'\n' )
mclc_of.write('' + ',' + 'Mean to Clipped RMS Ratio' + ',' + str(early_frb_avgs[5,0]) + ',' + str(early_frb_avgs[5,1]) + ',' + str(early_frb_avgs[5,2]) + ',' + str(early_frb_avgs[5,3]) + ',' + str(early_frb_avgs[5,4]) +'\n' )
mclc_of.write('' + ',' + 'Peak to RMS Ratio' + ',' + str(early_frb_avgs[6,0]) + ',' + str(early_frb_avgs[6,1]) + ',' + str(early_frb_avgs[6,2]) + ',' + str(early_frb_avgs[6,3]) + ',' + str(early_frb_avgs[6,4]) +'\n' )
mclc_of.write('' + ',' + 'Peak to Median Absolute Deviation' + ',' + str(early_frb_avgs[7,0]) + ',' + str(early_frb_avgs[7,1]) + ',' + str(early_frb_avgs[7,2]) + ',' + str(early_frb_avgs[7,3]) + ',' + str(early_frb_avgs[7,4]) +'\n' )
#late FRB
mclc_of.write('Late FRB' + ',' + 'RMS' + ',' + str(late_frb_avgs[0,0]) + ',' + str(late_frb_avgs[0,1]) + ',' + str(late_frb_avgs[0,2]) + ',' + str(late_frb_avgs[0,3]) + ',' + str(late_frb_avgs[0,4]) +'\n' )
mclc_of.write('' + ',' + 'RMS Ratio' + ',' + str(late_frb_avgs[1,0]) + ',' + str(late_frb_avgs[1,1]) + ',' + str(late_frb_avgs[1,2]) + ',' + str(late_frb_avgs[1,3]) + ',' + str(late_frb_avgs[1,4]) +'\n' )
mclc_of.write('' + ',' + 'Skewness' + ',' + str(late_frb_avgs[2,0]) + ',' + str(late_frb_avgs[2,1]) + ',' + str(late_frb_avgs[2,2]) + ',' + str(late_frb_avgs[2,3]) + ',' + str(late_frb_avgs[2,4]) +'\n' )
mclc_of.write('' + ',' + 'Kurtosis' + ',' + str(late_frb_avgs[3,0]) + ',' + str(late_frb_avgs[3,1]) + ',' + str(late_frb_avgs[3,2]) + ',' + str(late_frb_avgs[3,3]) + ',' + str(late_frb_avgs[3,4]) +'\n' )
mclc_of.write('' + ',' + 'Smoothness' + ',' + str(late_frb_avgs[4,0]) + ',' + str(late_frb_avgs[4,1]) + ',' + str(late_frb_avgs[4,2]) + ',' + str(late_frb_avgs[4,3]) + ',' + str(late_frb_avgs[4,4]) +'\n' )
mclc_of.write('' + ',' + 'Mean to Clipped RMS Ratio' + ',' + str(late_frb_avgs[5,0]) + ',' + str(late_frb_avgs[5,1]) + ',' + str(late_frb_avgs[5,2]) + ',' + str(late_frb_avgs[5,3]) + ',' + str(late_frb_avgs[5,4]) +'\n' )
mclc_of.write('' + ',' + 'Peak to RMS Ratio' + ',' + str(late_frb_avgs[6,0]) + ',' + str(late_frb_avgs[6,1]) + ',' + str(late_frb_avgs[6,2]) + ',' + str(late_frb_avgs[6,3]) + ',' + str(late_frb_avgs[6,4]) +'\n' )
mclc_of.write('' + ',' + 'Peak to Median Absolute Deviation' + ',' + str(late_frb_avgs[7,0]) + ',' + str(late_frb_avgs[7,1]) + ',' + str(late_frb_avgs[7,2]) + ',' + str(late_frb_avgs[7,3]) + ',' + str(late_frb_avgs[7,4]) +'\n' )
#linear decay FRB
mclc_of.write('Linear Decay FRB' + ',' + 'RMS' + ',' + str(lineardecay_avgs[0,0]) + ',' + str(lineardecay_avgs[0,1]) + ',' + str(lineardecay_avgs[0,2]) + ',' + str(lineardecay_avgs[0,3]) + ',' + str(lineardecay_avgs[0,4]) +'\n' )
mclc_of.write('' + ',' + 'RMS Ratio' + ',' + str(lineardecay_avgs[1,0]) + ',' + str(lineardecay_avgs[1,1]) + ',' + str(lineardecay_avgs[1,2]) + ',' + str(lineardecay_avgs[1,3]) + ',' + str(lineardecay_avgs[1,4]) +'\n' )
mclc_of.write('' + ',' + 'Skew' + ',' + str(lineardecay_avgs[2,0]) + ',' + str(lineardecay_avgs[2,1]) + ',' + str(lineardecay_avgs[2,2]) + ',' + str(lineardecay_avgs[2,3]) + ',' + str(lineardecay_avgs[2,4]) +'\n' )
mclc_of.write('' + ',' + 'Kurtosis' + ',' + str(lineardecay_avgs[3,0]) + ',' + str(lineardecay_avgs[3,1]) + ',' + str(lineardecay_avgs[3,2]) + ',' + str(lineardecay_avgs[3,3]) + ',' + str(lineardecay_avgs[3,4]) +'\n' )
mclc_of.write('' + ',' + 'Smoothness' + ',' + str(lineardecay_avgs[4,0]) + ',' + str(lineardecay_avgs[4,1]) + ',' + str(lineardecay_avgs[4,2]) + ',' + str(lineardecay_avgs[4,3]) + ',' + str(lineardecay_avgs[4,4]) +'\n' )
mclc_of.write('' + ',' + 'Mean to Clipped RMS Ratio' + ',' + str(lineardecay_avgs[5,0]) + ',' + str(lineardecay_avgs[5,1]) + ',' + str(lineardecay_avgs[5,2]) + ',' + str(lineardecay_avgs[5,3]) + ',' + str(lineardecay_avgs[5,4]) +'\n' )
mclc_of.write('' + ',' + 'Peak to RMS Ratio' + ',' + str(lineardecay_avgs[6,0]) + ',' + str(lineardecay_avgs[6,1]) + ',' + str(lineardecay_avgs[6,2]) + ',' + str(lineardecay_avgs[6,3]) + ',' + str(lineardecay_avgs[6,4]) +'\n' )
mclc_of.write('' + ',' + 'Peak to Median Absolute Deviation' + ',' + str(lineardecay_avgs[7,0]) + ',' + str(lineardecay_avgs[7,1]) + ',' + str(lineardecay_avgs[7,2]) + ',' + str(lineardecay_avgs[7,3]) + ',' + str(lineardecay_avgs[7,4]) +'\n' )
#exponential decay
mclc_of.write('Exponential Decay FRB' + ',' + 'RMS' + ',' + str(expdecay_avgs[0,0]) + ',' + str(expdecay_avgs[0,1]) + ',' + str(expdecay_avgs[0,2]) + ',' + str(expdecay_avgs[0,3]) + ',' + str(expdecay_avgs[0,4]) +'\n' )
mclc_of.write('' + ',' + 'RMS Ratio' + ',' + str(expdecay_avgs[1,0]) + ',' + str(expdecay_avgs[1,1]) + ',' + str(expdecay_avgs[1,2]) + ',' + str(expdecay_avgs[1,3]) + ',' + str(expdecay_avgs[1,4]) +'\n' )
mclc_of.write('' + ',' + 'Skewness' + ',' + str(expdecay_avgs[2,0]) + ',' + str(expdecay_avgs[2,1]) + ',' + str(expdecay_avgs[2,2]) + ',' + str(expdecay_avgs[2,3]) + ',' + str(expdecay_avgs[2,4]) +'\n' )
mclc_of.write('' + ',' + 'Kurtosis' + ',' + str(expdecay_avgs[3,0]) + ',' + str(expdecay_avgs[3,1]) + ',' + str(expdecay_avgs[3,2]) + ',' + str(expdecay_avgs[3,3]) + ',' + str(expdecay_avgs[3,4]) +'\n' )
mclc_of.write('' + ',' + 'Smoothness' + ',' + str(expdecay_avgs[4,0]) + ',' + str(expdecay_avgs[4,1]) + ',' + str(expdecay_avgs[4,2]) + ',' + str(expdecay_avgs[4,3]) + ',' + str(expdecay_avgs[4,4]) +'\n' )
mclc_of.write('' + ',' + 'Mean to Clipped RMS Ratio' + ',' + str(expdecay_avgs[5,0]) + ',' + str(expdecay_avgs[5,1]) + ',' + str(expdecay_avgs[5,2]) + ',' + str(expdecay_avgs[5,3]) + ',' + str(expdecay_avgs[5,4]) +'\n' )
mclc_of.write('' + ',' + 'Peak to RMS Ratio' + ',' + str(expdecay_avgs[6,0]) + ',' + str(expdecay_avgs[6,1]) + ',' + str(expdecay_avgs[6,2]) + ',' + str(expdecay_avgs[6,3]) + ',' + str(expdecay_avgs[6,4]) +'\n' )
mclc_of.write('' + ',' + 'Peak to Median Absolute Deviation' + ',' + str(expdecay_avgs[7,0]) + ',' + str(expdecay_avgs[7,1]) + ',' + str(expdecay_avgs[7,2]) + ',' + str(expdecay_avgs[7,3]) + ',' + str(expdecay_avgs[7,4]) +'\n' )
#narrow Gaussian FRB
mclc_of.write('Narrow Gaussian FRB' + ',' + 'RMS' + ',' + str(narrowGauss_avgs[0,0]) + ',' + str(narrowGauss_avgs[0,1]) + ',' + str(narrowGauss_avgs[0,2]) + ',' + str(narrowGauss_avgs[0,3]) + ',' + str(narrowGauss_avgs[0,4]) +'\n' )
mclc_of.write('' + ',' + 'RMS Ratio' + ',' + str(narrowGauss_avgs[1,0]) + ',' + str(narrowGauss_avgs[1,1]) + ',' + str(narrowGauss_avgs[1,2]) + ',' + str(narrowGauss_avgs[1,3]) + ',' + str(narrowGauss_avgs[1,4]) +'\n' )
mclc_of.write('' + ',' + 'Skewness' + ',' + str(narrowGauss_avgs[2,0]) + ',' + str(narrowGauss_avgs[2,1]) + ',' + str(narrowGauss_avgs[2,2]) + ',' + str(narrowGauss_avgs[2,3]) + ',' + str(narrowGauss_avgs[2,4]) +'\n' )
mclc_of.write('' + ',' + 'Kurtosis' + ',' + str(narrowGauss_avgs[3,0]) + ',' + str(narrowGauss_avgs[3,1]) + ',' + str(narrowGauss_avgs[3,2]) + ',' + str(narrowGauss_avgs[3,3]) + ',' + str(narrowGauss_avgs[3,4]) +'\n' )
mclc_of.write('' + ',' + 'Smoothness' + ',' + str(narrowGauss_avgs[4,0]) + ',' + str(narrowGauss_avgs[4,1]) + ',' + str(narrowGauss_avgs[4,2]) + ',' + str(narrowGauss_avgs[4,3]) + ',' + str(narrowGauss_avgs[4,4]) +'\n' )
mclc_of.write('' + ',' + 'Mean to Clipped RMS Ratio' + ',' + str(narrowGauss_avgs[5,0]) + ',' + str(narrowGauss_avgs[5,1]) + ',' + str(narrowGauss_avgs[5,2]) + ',' + str(narrowGauss_avgs[5,3]) + ',' + str(narrowGauss_avgs[5,4]) +'\n' )
mclc_of.write('' + ',' + 'Peak to RMS Ratio' + ',' + str(narrowGauss_avgs[6,0]) + ',' + str(narrowGauss_avgs[6,1]) + ',' + str(narrowGauss_avgs[6,2]) + ',' + str(narrowGauss_avgs[6,3]) + ',' + str(narrowGauss_avgs[6,4]) +'\n' )
mclc_of.write('' + ',' + 'Peak to Median Absolute Deviation' + ',' + str(narrowGauss_avgs[7,0]) + ',' + str(narrowGauss_avgs[7,1]) + ',' + str(narrowGauss_avgs[7,2]) + ',' + str(narrowGauss_avgs[7,3]) + ',' + str(narrowGauss_avgs[7,4]) +'\n' )
#broad Gaussian FRB
mclc_of.write('Broad Gaussian FRB' + ',' + 'RMS' + ',' + str(broadGauss_avgs[0,0]) + ',' + str(broadGauss_avgs[0,1]) + ',' + str(broadGauss_avgs[0,2]) + ',' + str(broadGauss_avgs[0,3]) + ',' + str(broadGauss_avgs[0,4]) +'\n' )
mclc_of.write('' + ',' + 'RMS Ratio' + ',' + str(broadGauss_avgs[1,0]) + ',' + str(broadGauss_avgs[1,1]) + ',' + str(broadGauss_avgs[1,2]) + ',' + str(broadGauss_avgs[1,3]) + ',' + str(broadGauss_avgs[1,4]) +'\n' )
mclc_of.write('' + ',' + 'Skewness' + ',' + str(broadGauss_avgs[2,0]) + ',' + str(broadGauss_avgs[2,1]) + ',' + str(broadGauss_avgs[2,2]) + ',' + str(broadGauss_avgs[2,3]) + ',' + str(broadGauss_avgs[2,4]) +'\n' )
mclc_of.write('' + ',' + 'Kurtosis' + ',' + str(broadGauss_avgs[3,0]) + ',' + str(broadGauss_avgs[3,1]) + ',' + str(broadGauss_avgs[3,2]) + ',' + str(broadGauss_avgs[3,3]) + ',' + str(broadGauss_avgs[3,4]) +'\n' )
mclc_of.write('' + ',' + 'Smoothness' + ',' + str(broadGauss_avgs[4,0]) + ',' + str(broadGauss_avgs[4,1]) + ',' + str(broadGauss_avgs[4,2]) + ',' + str(broadGauss_avgs[4,3]) + ',' + str(broadGauss_avgs[4,4]) +'\n' )
mclc_of.write('' + ',' + 'Mean to Clipped RMS Ratio' + ',' + str(broadGauss_avgs[5,0]) + ',' + str(broadGauss_avgs[5,1]) + ',' + str(broadGauss_avgs[5,2]) + ',' + str(broadGauss_avgs[5,3]) + ',' + str(broadGauss_avgs[5,4]) +'\n' )
mclc_of.write('' + ',' + 'Peak to RMS Ratio' + ',' + str(broadGauss_avgs[6,0]) + ',' + str(broadGauss_avgs[6,1]) + ',' + str(broadGauss_avgs[6,2]) + ',' + str(broadGauss_avgs[6,3]) + ',' + str(broadGauss_avgs[6,4]) +'\n' )
mclc_of.write('' + ',' + 'Peak to Median Absolute Deviation' + ',' + str(broadGauss_avgs[7,0]) + ',' + str(broadGauss_avgs[7,1]) + ',' + str(broadGauss_avgs[7,2]) + ',' + str(broadGauss_avgs[7,3]) + ',' + str(broadGauss_avgs[7,4]) +'\n' )
#now on to the bad candidates
#bad periodic
mclc_of.write('Bad Periodic Data' + ',' + 'RMS' + ',' + str(badperiodic_avgs[0,0]) + ',' + str(badperiodic_avgs[0,1]) + ',' + str(badperiodic_avgs[0,2]) + ',' + str(badperiodic_avgs[0,3]) + ',' + str(badperiodic_avgs[0,4]) +'\n' )
mclc_of.write('' + ',' + 'RMS Ratio' + ',' + str(badperiodic_avgs[1,0]) + ',' + str(badperiodic_avgs[1,1]) + ',' + str(badperiodic_avgs[1,2]) + ',' + str(badperiodic_avgs[1,3]) + ',' + str(badperiodic_avgs[1,4]) +'\n' )
mclc_of.write('' + ',' + 'Skewness' + ',' + str(badperiodic_avgs[2,0]) + ',' + str(badperiodic_avgs[2,1]) + ',' + str(badperiodic_avgs[2,2]) + ',' + str(badperiodic_avgs[2,3]) + ',' + str(badperiodic_avgs[2,4]) +'\n' )
mclc_of.write('' + ',' + 'Kurtosis' + ',' + str(badperiodic_avgs[3,0]) + ',' + str(badperiodic_avgs[3,1]) + ',' + str(badperiodic_avgs[3,2]) + ',' + str(badperiodic_avgs[3,3]) + ',' + str(badperiodic_avgs[3,4]) +'\n' )
mclc_of.write('' + ',' + 'Smoothness' + ',' + str(badperiodic_avgs[4,0]) + ',' + str(badperiodic_avgs[4,1]) + ',' + str(badperiodic_avgs[4,2]) + ',' + str(badperiodic_avgs[4,3]) + ',' + str(badperiodic_avgs[4,4]) +'\n' )
mclc_of.write('' + ',' + 'Mean to Clipped RMS Ratio' + ',' + str(badperiodic_avgs[5,0]) + ',' + str(badperiodic_avgs[5,1]) + ',' + str(badperiodic_avgs[5,2]) + ',' + str(badperiodic_avgs[5,3]) + ',' + str(badperiodic_avgs[5,4]) +'\n' )
mclc_of.write('' + ',' + 'Peak to RMS Ratio' + ',' + str(badperiodic_avgs[6,0]) + ',' + str(badperiodic_avgs[6,1]) + ',' + str(badperiodic_avgs[6,2]) + ',' + str(badperiodic_avgs[6,3]) + ',' + str(badperiodic_avgs[6,4]) +'\n' )
mclc_of.write('' + ',' + 'Peak to Median Absolute Deviation' + ',' + str(badperiodic_avgs[7,0]) + ',' + str(badperiodic_avgs[7,1]) + ',' + str(badperiodic_avgs[7,2]) + ',' + str(badperiodic_avgs[7,3]) + ',' + str(badperiodic_avgs[7,4]) +'\n' )
#bad noisy data
mclc_of.write('Bad Noisy Data' + ',' + 'RMS' + ',' + str(badnoisy_avgs[0,0]) + ',' + str(badnoisy_avgs[0,1]) + ',' + str(badnoisy_avgs[0,2]) + ',' + str(badnoisy_avgs[0,3]) + ',' + str(badnoisy_avgs[0,4]) +'\n' )
mclc_of.write('' + ',' + 'RMS Ratio' + ',' + str(badnoisy_avgs[1,0]) + ',' + str(badnoisy_avgs[1,1]) + ',' + str(badnoisy_avgs[1,2]) + ',' + str(badnoisy_avgs[1,3]) + ',' + str(badnoisy_avgs[1,4]) +'\n' )
mclc_of.write('' + ',' + 'Skewness' + ',' + str(badnoisy_avgs[2,0]) + ',' + str(badnoisy_avgs[2,1]) + ',' + str(badnoisy_avgs[2,2]) + ',' + str(badnoisy_avgs[2,3]) + ',' + str(badnoisy_avgs[2,4]) +'\n' )
mclc_of.write('' + ',' + 'Kurtosis' + ',' + str(badnoisy_avgs[3,0]) + ',' + str(badnoisy_avgs[3,1]) + ',' + str(badnoisy_avgs[3,2]) + ',' + str(badnoisy_avgs[3,3]) + ',' + str(badnoisy_avgs[3,4]) +'\n' )
mclc_of.write('' + ',' + 'Smoothness' + ',' + str(badnoisy_avgs[4,0]) + ',' + str(badnoisy_avgs[4,1]) + ',' + str(badnoisy_avgs[4,2]) + ',' + str(badnoisy_avgs[4,3]) + ',' + str(badnoisy_avgs[4,4]) +'\n' )
mclc_of.write('' + ',' + 'Mean to Clipped RMS Ratio' + ',' + str(badnoisy_avgs[5,0]) + ',' + str(badnoisy_avgs[5,1]) + ',' + str(badnoisy_avgs[5,2]) + ',' + str(badnoisy_avgs[5,3]) + ',' + str(badnoisy_avgs[5,4]) +'\n' )
mclc_of.write('' + ',' + 'Peak to RMS Ratio' + ',' + str(badnoisy_avgs[6,0]) + ',' + str(badnoisy_avgs[6,1]) + ',' + str(badnoisy_avgs[6,2]) + ',' + str(badnoisy_avgs[6,3]) + ',' + str(badnoisy_avgs[6,4]) +'\n' )
mclc_of.write('' + ',' + 'Peak to Median Absolute Deviation' + ',' + str(badnoisy_avgs[7,0]) + ',' + str(badnoisy_avgs[7,1]) + ',' + str(badnoisy_avgs[7,2]) + ',' + str(badnoisy_avgs[7,3]) + ',' + str(badnoisy_avgs[7,4]) +'\n' )
#bad 2-levels
mclc_of.write('Bad 2 Level Data' + ',' + 'RMS' + ',' + str(bad2levels_avgs[0,0]) + ',' + str(bad2levels_avgs[0,1]) + ',' + str(bad2levels_avgs[0,2]) + ',' + str(bad2levels_avgs[0,3]) + ',' + str(bad2levels_avgs[0,4]) +'\n' )
mclc_of.write('' + ',' + 'RMS Ratio' + ',' + str(bad2levels_avgs[1,0]) + ',' + str(bad2levels_avgs[1,1]) + ',' + str(bad2levels_avgs[1,2]) + ',' + str(bad2levels_avgs[1,3]) + ',' + str(bad2levels_avgs[1,4]) +'\n' )
mclc_of.write('' + ',' + 'Skewness' + ',' + str(bad2levels_avgs[2,0]) + ',' + str(bad2levels_avgs[2,1]) + ',' + str(bad2levels_avgs[2,2]) + ',' + str(bad2levels_avgs[2,3]) + ',' + str(bad2levels_avgs[2,4]) +'\n' )
mclc_of.write('' + ',' + 'Kurtosis' + ',' + str(bad2levels_avgs[3,0]) + ',' + str(bad2levels_avgs[3,1]) + ',' + str(bad2levels_avgs[3,2]) + ',' + str(bad2levels_avgs[3,3]) + ',' + str(bad2levels_avgs[3,4]) +'\n' )
mclc_of.write('' + ',' + 'Smoothness' + ',' + str(bad2levels_avgs[4,0]) + ',' + str(bad2levels_avgs[4,1]) + ',' + str(bad2levels_avgs[4,2]) + ',' + str(bad2levels_avgs[4,3]) + ',' + str(bad2levels_avgs[4,4]) +'\n' )
mclc_of.write('' + ',' + 'Mean to Clipped RMS Ratio' + ',' + str(bad2levels_avgs[5,0]) + ',' + str(bad2levels_avgs[5,1]) + ',' + str(bad2levels_avgs[5,2]) + ',' + str(bad2levels_avgs[5,3]) + ',' + str(bad2levels_avgs[5,4]) +'\n' )
mclc_of.write('' + ',' + 'Peak to RMS Ratio' + ',' + str(bad2levels_avgs[6,0]) + ',' + str(bad2levels_avgs[6,1]) + ',' + str(bad2levels_avgs[6,2]) + ',' + str(bad2levels_avgs[6,3]) + ',' + str(bad2levels_avgs[6,4]) +'\n' )
mclc_of.write('' + ',' + 'Peak to Median Absolute Deviation' + ',' + str(bad2levels_avgs[7,0]) + ',' + str(bad2levels_avgs[7,1]) + ',' + str(bad2levels_avgs[7,2]) + ',' + str(bad2levels_avgs[7,3]) + ',' + str(bad2levels_avgs[7,4]) +'\n' )
#bad noise change
mclc_of.write('Bad Noise Change Data' + ',' + 'RMS' + ',' + str(badnoisechange_avgs[0,0]) + ',' + str(badnoisechange_avgs[0,1]) + ',' + str(badnoisechange_avgs[0,2]) + ',' + str(badnoisechange_avgs[0,3]) + ',' + str(badnoisechange_avgs[0,4]) +'\n' )
mclc_of.write('' + ',' + 'RMS Ratio' + ',' + str(badnoisechange_avgs[1,0]) + ',' + str(badnoisechange_avgs[1,1]) + ',' + str(badnoisechange_avgs[1,2]) + ',' + str(badnoisechange_avgs[1,3]) + ',' + str(badnoisechange_avgs[1,4]) +'\n' )
mclc_of.write('' + ',' + 'Skewness' + ',' + str(badnoisechange_avgs[2,0]) + ',' + str(badnoisechange_avgs[2,1]) + ',' + str(badnoisechange_avgs[2,2]) + ',' + str(badnoisechange_avgs[2,3]) + ',' + str(badnoisechange_avgs[2,4]) +'\n' )
mclc_of.write('' + ',' + 'Kurtosis' + ',' + str(badnoisechange_avgs[3,0]) + ',' + str(badnoisechange_avgs[3,1]) + ',' + str(badnoisechange_avgs[3,2]) + ',' + str(badnoisechange_avgs[3,3]) + ',' + str(badnoisechange_avgs[3,4]) +'\n' )
mclc_of.write('' + ',' + 'Smoothness' + ',' + str(badnoisechange_avgs[4,0]) + ',' + str(badnoisechange_avgs[4,1]) + ',' + str(badnoisechange_avgs[4,2]) + ',' + str(badnoisechange_avgs[4,3]) + ',' + str(badnoisechange_avgs[4,4]) +'\n' )
mclc_of.write('' + ',' + 'Mean to Clipped RMS Ratio' + ',' + str(badnoisechange_avgs[5,0]) + ',' + str(badnoisechange_avgs[5,1]) + ',' + str(badnoisechange_avgs[5,2]) + ',' + str(badnoisechange_avgs[5,3]) + ',' + str(badnoisechange_avgs[5,4]) +'\n' )
mclc_of.write('' + ',' + 'Peak to RMS Ratio' + ',' + str(badnoisechange_avgs[6,0]) + ',' + str(badnoisechange_avgs[6,1]) + ',' + str(badnoisechange_avgs[6,2]) + ',' + str(badnoisechange_avgs[6,3]) + ',' + str(badnoisechange_avgs[6,4]) +'\n' )
mclc_of.write('' + ',' + 'Peak to Median Absolute Deviation' + ',' + str(badnoisechange_avgs[7,0]) + ',' + str(badnoisechange_avgs[7,1]) + ',' + str(badnoisechange_avgs[7,2]) + ',' + str(badnoisechange_avgs[7,3]) + ',' + str(badnoisechange_avgs[7,4]) +'\n' )
#bad sawtooth 
mclc_of.write('Bad Sawtooth Data' + ',' + 'RMS' + ',' + str(badsawtooth_avgs[0,0]) + ',' + str(badsawtooth_avgs[0,1]) + ',' + str(badsawtooth_avgs[0,2]) + ',' + str(badsawtooth_avgs[0,3]) + ',' + str(badsawtooth_avgs[0,4]) +'\n' )
mclc_of.write('' + ',' + 'RMS Ratio' + ',' + str(badsawtooth_avgs[1,0]) + ',' + str(badsawtooth_avgs[1,1]) + ',' + str(badsawtooth_avgs[1,2]) + ',' + str(badsawtooth_avgs[1,3]) + ',' + str(badsawtooth_avgs[1,4]) +'\n' )
mclc_of.write('' + ',' + 'Skewness' + ',' + str(badsawtooth_avgs[2,0]) + ',' + str(badsawtooth_avgs[2,1]) + ',' + str(badsawtooth_avgs[2,2]) + ',' + str(badsawtooth_avgs[2,3]) + ',' + str(badsawtooth_avgs[2,4]) +'\n' )
mclc_of.write('' + ',' + 'Kurtosis' + ',' + str(badsawtooth_avgs[3,0]) + ',' + str(badsawtooth_avgs[3,1]) + ',' + str(badsawtooth_avgs[3,2]) + ',' + str(badsawtooth_avgs[3,3]) + ',' + str(badsawtooth_avgs[3,4]) +'\n' )
mclc_of.write('' + ',' + 'Smoothness' + ',' + str(badsawtooth_avgs[4,0]) + ',' + str(badsawtooth_avgs[4,1]) + ',' + str(badsawtooth_avgs[4,2]) + ',' + str(badsawtooth_avgs[4,3]) + ',' + str(badsawtooth_avgs[4,4]) +'\n' )
mclc_of.write('' + ',' + 'Mean to Clipped RMS Ratio' + ',' + str(badsawtooth_avgs[5,0]) + ',' + str(badsawtooth_avgs[5,1]) + ',' + str(badsawtooth_avgs[5,2]) + ',' + str(badsawtooth_avgs[5,3]) + ',' + str(badsawtooth_avgs[5,4]) +'\n' )
mclc_of.write('' + ',' + 'Peak to RMS Ratio' + ',' + str(badsawtooth_avgs[6,0]) + ',' + str(badsawtooth_avgs[6,1]) + ',' + str(badsawtooth_avgs[6,2]) + ',' + str(badsawtooth_avgs[6,3]) + ',' + str(badsawtooth_avgs[6,4]) +'\n' )
mclc_of.write('' + ',' + 'Peak to Median Absolute Deviation' + ',' + str(badsawtooth_avgs[7,0]) + ',' + str(badsawtooth_avgs[7,1]) + ',' + str(badsawtooth_avgs[7,2]) + ',' + str(badsawtooth_avgs[7,3]) + ',' + str(badsawtooth_avgs[7,4]) +'\n' )

#close file
mclc_of.close()