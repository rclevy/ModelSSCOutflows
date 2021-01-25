#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: rebeccalevy

ComparePCygniModel.py

Compare model P-Cygni profile to actual profile
This file is run from within ModelOutflowPCygni.py
WARNING: THE BEAM SIZE IS HARD CODED ON LINES 106-107

Change log:
2019-03-08: File created
2019-03-26: Parse suffixes, add elapsed time output
2019-08-15: Don't subtract continuum from model or data
2019-08-20: Tried to hide astropy deprecation warnings but filters are ignored, removed package that was triggering this warning
2019-11-04: Overplot locations of other lines, add secondary x- and y-axes
"""

import argparse

parser=argparse.ArgumentParser(
    description='''Required packages: matplotlib, astropy, numpy, pandas, scipy,spectral_cube''',
    epilog='''Author: R. C. Levy - Last updated: 2019-11-04''')
parser.add_argument('theta',type=float, help='Outflow opening angle (degrees)')
parser.add_argument('inc',type=float,help='Elevation outflow orientation angle (degrees)')
parser.add_argument('phi',type=float,help='Azimuthal outflow orientation angle (degrees)')
parser.add_argument('path',type=str,help='Path to working directory.')
parser.add_argument('--suffix',type=str,help='Suffix of file to use')
parser.add_argument('--line',type=str,help='Line being fit')
parser.add_argument('--SSC_Number',type=str,help='SSC to fit')
parser.add_argument('--ymax',type=float,help='maximum of y-axis in K')
parser.add_argument('--saveplot',type=str,help='save plots (True) or not (False), default False')
args=parser.parse_args()    

#import packages
import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import os
import sys
plt.rcParams['font.family'] = 'serif'


#parameters: theta, phi, inc
theta = args.theta #outflow opening angle, degrees
inc = args.inc #outflow elevation angle, degrees
phi = args.phi #outflow azimuthal angle, degrees
path = args.path
if args.suffix:
	suffix = '_'+args.suffix
else:
	suffix=''
if args.SSC_Number:
	ssc=args.SSC_Number
else:
	ssc='14'
if args.line:
	line = args.line
else:
	line='H13CN(4-3)'
if args.ymax:
	ymax = args.ymax
else:
	ymax=np.nan
if args.saveplot:
	if args.saveplot=='True':
		saveplot=True
	else:
		saveplot=False
else:
	saveplot=False
line_suff = '_'+line
line_str_simple = line.split('(')[0]
line_str_lower = line.lower()
line_str_lower_nosym = line.replace('(','').replace(')','').replace('-','')
line_str_format = line.replace('13','$^{13}$')

transition_data = pd.read_csv(path+'Band7_HighRes_LineID.csv')
idx = transition_data['Line'].values.tolist().index(line)
nu = transition_data['Freq'][idx] #GHz
c=2.9979E5 #km/s

#load SSC data table
if os.path.exists(path+'Levy2020_Table1.csv')==False:
	print('Missing SSC parameter file "'+path+'Levy2020_Table1.csv"! Exiting...')
	sys.exit()
SSC_list=pd.read_csv(path+'Levy2020_Table1.csv')
SSC_num=SSC_list['SSC Number'].values.tolist()

#load spectrum
if os.path.exists(path+'SSCspectra/SSC'+ssc+'_'+line_str_simple+'_spectrum_srcsize_rmforeground.csv')==False:
	print('Measured spectrum file does not exist "'+path+'SSCspectra/SSC'+ssc+'_'+line_str_simple+'_spectrum_srcsize_rmforeground.csv". Exiting....')
	sys.exit()
spec_file = pd.read_csv(path+'SSCspectra/SSC'+ssc+'_'+line_str_simple+'_spectrum_srcsize_rmforeground.csv')
spec=spec_file['spec_mJybeam'].values
espec=spec_file['espec_mJybeam'].values

vel_axis=spec_file['vel_kms'].values
f_axis=spec_file['freq_rest_GHz'].values
dv=vel_axis[0]-vel_axis[1]


#BEAM SIZE IS HARD CODED
bmaj = 0.028 #arcsec
bmin = 0.028 #arcsec
spec = 1.222E3*spec/(nu**2*bmaj*bmin)
espec = 1.222E3*espec/(nu**2*bmaj*bmin)


#remove continuum from spec
#do this "lazily" and just remove the median
cont = np.nanmedian(spec)


#load model spectrum
if 'nomodel' not in suffix:
	model_file = path+'Model_Spec_Tables/ModelSpec_SSC'+ssc+'_'+line+'_theta'+str(theta)+'_inc'+str(inc)+'_phi'+str(phi)+suffix+'.txt'
else:
	suff = suffix.replace('_nomodel','_smoothsphere')
	model_file = path+'Model_Spec_Tables/ModelSpec_SSC'+ssc+'_'+line+'_theta'+str(theta)+'_inc'+str(inc)+'_phi'+str(phi)+suff+'.txt'
spec_model = np.loadtxt(model_file,comments='#')
vel_model = spec_model[:,0]
freq_model = nu*(1-vel_model/c)
spec_model = spec_model[:,1]


#fit the model with 2 gaussians+constant to get velocity range over which to compute rms and chisqr
def fit_func(x,a1,b1,c1,a2,b2,c2,d):
	return a1*np.exp(-((x-b1)**2/(2*c1**2)))+a2*np.exp(-((x-b2)**2/(2*c2**2)))+d
priors = ([-np.inf, -50.0, 0.0, 0.0, 0.0, 0.0, 0.0],
			  [0.0, 0.0, 50.0, np.inf, 50.0, 50.0, np.inf])
p, ep = curve_fit(fit_func, vel_model, spec_model,bounds=priors)

v_rng_model = np.array([p[1]-3*p[2],p[4]+2*p[5]])
v_rng_rms = np.array([np.min(vel_axis),p[1]-3*p[2],np.max(vel_model),np.max(vel_axis)])


#find chisq between data and model
#interpolate the data onto the model velocity axis (since it's shorter)
vel_int = np.arange(np.max(vel_axis),np.min(vel_axis),-dv)
f_data = interp1d(vel_axis,spec,fill_value='extrapolate',kind='zero')
f_model = interp1d(vel_model,spec_model,fill_value='extrapolate',kind='zero')
spec_int = f_data(vel_int)
spec_model_int = f_model(vel_int)
idx_signal = np.intersect1d(np.where(vel_int >= v_rng_model[0])[0],np.where(vel_int <= v_rng_model[1])[0])
if line=='H13CN(4-3)':
	idx_rms = np.where(vel_axis >= v_rng_rms[2])[0]
else:
	idx_rms = np.concatenate([np.intersect1d(np.where(vel_axis >= v_rng_rms[2])[0],np.where(vel_axis <= v_rng_rms[3])[0]),
		np.intersect1d(np.where(vel_axis >= v_rng_rms[0])[0],np.where(vel_axis <= v_rng_rms[1])[0])])

	

rms = np.sqrt(np.nanmean((spec[idx_rms]-cont)**2))   
chisq = np.nansum((spec_model_int[idx_signal]-spec_int[idx_signal])**2/rms**2)
chisqr = chisq/len(vel_int[idx_signal])

#write chisqr to text log
if os.path.isdir(path+'ParameterDegeneracies')==False:
	os.system('mkdir '+path+'ParameterDegeneracies')
with open(path+'ParameterDegeneracies/this_chisqr.txt','w') as fp:
	fp.write(str(chisqr))

#get min and max of velocity axis
vmin=-450
vmax=450

#plot spectrum and model
fill_col = 'gold'
text_col = 'darkgoldenrod'
edge_col = 'w'
spec[spec==0]=np.nan
plt.close()
plt.figure(1)
plt.clf()
plt.fill_between(vel_axis,spec-espec,spec+espec,
		facecolor='darkgray',edgecolor=None,step='pre',zorder=5)
plt.step(vel_axis,spec,'k-',
         label=r'SSC '+ssc+' Spectrum:\nrms = %.1f K' %rms,zorder=6)
if 'nomodel' not in suffix:
	plt.plot(vel_int,spec_model_int,'r-',lw=0.75,zorder=7)
	plt.plot(vel_model,spec_model,'r-',
		label=r'Model: $\chi^2_r$ = %.1f' %chisqr,zorder=7)
	vfrac = 0.033
	plt.plot([vel_model[0],vel_model[0]],[spec_model[0]*(1-vfrac),spec_model[0]*(1+vfrac)],'r-',zorder=7)
	plt.plot([vel_model[-1],vel_model[-1]],[spec_model[-1]*(1-vfrac),spec_model[-1]*(1+vfrac)],'r-',zorder=7)

plt.xlabel('Velocity (km s$^{-1}$)')
plt.ylabel('T$_{\mathrm{B}}$ (K)')
plt.ylim(bottom=0.)#,top=np.max(spec_model)*1.25)
if np.isnan(ymax)==False:
	plt.ylim(top=ymax)

if line=='H13CN(4-3)':
	#overplot other lines
	pad = 5. #km/s, channel size
	plt.axvline(x=0.,color='dimgray',linestyle='--',linewidth=1.,zorder=1)
	plt.text(0,plt.gca().get_ylim()[1]*0.99,'H$^{13}$CN 4-3',va='top',ha='center',fontsize=7.,color='dimgray',backgroundcolor='w')
	other_lines = {
		# '$^{34}$SO$_2$ 15-15':344.98758470,
		# '$^{34}$SO$_2$ 11-11':344.99816020,
		# #'CH$_3$OCHO 47-47':345.00410700,
		# #'SO$_2$ v$_2$=2':345.01712700,
		# #'CH$_3$OCHO 28$_{14,14}$-27$_{14,13}$':345.06905900,
		# #'SO$_2$ v$_1$=1 19-18':345.08676900,
		# #'CH$_3$OCHO 28$_{14,15}$-27$_{14,14}$':345.09148400,
		# #'SO$_2$ v$_1$=1 24-23':345.09177900,
		# 'SO$_2$ 5-6':345.14895900,
		# # '$^{34}$SO$_2$ 8-8':345.16866300,
		#  'H$^{13}$CN 4-3 v$_{2}$=1': 345.23871100,
		# # #'CH$_{3}$OCN 34$_{9,25}$-33$_{9,24}$':345.26528880,
		# # '$^{34}$SO$_2$ 9$_{4}$-9$_{3}$':345.28562,
		# #'CH$_3$OCN 34$_{11,23}$-33$_{11,22}$':345.32048650,
		# # 'SO$_2$ 26-27':345.44898150,
		# #'CH$_{3}$OCHO 28$_{13}$-27$_{13}$':345.46699000,
		# #'CH$_{3}$OCHO 28$_{13,16}$-27$_{13,15}$':345.48666100,
		# #'CH$_3$OCHO 15-15':345.54761600,
		# # '$^{34}$SO$_2$ 6$_{4,2}$-6$_{3,3}$':345.55309270,
		#  'HC$_3$N 38-37':345.60901000,
		# 'HC$_3$N 38-37 v$_5$=1':345.63212660,
		# #'CH$_3$OCHO v=1':345.64057120,
		# '$^{34}$SO$_2$ 5$_{4,2}$-5$_{3,3}$':345.65129000,
		# '$^{34}$SO$_2$ 4$_{4,0}$-4$_{3,1}$':345.67878400,
		#'CO 3-2':345.79598990
	}
	for ll in other_lines:
		plt.axvline(x=c*(1-other_lines[ll]/nu),color='gray',linestyle='--',linewidth=1.,zorder=1)
	# 	plt.text(c*(1-other_lines[ll]/nu)+pad,plt.gca().get_ylim()[1]*0.01,ll,rotation=90.,va='bottom',ha='left',fontsize=5.,color='dimgray')

	# plt.axvline(x=c*(1-345.16866600/nu),color='gray',linestyle='--',linewidth=1.,zorder=1) 
	# plt.text(c*(1-345.16866600/nu)+pad,plt.gca().get_ylim()[0]/1.5,'$^{34}$SO$_{2}$ 8(4,4)-8(3,5)',rotation=90.,va='bottom',ha='left',fontsize=6.,color='gray')
	# plt.axvline(x=c*(1-345.14896960/nu),color='gray',linestyle='--',linewidth=1.,zorder=1) 
	# plt.text(c*(1-345.14896960/nu)+pad,plt.gca().get_ylim()[0]/1.5,'SO$_{2}$ 5(5,1)-6(4,2)',rotation=90.,va='bottom',ha='left',fontsize=6.,color='gray')
	# plt.axvline(x=c*(1-345.10476260/nu),color='gray',linestyle='--',linewidth=1.,zorder=1) 
	# plt.text(c*(1-345.10476260/nu)+pad,plt.gca().get_ylim()[0]/1.5,'CH$_3$OCHO 47-47',rotation=90.,va='bottom',ha='left',fontsize=6.,color='gray')
	# plt.axvline(x=c*(1-345.09177900/nu),color='gray',linestyle='--',linewidth=1.,zorder=1)
	# plt.text(c*(1-345.09177900/nu)+pad,plt.gca().get_ylim()[0]/1.5,'SO$_{2}$ 24(2,22)-23(3,21) v$_1$=1',rotation=90.,va='bottom',ha='left',fontsize=6.,color='gray')
	# plt.axvline(x=c*(1-345.12253480 /nu),color='gray',linestyle='--',linewidth=1.,zorder=1)
	# plt.text(c*(1-345.12253480 /nu)+pad,plt.gca().get_ylim()[0]/1.5,'HC$_{3}$N 38-37 v$_{4}$=1',rotation=90.,va='bottom',ha='left',fontsize=6.,color='gray')

	plt.axvspan(c*(1-346.0/nu),c*(1-345.71/nu),color=fill_col,ec=edge_col,alpha=0.2,zorder=1)
	plt.text(0.5*(vmin+c*(1-345.71/nu)),0,'CO',fontsize=6,va='bottom',ha='center',color=text_col)
	plt.axvspan(c*(1-345.67878400/nu),c*(1-345.55309270/nu),color=fill_col,ec=edge_col,alpha=0.2,zorder=1)
	plt.text(c*(1-345.616/nu),0,'HC$_3$N\nHC$_3$N v$_5$=1\n$^{34}$SO$_2$\nCH$_3$OCHO',fontsize=6,va='bottom',ha='center',color=text_col)
	plt.axvspan(c*(1-345.44/nu),c*(1-345.5/nu),color=fill_col,ec=edge_col,alpha=0.2,zorder=1)
	plt.text(c*(1-345.47/nu),0,'SO$_2$\nCH$_3$OCHO',fontsize=6,va='bottom',ha='center',color=text_col)
	plt.axvspan(c*(1-345.22/nu),c*(1-345.28/nu),color=fill_col,ec=edge_col,alpha=0.2,zorder=1)
	plt.text(c*(1-345.25/nu),0,'H$^{13}$CN v$_2$=1\nCH$_3$OCHO\n$^{34}$SO$_2$',fontsize=6,va='bottom',ha='center',color=text_col)
	plt.axvspan(c*(1-344.98/nu),c*(1-345.17/nu),color=fill_col,ec=edge_col,alpha=0.2,zorder=1)
	plt.text(c*(1-345.075/nu),0,'SO$_2$\n$^{34}$SO$_2$\nCH$_3$OCHO',fontsize=6,va='bottom',ha='center',color=text_col)


elif line=='HCN(4-3)':
	pad = 3.
	plt.axvline(x=0.,color='dimgray',linestyle='--',linewidth=1.,zorder=1)
	plt.text(0,plt.gca().get_ylim()[1]*0.99,'HCN 4-3',va='top',ha='center',fontsize=7.,color='dimgray',backgroundcolor='w')
	other_lines = {
		#'$^{33}$SO$_2$':354.24381770,
		#'HCO$^+$ 4-3 v$_3$=1':354.37769960,
		# 'CH$_3$OCHO v=1':354.42726700,
		# '$^{13}$CH$_{3}$OH 4-3':354.44595300,
		# 'HCN 4-3 v$_2$=1 $\ell$=1e':354.46043300,
		#'CH$_3$OCHO v=1':354.47735200,
		#'CH$_3$OCHO v=1': 354.57396900,
		#'CH$_{3}$OCHO 33-32':354.60799600,
		#'SO$_2$ v$_2$=1': 354.62412000,
		#'CH$_3$OCHO v=1':354.67451780,
		#'HC$_3$N 39-38':354.69745600,
		#'CH$_{3}$OCHO 12-11':354.74241400,
		#'SO$_2$ v$_2$=1':354.79999200,
		#'OS$^{17}$O':354.79090180
	}

	# for ll in other_lines:
	# 	plt.axvline(x=c*(1-other_lines[ll]/nu),color='gray',linestyle='--',linewidth=1.,zorder=1)
	# 	plt.text(c*(1-other_lines[ll]/nu)+pad,plt.gca().get_ylim()[1]*0.05,ll,rotation=90.,va='bottom',ha='left',fontsize=6.,color='dimgray')

	#plt.axvline(x=c*(1-354.41048300/nu),color='gray',linestyle='--',linewidth=1.,zorder=1)
	#plt.text(c*(1-354.41048300/nu)+pad,plt.gca().get_ylim()[1]*0.2,'CH$_3$O$^{13}$CHO 56-56 v$_t$=1-1',rotation=90.,va='bottom',ha='left',fontsize=6.,color='dimgray')
	#plt.axvline(x=c*(1-354.40827260/nu),color='gray',linestyle='--',linewidth=1.,zorder=1)
	#plt.text(c*(1-354.40827260/nu)+pad,plt.gca().get_ylim()[1]*0.2,'H$_2$SO$_4$ 35-34',rotation=90.,va='bottom',ha='left',fontsize=6.,color='dimgray')
	#plt.axvline(x=c*(1-354.40415490/nu),color='gray',linestyle='--',linewidth=1.,zorder=1)
	#plt.text(c*(1-354.40415490/nu)+pad,plt.gca().get_ylim()[1]*0.2,'CH$_3$OCHO 68-68 v=1',rotation=90.,va='bottom',ha='left',fontsize=6.,color='dimgray')
	#plt.axvline(x=c*(1-354.39388500/nu),color='gray',linestyle='--',linewidth=1.,zorder=1)
	#plt.text(c*(1-354.39388500/nu)+pad,plt.gca().get_ylim()[1]*0.2,'H$_{2}^{13}$CO 18-19',rotation=90.,va='bottom',ha='left',fontsize=6.,color='dimgray')
	#plt.axvline(x=c*(1-354.39774900/nu),color='gray',linestyle='--',linewidth=1.,zorder=1)
	#plt.text(c*(1-354.39774900/nu)+pad,plt.gca().get_ylim()[1]*0.2,'$^{34}$SO$_2$ 19-20',rotation=90.,va='bottom',ha='left',fontsize=6.,color='dimgray')
	# plt.axvline(x=c*(1-354.57170300/nu),color='gray',linestyle='--',linewidth=1.,zorder=1)
	# plt.text(c*(1-354.57170300/nu)+pad,plt.gca().get_ylim()[1]*0.2,'CH$_3$O$^{13}$CHO',rotation=90.,va='bottom',ha='left',fontsize=6.,color='dimgray')
	# plt.axvline(x=c*(1-354.56869200/nu),color='gray',linestyle='--',linewidth=1.,zorder=1)
	# plt.text(c*(1-354.56869200/nu)+pad,plt.gca().get_ylim()[1]*0.2,'CH$_3$O$^{13}$CHO',rotation=90.,va='bottom',ha='left',fontsize=6.,color='dimgray')
	# plt.axvline(x=c*(1-354.55748700/nu),color='gray',linestyle='--',linewidth=1.,zorder=1)
	# plt.text(c*(1-354.55748700/nu)+pad,plt.gca().get_ylim()[1]*0.2,'CH$_3$O$^{13}$CHO',rotation=90.,va='bottom',ha='left',fontsize=6.,color='dimgray')
	# plt.axvline(x=c*(1-354.54653500/nu),color='gray',linestyle='--',linewidth=1.,zorder=1)
	# plt.text(c*(1-354.54653500/nu)+pad,plt.gca().get_ylim()[1]*0.2,'CH$_3$O$^{13}$CHO',rotation=90.,va='bottom',ha='left',fontsize=6.,color='dimgray')
	# plt.axvline(x=c*(1-354.54145100/nu),color='gray',linestyle='--',linewidth=1.,zorder=1)
	# plt.text(c*(1-354.54145100/nu)+pad,plt.gca().get_ylim()[1]*0.2,'CH$_3$O$^{13}$CHO',rotation=90.,va='bottom',ha='left',fontsize=6.,color='dimgray')


	plt.axvspan(c*(1-354.740/nu),c*(1-354.625/nu),color=fill_col,ec=edge_col,alpha=0.2,zorder=1)
	plt.text(c*(1-354.6825/nu),0,'HC$_3$N\nSO$_2$',fontsize=6,va='bottom',ha='center',color=text_col)
	plt.axvspan(c*(1-354.579/nu),c*(1-354.60/nu),color=fill_col,ec=edge_col,alpha=0.2,zorder=1)
	plt.text(c*(1-354.589/nu),0,'CH$_3$OCHO\nSiC',fontsize=6,va='bottom',ha='center',color=text_col)
	plt.axvspan(c*(1-354.41/nu),c*(1-354.48/nu),color=fill_col,ec=edge_col,alpha=0.2,zorder=1)
	plt.text(c*(1-354.445/nu),0,'HCN v$_2$=1\n$^{13}$CH$_3$OH',fontsize=6,va='bottom',ha='center',color=text_col)
	# plt.axvspan(c*(1-354.3675/nu),c*(1-354.3875/nu),color='gray',ec='lightgray',alpha=0.2,hatch='///')
	# plt.text(c*(1-354.3775/nu),0,'HCO$^+$ v$_3$=1',fontsize=6,va='bottom',ha='center',color='gray')
	plt.axvspan(c*(1-354.225/nu),c*(1-354.275/nu),color=fill_col,ec=edge_col,alpha=0.2,zorder=1)
	plt.text(c*(1-354.25/nu),0,'$^{33}$SO$_2$',fontsize=6,va='bottom',ha='center',color=text_col)
	# plt.axvspan(c*(1-354.024/nu),c*(1-353.914/nu),color='gray',ec='lightgray',alpha=0.2,hatch='///')
	# plt.text(c*(1-354.014/nu),0,'CO$^+$',fontsize=6,va='bottom',ha='center',color='gray')


elif line=='CS(7-6)':
	pad = 2.
	plt.axvline(x=0.,color='dimgray',linestyle='--',linewidth=1.,zorder=1)
	plt.text(0.,plt.gca().get_ylim()[1]*0.99,'CS 7-6',va='top',ha='center',fontsize=7.,color='dimgray',backgroundcolor='w')
	other_lines = {
		# 'CO 3-2 v=1': 342.64765600,
		# 'CH$_3$CH$_2$CN':342.67756430,
		# 'CH$_3$OCHO': 342.69291300,
		# 'CH$_3$OH v$_t$=0-2': 342.72978100,
		 'SO$_2$': 342.76162320,
		# #'UNID':342.77800000,
		# 'SO$_2$ v$_2$=1': 342.79160660,
		# 'SiC$_2$': 342.80495000,
		# #'CH$_3$OCHO v=1':342.81558960,
		# 'SiC$_2$ v$_3$=1':342.81638900,
		# '$^{33}$SO$_2$':342.82086720,
		 '$^{33}$SO$_2$':342.82951320,
		# #'NH$_3$ v$_2$=1':342.86536220, #splatalogue intensity = -8.6
		#  'H$_2$CS':342.94645600,
		# #'HeII 42$\\alpha$':342.89323110,
		 'OS$^{18}$O':342.89444110,
		# 'CH$_3$OCHO': 342.96817120,
		# '$^{29}$SiO 8-7': 342.98084250
		'$^{33}$SO':343.08684700,
		#'SiS v=1': 343.10097600,
		'H$_2$CS': 343.20188000,
		'H$_2$CS 10-9': 343.30850030
	}
	# for ll in other_lines:
	# 	plt.axvline(x=c*(1-other_lines[ll]/nu),color='gray',linestyle='--',linewidth=1.,zorder=1)
	# 	plt.text(c*(1-other_lines[ll]/nu)+pad,plt.gca().get_ylim()[1]*0.05,ll,rotation=90.,va='bottom',ha='left',fontsize=6.,color='dimgray')

	plt.axvspan(c*(1-343.34/nu),c*(1-343.30/nu),color=fill_col,ec=edge_col,alpha=0.2,zorder=1)
	plt.text(c*(1-343.32/nu),0,'H$_2$CS\nH$_2^{13}$CO\nH$_2$C$^{17}$O',fontsize=6,va='bottom',ha='center',color=text_col)
	plt.axvspan(c*(1-343.18/nu),c*(1-343.22/nu),color=fill_col,ec=edge_col,alpha=0.2,zorder=1)
	plt.text(c*(1-343.20/nu),0,'OS$^{17}$O\nNH$_2$CHO\nH$_2$CS\nCH$_3$OH',fontsize=6,va='bottom',ha='center',color=text_col)
	plt.axvspan(c*(1-343.14/nu),c*(1-343.06/nu),color=fill_col,ec=edge_col,alpha=0.2,zorder=1)
	plt.text(c*(1-343.1/nu),0,'SiS v=1\n$^{33}$SO',fontsize=6,va='bottom',ha='center',color=text_col)
	plt.axvspan(c*(1-342.98/nu),c*(1-342.946/nu),color=fill_col,ec=edge_col,alpha=0.2,zorder=1)
	plt.text(c*(1-342.968/nu),0,'$^{29}$SiO 8-7\nCH$_3$OCHO\nH$_2$CS',fontsize=6,va='bottom',ha='center',color=text_col)
	plt.axvspan(c*(1-342.825/nu),c*(1-342.725/nu),color=fill_col,ec=edge_col,alpha=0.2,zorder=1)
	plt.text(c*(1-342.775/nu),0,'$^{33}$SO$_2$\nSO$_2$\nCH$_3$OH',fontsize=6,va='bottom',ha='center',color=text_col)
	plt.axvspan(c*(1-342.48/nu),c*(1-342.625/nu),color=fill_col,ec=edge_col,alpha=0.2,zorder=1)
	plt.text(c*(1-342.5525/nu),0,'CH$_{3}$OCHO\nSiO v=2?\nOS$^{17}$O\nCH$_3$OH',fontsize=6,va='bottom',ha='center',color=text_col)
	plt.axvspan(c*(1-342.397/nu),c*(1-342.417/nu),color=fill_col,ec=edge_col,alpha=0.2,zorder=1)
	plt.text(c*(1-342.407/nu),0,'HNCO?',fontsize=6,va='bottom',ha='center',color=text_col)
	if ssc=='4a':
		plt.axvspan(c*(1-342.889/nu),c*(1-342.899/nu),color=fill_col,ec=edge_col,alpha=0.2,zorder=1)
		plt.text(c*(1-342.89444110/nu),0,'OS$^{18}$O',fontsize=6,va='bottom',ha='center',color=text_col)

plt.minorticks_on()
leg=plt.legend(fontsize=8,loc='upper right')
plt.ylim(bottom=0.)#,top=np.max(spec_model)*1.25)
if np.isnan(ymax)==False:
	plt.ylim(top=ymax)


#add second x-axis with rest freq
ax1=plt.gca()
xt1=ax1.get_xticks()
ax1.set_xticks(xt1)
ax2=ax1.twiny()
xt2=nu*(1-xt1/c)
xt2_re=np.arange(np.round(xt2[0],1),np.round(xt2[-1],1)-0.1,-0.1)
ax2.set_xticks(c*(1-xt2_re/nu))
ax1.set_xlim(left=c*(1-xt2_re[0]/nu),right=c*(1-xt2_re[-1]/nu))
ax2.set_xticklabels(np.round(xt2_re,2))
#ax2.set_xticks(xt1)
#ax2.set_xticklabels(np.round(xt2,2))
ax2.set_xlabel('Rest Frequency (GHz)')
ax1.set_xlim(left=vmin,right=vmax)
ax2.set_xlim(left=vmin,right=vmax)
ax2.minorticks_on()


#add second y-axis with intensity (mJy/beam)
def I2T(x):
	return 1.222E3*x/(nu**2*bmaj*bmin)
def T2I(x):
	return x*nu**2*bmaj*bmin/(1.222E3)


ax3 = ax1.secondary_yaxis('right',functions=(T2I,I2T))
ax3.set_ylabel('Intensity (mJy beam$^{-1}$)')
ax3.minorticks_on()
#plt.show()
if saveplot==True:
	if os.path.isdir(path+'Data_Comp/')==False:
		os.system('mkdir '+path+'Data_Comp/')
	if 'nomodel' not in suffix:
		plt.savefig(path+'Data_Comp/ModelComp_SSC'+ssc+'_'+line+'_theta'+str(theta)+'_inc'+str(inc)+'_phi'+str(phi)+suffix+'.pdf')
	else:
		plt.savefig(path+'Data_Comp/NoModel/ModelComp_SSC'+ssc+'_'+line+'_theta'+str(theta)+'_inc'+str(inc)+'_phi'+str(phi)+'_nomodel.pdf')


#open Gaussian fit and add it to the plot
leg.remove() #remove old legend
if os.path.exists(path+'ProfileFitParams/SSC'+ssc+'_PCygni_'+line+'_fitparams.txt')==False:
	print('Cannot find fits to spectral profile "'+path+'ProfileFitParams/SSC'+ssc+'_PCygni_'+line+'_fitparams.txt". Exiting...')
	sys.exit()
a1,b1,c1,a2,b2,c2,d = np.loadtxt(path+'ProfileFitParams/SSC'+ssc+'_PCygni_'+line+'_fitparams.txt',comments='#')[0]
gfit = a1*np.exp(-((vel_axis-b1)**2/(2*c1**2)))+a2*np.exp(-((vel_axis-b2)**2/(2*c2**2)))+d
gfit_K = 1.222E3*gfit/(nu**2*bmaj*bmin)
#compute chisqr between data and fit
chisq = np.nansum((gfit_K-spec)**2/rms**2)
chisqr = chisq/(len(spec)-7)
ax1.plot(vel_axis,gfit_K,'b--',zorder=8,label='Gaussian Fit: $\chi^2_r$ = %.1f' %chisqr)
ax1.legend(loc='upper right')
if saveplot==True:
	if os.path.isdir(path+'Data_Comp/GaussFit/')==False:
		os.system('mkdir '+path+'Data_Comp/GaussFit')
	if 'nomodel' not in suffix:
		plt.savefig(path+'Data_Comp/GaussFit/ModelGaussComp_SSC'+ssc+'_'+line+'_theta'+str(theta)+'_inc'+str(inc)+'_phi'+str(phi)+suffix+'.pdf')
	else:
		plt.savefig(path+'Data_Comp/NoModel/ModelGaussComp_SSC'+ssc+'_'+line+'_theta'+str(theta)+'_inc'+str(inc)+'_phi'+str(phi)+'_nomodel.pdf')

plt.close()

