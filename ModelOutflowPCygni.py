import argparse

parser=argparse.ArgumentParser(
	prog='ModelOutflowPCygni',
	formatter_class=argparse.RawDescriptionHelpFormatter,
	description=r'''
	Model the spectrum through an outflow with a given opening angle and orientation along the line of sight. For more details, see Levy et al. 2020, ApJ, submitted (arXiv:2011.05334).

	Parameters
	----------
	ssc : str
		name of SSC to fit
	line : str
		name of line to fit, must match path+'/Band7_HighRes_LineID.csv', currently only CS(7-6) and H13CN(4-3) are fully supported
	path : str
		path name of locations of input files and line list, include / at end

	Returns
	_______
	None

	
	Notes
	_____
	Required packages: argparse, datetime, matplotlib, mpl_toolkits, numpy, pandas, os, time, scipy
	Author: R. C. Levy (rlevy.astro@gmail.com)
	Please cite: Levy et al. 2020, ApJ, submitted (arXiv:2011.05334)
	Last updated: 2021-01-25
	
	Change log:
	2019-03-11: Implemented function to mask continuum source of a given size, removed non-spectral functions, RCL
	2019-03-18: Implemented function to make (hard,sharp) spherical continuum source and smooth continuum source. Added continuum properties to intensity calculation, RCL
	2019-03-25: Added test capability to add suffix to output files, RCL
	2019-03-26: Change the way tau_cont is defined, RCL
	2019-05-07: Updated Tex to correct values (from position changes), RCL
	2019-08-19: Changed the way the continuum tau is calculated, major restructuring for integration into MCMC, RCL
	2019-08-21: Changed that get output when saveplots=False (for speed in MCMC), RCL
	2019-08-27: Something with the continuum optical depth is wrong, RCL
	2019-08-28: Fixing continuum optical depth calculation, split saveplots keyword into savemodel and save3dplots, added printdiagnostics keyword, RCL
	2019-09-17: Box is twice continuum source size, no outflow or gas inside continuum source, no continuum beyond continuum source size, RCL
	2020-03-04: Fixed continuum source masking problems
	2020-04-06: Fixed error with excitation temperature assigment, got rid of bump feature, RCL
	2020-04-07: Add hot diffuse gas component inside continuum source, RCL
	2020-04-28: Add a fast, cold outflow component, RCL
	2020-04-29: Add plotting for spectrum off source, add save as fits cube, RCL
	2021-01-25: Cleaned up for publication, RCL
	'''
,
	epilog='''Author: R. C. Levy - Last updated: 2021-01-25''')
parser.add_argument('ssc',type=str,help='SSC to model.')
parser.add_argument('line',type=str,help='name of line to fit, must match path+"/Band7_HighRes_LineID.csv"')
parser.add_argument('path',type=str,help='path name of locations of input files and line list')
args=parser.parse_args()

ssc = args.ssc
line = args.line
global path
path = args.path


def main(ssc,line,path):
	r'''
	load model parameters from file
	parameter file must include the following keywords and comma separated values

	Parameters
	----------
	ssc : str
		name of SSC to fit
	line : str
		name of line to fit, must match PATH+'/Band7_HighRes_LineID.csv'
	theta : float
		full opening angle of outflow (degrees)
	inc : float
		inclination of the outflow relative to the line of sight in the vertical direction (degrees)
	phi : float
		orientation of the outflow relative to the line of sight in the horizontal direction (degrees)
	v_outflow : float
		velocity of the outflow in the simulation, assume constant (km/s)
	v_outflow_FWHM : float
		FWHM velocity dispersion in the outflow, assumed constant (km/s)
	n_cen : float
		central H2 volume density, density ~ 1/r^2 (cm^-3)
	cont_type : str
		choose the model for the continuum source, only smoothsphere fully supported
	T_cont : float
		central continuum source temperature (K)
	T_ex : float
		measured gas excitation temperature, taken from HCN 4-3 line (K)
	bmaj : float
		beam FWHM major axis of observations (arcsec)
	bmin : float
		beam FWHM minor axis of observations (arcsec)
	source_sz_arcsec : float
		measured FWHM size of the continuum source, this sets the size of the box in the model (arcsec)
	savemodel : boolean
		boolean to set whether the model is saved to a text file and plotted over data
	save3dplots : boolean
		boolean to set whether output 3D plots are plotted and saved
	printdiagnostics : boolean
		boolean to set whether diagnostic print statements are printed or not

	'''
	import pandas as pd
	import os
	import sys

	#check if path exists
	if os.path.isdir(path)==False:
		print('Path does not exist! I will create it, but other paths may be wrong!')
		os.system('mkdir '+path)

	line_str = line.split('(')[0]

	#check if path to parameter files exists
	if os.path.isdir(path+'Model_Params/')==False:
		print('Directoey to input files "'+path+'Model_Params/" does not exist! I will create it, but other paths may be wrong!')
		os.system('mkdir '+path+'Model_Params/')

	#open the input parameter file
	input_file_name=path+'Model_Params/SSC'+ssc+'_'+line_str+'_modelparams.txt'
	if os.path.exists(input_file_name)==False:
		print('Parameter file expected at "'+input_file_name+'" but not found! Exiting...')
		sys.exit()
	params = pd.read_csv(input_file_name)	

	ssc=params['value'].values[params['param'].values.tolist().index('ssc')]
	line=params['value'].values[params['param'].values.tolist().index('line')]
	theta=float(params['value'].values[params['param'].values.tolist().index('theta')])
	inc=float(params['value'].values[params['param'].values.tolist().index('inc')])
	phi=float(params['value'].values[params['param'].values.tolist().index('phi')])
	v_outflow=float(params['value'].values[params['param'].values.tolist().index('v_outflow')])
	v_outflow_fast=float(params['value'].values[params['param'].values.tolist().index('v_outflow_fast')])
	v_outflow_FWHM=float(params['value'].values[params['param'].values.tolist().index('v_outflow_FWHM')])
	v_hot_FWHM=float(params['value'].values[params['param'].values.tolist().index('v_hot_FWHM')])
	v_outflow_fast_FWHM=float(params['value'].values[params['param'].values.tolist().index('v_outflow_fast_FWHM')])
	n_cen=float(params['value'].values[params['param'].values.tolist().index('n_cen')])
	n_cen_fast=float(params['value'].values[params['param'].values.tolist().index('n_cen_fast')])
	cont_type=params['value'].values[params['param'].values.tolist().index('cont_type')]
	T_ex=float(params['value'].values[params['param'].values.tolist().index('T_ex')])
	T_hot=float(params['value'].values[params['param'].values.tolist().index('T_hot')])
	n_cen_hot=float(params['value'].values[params['param'].values.tolist().index('n_cen_hot')])
	T_cont=float(params['value'].values[params['param'].values.tolist().index('T_cont')])
	tau_cont_max=float(params['value'].values[params['param'].values.tolist().index('tau_cont_max')])
	source_sz_arcsec=float(params['value'].values[params['param'].values.tolist().index('source_sz_arcsec')])
	bmaj=float(params['value'].values[params['param'].values.tolist().index('bmaj')])
	bmin=float(params['value'].values[params['param'].values.tolist().index('bmin')])
	savemodel=params['value'].values[params['param'].values.tolist().index('savemodel')]
	save3dplots=params['value'].values[params['param'].values.tolist().index('save3dplots')]
	printdiagnostics=params['value'].values[params['param'].values.tolist().index('printdiagnostics')]
	if savemodel=='True':
		savemodel=True
	else:
		savemodel=False
	if save3dplots=='True':
		save3dplots=True
	else:
		save3dplots=False
	if printdiagnostics=='True':
		printdiagnostics=True
	else:
		printdiagnostics=False

	#run the modeling
	model_outflow_pcygni(ssc,line,theta,inc,phi,
		v_outflow,v_outflow_fast,v_outflow_FWHM,v_outflow_fast_FWHM,n_cen,n_cen_fast,cont_type,T_cont,tau_cont_max,
		T_ex,T_hot,n_cen_hot,v_hot_FWHM,
		source_sz_arcsec,bmaj,bmin,
		savemodel,save3dplots,printdiagnostics)



def model_outflow_pcygni(ssc,line,theta,inc,phi,v_outflow,v_outflow_fast,v_outflow_FWHM,v_outflow_fast_FWHM,n_cen,n_cen_fast,cont_type,T_cont,tau_cont_max,T_ex,T_hot,n_cen_hot,v_hot_FWHM,source_sz_arcsec,bmaj,bmin,savemodel,save3dplots,printdiagnostics):

	#import modules
	import numpy as np
	import matplotlib.pyplot as plt
	plt.rcParams['font.family'] = 'serif'
	plt.rcParams['mathtext.rm'] = 'serif'
	plt.rcParams['mathtext.fontset'] = 'cm'
	from matplotlib.colors import LogNorm, SymLogNorm, Normalize, LinearSegmentedColormap, BoundaryNorm
	from matplotlib import cm
	from mpl_toolkits.mplot3d import Axes3D 
	from scipy.ndimage.interpolation import rotate
	from scipy.interpolate import interp1d
	from scipy.interpolate import RegularGridInterpolator as rgi 
	from scipy.integrate import cumtrapz
	from scipy.special import erf
	import datetime
	import time
	import os
	import sys
	import pandas as pd
	np.seterr(divide='ignore',invalid='ignore')
	start = time.time()

	############################
	##### DEFINE FUNCTIONS #####
	############################
	def construct_density(r_cont_src_pc,n_cen,rr,rr_pc,box_sz):
		#make the 3D density profile of the cold, outflowing component

		#H2 density goes like 1/r(pc)^2 outside continuum source
		rr_pc_cont_shift = rr_pc.copy()
		rr_pc_cont_shift[rr_pc_cont_shift<r_cont_src_pc]=0.0
		n = 1/rr_pc_cont_shift**2
		n[np.isinf(n)==True]=np.nan
		n = n/np.nanmax(n)
		density = n*n_cen #cm^-3

		#check if path exists
		if os.path.isdir(path+'Model_DensityProfile/')==False:
			print('Directory "'+path+'Model_DensityProfile/" does not exist. I will create it.')
			os.system('mkdir '+path+'Model_DensityProfile/')

		plt.figure(3)
		plt.clf()
		plt.plot(rr_pc[r_cen,r_cen,r_cen:],density[r_cen,r_cen,r_cen:],'s-',color='b',label='x=y=0, z<0',lw=2.0)
		plt.plot(rr_pc[r_cen,r_cen,0:r_cen],density[r_cen,r_cen,0:r_cen],'o-',color='r',label='x=y=0, z>0',lw=1.5)
		plt.plot(rr_pc[:,r_cen,r_cen],density[:,r_cen,r_cen],'d-',color='g',label='z=y=0',lw=1.0)
		plt.plot(rr_pc[r_cen,:,r_cen],density[r_cen,:,r_cen],'.--',color='gold',label='x=z=0',lw=0.75)
		plt.axvline(r_cont_src_pc,color='k',linestyle='--',label='Continuum Radius',zorder=1)
		plt.axhline(n_cen,color='gray',linestyle='-',label='n$_{\mathrm{max}}$',zorder=1)
		plt.legend(ncol=2,fontsize=8)
		plt.xlabel('Radius (pc)')
		plt.ylabel('H$_2$ Density (cm$^{-3}$)')
		plt.xlim(left=0.,right=2*r_cont_src_pc)
		plt.ylim(bottom=0.)
		plt.gca().ticklabel_format(axis='y',style='sci',scilimits=(0,0))
		plt.minorticks_on()
		plt.savefig(path+'Model_DensityProfile/ModelDensity_theta'+str(theta)+'_inc'+str(inc)+'_phi'+str(phi)+'.pdf')
		plt.close()

		return density

	def construct_density_hotcomp(r_cont_src_pc,n_cen,rr,rr_pc,box_sz):
		#make the 3D density profile of the hot central component

		#H2 density goes like 1/r(pc)^2 
		n = 1/rr_pc**2
		n[np.isinf(n)==True]=np.nan
		n = n/np.nanmax(n)
		density = n*n_cen #cm^-3
		density[rr_pc > r_cont_src_pc]=np.nan

		#check if path exists
		if os.path.isdir(path+'Model_DensityProfile/')==False:
			print('Directory "'+path+'Model_DensityProfile/" does not exist. I will create it.')
			os.system('mkdir '+path+'Model_DensityProfile/')

		plt.figure(3)
		plt.clf()
		plt.plot(rr_pc[r_cen,r_cen,r_cen:],density[r_cen,r_cen,r_cen:],'s-',color='b',label='x=y=0, z<0',lw=2.0)
		plt.plot(rr_pc[r_cen,r_cen,0:r_cen],density[r_cen,r_cen,0:r_cen],'o-',color='r',label='x=y=0, z>0',lw=1.5)
		plt.plot(rr_pc[:,r_cen,r_cen],density[:,r_cen,r_cen],'d-',color='g',label='z=y=0',lw=1.0)
		plt.plot(rr_pc[r_cen,:,r_cen],density[r_cen,:,r_cen],'.--',color='gold',label='x=z=0',lw=0.75)
		plt.axvline(r_cont_src_pc,color='k',linestyle='--',label='Continuum Radius',zorder=1)
		plt.axhline(n_cen,color='gray',linestyle='-',label='n$_{\mathrm{max}}$',zorder=1)
		plt.legend(ncol=2,fontsize=8)
		plt.xlabel('Radius (pc)')
		plt.ylabel('H$_2$ Density (cm$^{-3}$)')
		plt.xlim(left=0.,right=2*r_cont_src_pc)
		plt.ylim(bottom=0.)
		plt.minorticks_on()
		plt.gca().ticklabel_format(axis='y',style='sci',scilimits=(0,0))
		plt.savefig(path+'Model_DensityProfile/ModelDensity_Hot_theta'+str(theta)+'_inc'+str(inc)+'_phi'+str(phi)+'.pdf')
		plt.close()

		return density


	def mask_outflow(r,theta,box,source_sz_pix):
		#make the 3D outflow mask
		cone_radius_z = np.abs(r)*np.tan(np.radians(theta/2))
		xx,yy=np.meshgrid(r,r)
		rr_slice = np.sqrt(xx**2+yy**2)
		outflow_mask = box.copy()
		if theta < 180:
			for i in range(outflow_mask.shape[2]):
				mask_slice = outflow_mask[:,:,i].copy()
				mask_slice[rr_slice <= cone_radius_z[i]]=1.0
				outflow_mask[:,:,i]=mask_slice.copy()
		else:
			outflow_mask = box.copy()+1.0
		return outflow_mask

	def mask_outflow_spec(r,theta,box_spec,source_sz_pix):
		#make the 4D outflow mask
		cone_radius_z = np.abs(r)*np.tan(np.radians(theta/2))
		xx,yy=np.meshgrid(r,r)
		rr_slice = np.sqrt(xx**2+yy**2)
		outflow_mask = box_spec.copy()
		if theta < 180:
			for i in range(outflow_mask.shape[2]):
				mask_slice = outflow_mask[:,:,i,:].copy()
				mask_slice[rr_slice <= cone_radius_z[i]]=1.0
				outflow_mask[:,:,i,:]=mask_slice.copy()
		else:
			outflow_mask = box_spec.copy()+1.0
		return outflow_mask

	def project_outflow(outflow_mask,outflow_mask_spec,inc,phi):
		#project the outflow mask to the specified orientation
		outflow_mask_inc = rotate(outflow_mask,inc,axes=(2,0),order=5,reshape=False)
		outflow_mask_proj = rotate(outflow_mask_inc,phi,axes=(1,2),order=5,reshape=False)
		outflow_mask_proj[outflow_mask_proj<0.9]=0.0 #clean up
		outflow_mask_spec_inc = rotate(outflow_mask_spec,inc,axes=(2,0),order=5,reshape=False)
		outflow_mask_spec_proj = rotate(outflow_mask_spec_inc,phi,axes=(1,2),order=5,reshape=False)
		outflow_mask_spec_proj[outflow_mask_spec_proj<0.9]=0.0 #clean up
		return outflow_mask_proj,outflow_mask_spec_proj

	def mask_cubes(data,outflow_mask_proj):
		#apply the outflow mask to the simulated cubes
		data_mask = data.copy()
		data_mask[outflow_mask_proj==0.0]=np.nan
		data_mask[data_mask == 0.0]=np.nan
		vmax_data_mask = np.nanmax(data_mask[np.isinf(data_mask)==False])
		vmin_data_mask = np.nanmin(data_mask[np.isinf(data_mask)==False])
		return data_mask, vmax_data_mask, vmin_data_mask

	def mask_outside_cont_src(data,r,r_cont_src):
		#mask out data outside of spherical continuum source
		xx, yy, zz = np.meshgrid(r, r, r)
		rxyz = np.sqrt(xx**2+yy**2+zz**2)
		beam_mask = np.ones(data.shape)
		beam_mask[rxyz>=r_cont_src] = 0.0
		data_masked = data*beam_mask
		return data_masked

	def mask_inside_cont_src(data,r,r_cont_src):
		#mask out data inside of spherical continuum source
		xx, yy, zz = np.meshgrid(r, r, r)
		rxyz = np.sqrt(xx**2+yy**2+zz**2)
		beam_mask = np.ones(data.shape)
		beam_mask[rxyz<r_cont_src] = 0.0
		data_masked = data*beam_mask
		return data_masked

	def add_const_Tcont(data,r,r_cont_src,T_cont):
		#add a constant continuum temperature component
		xx, yy, zz = np.meshgrid(r, r, r)
		rxyz = np.sqrt(xx**2+yy**2+zz**2)
		beam_mask = np.ones(data.shape)
		beam_mask[rxyz<=r_cont_src] = T_cont
		data_masked = data*beam_mask
		return data_masked

	def mask_fov(data,r,r_mask):
		#mask the final cube to the observed fov (a cylinder along z)
		xx, yy = np.meshgrid(r, r)
		rxy = np.sqrt(xx**2+yy**2)
		beam_mask = np.ones(data.shape)
		beam_mask[rxy>=r_mask] = 0.0
		data_masked = data*beam_mask
		return data_masked

	def mask_outside_fov(data,r,r_mask):
		#mask out regions inside the fov
		xx, yy = np.meshgrid(r, r)
		rxy = np.sqrt(xx**2+yy**2)
		beam_mask = np.ones(data.shape)
		beam_mask[rxy<r_mask] = 0.0
		data_masked = data*beam_mask
		return data_masked

	def add_smooth_cont_source(box_spec,rr,T_max,tau_max,source_sz_pix):
		#add a smooth continuum source at the center
		#T = T_cont
		#tau decays like a Gaussian spatially and are constant in frequency
		sx = box_spec.shape[0]
		sy = box_spec.shape[1]
		sz = box_spec.shape[2]
		snu = box_spec.shape[3]
		#get size parameters
		FWHM = source_sz_pix
		sigma = FWHM/2.355
		source_radius = FWHM/2.
		tau_cont = tau_max*np.exp(-rr**2/(2*(sigma)**2))
		tau_cont[rr>=source_sz_pix/2]=0.0 #maks outside continuum source
		tau_cont_ave = tau_cont.copy()
		tau_cont_ave[rr>=source_sz_pix/2]=np.nan
		tau_cont_ave = np.nanmean(tau_cont_ave)
		#want the change in tau_cont with z (since we're going to integrate through the box later)
		dtau_cont = np.zeros(tau_cont.shape)
		for i in range(tau_cont.shape[2]-2,-1,-1):
			dtau_cont[:,:,i]=np.abs(tau_cont[:,:,i+1]-tau_cont[:,:,i])
		if printdiagnostics==True:
			print('Average dtau_cont = %.2e' %(np.average(dtau_cont)))
		tau_cont_nu=tau_cont.reshape((sx,sy,sz,1))*np.ones((1,snu))
		dtau_cont_nu=dtau_cont.reshape((sx,sy,sz,1))*np.ones((1,snu))
		T_cont_nu = T_max*np.ones((sx,sy,sz,snu))
		return T_cont_nu,tau_cont_nu

	def line_of_sight_velocity(velocity_proj,cosPsi):
		#get line-of-sight velocity component
		velocity_los = velocity_proj*cosPsi
		velocity_los[np.isnan(velocity_los)==1]=0.0
		vlos = velocity_los.copy()
		return vlos

	def temperature_ex_nu(Tex,v_los,sigma_v,v_range,nu,c):
		#make a 4D cube of Tex
		Tex_nu = Tex*np.ones((v_los.shape[0],v_los.shape[1],v_los.shape[2],v_range.shape[0]))
		return Tex_nu

	def abs_coeff_nu(line,v_range,sigma_v,Tgas,density,v_los,r_cen):
		#calculate the 4D absorption coefficient
		#for the gas component
		#load line data for this molecular line
		Tex=Tgas
		line_str = line.split('(')[0]
		J_u = int(line.split('(')[1].split('-')[0])
		J_l = int(line.split('-')[1].split(')')[0])
		mol_dat=path+'MolecularLineData/'+line_str+'_data.csv'
		if os.path.isdir(path+'MolecularLineData/')==False:
			print('Directory with molecular line "'path+'MolecularLineData/'+line_str+'_data.csv" data not found! I will create it, but other paths may break.')
			os.system('mkdir '+path+'MolecularLineData/')
		if os.path.exists(path+'MolecularLineData/'+line_str+'_data.csv')==False:
			print('Molecular line data file not found "'path+'MolecularLineData/'+line_str+'_data.csv". Exiting...')
			sys.exit()
		this_line_data = pd.read_csv(mol_dat,comment='#')
		Ti = this_line_data['T'].values #K, Eu/k
		J = this_line_data['QN'].values
		gJ = this_line_data['g'].values
		Z = np.sum(gJ*np.exp(-Ti/Tex))
		gl=gJ[np.where(J==J_l)[0][0]].astype('float')
		gu=gJ[np.where(J==J_u)[0][0]].astype('float')
		Tl = Ti[np.where(J==J_l)[0][0]].astype('float') #K, Eu/k
		#load data for this specific transition
		if os.path.exists(path+'Band7_HighRes_LineID.csv')==False:
			print('Line list file not found "'path+'Band7_HighRes_LineID.csv". Exiting...')
			sys.exit()
		transition_data = pd.read_csv(path+'Band7_HighRes_LineID.csv')
		idx = transition_data['Line'].values.tolist().index(line)
		frac_abun = transition_data['abun_frac_H2'][idx]
		nu0 = transition_data['Freq'][idx]*1E9 #Hz
		Aul = transition_data['Aul'][idx] #Einstein A, s^-1
		c = 2.9979E10 #cm/s
		h = 6.6260755E-27 #erg s
		k = 1.380658E-16 #erg/K
		#intrinsic line shape, Draine 2001 Eq. 6.38
		b = (sigma_v*1E5)/(2*np.sqrt(np.log(2))) #cm/s
		#shift the velocity range based on v_los
		v_los_res = np.reshape(v_los,(v_los.shape[0],v_los.shape[1],v_los.shape[2],1))*np.ones((1,1,1,v_range.shape[0]))
		v_range_res = np.ones((v_los.shape[0],v_los.shape[1],v_los.shape[2],1))*np.reshape(v_range,(1,1,1,v_range.shape[0]))
		v_shift = (v_range_res-v_los_res)*1E5 #cm/s
		nu_shift = nu0*(1-v_los_res*1E5/c) #Hz
		phi_nu = 1/np.sqrt(np.pi)*1/nu_shift*c/b*np.exp(-(v_shift/b)**2) #s
		#abs cross section, Draine 2001 Eq. 6.18
		sigma_lu = gu/gl*c**2/(8*np.pi*nu_shift**2)*Aul*phi_nu #cm^2
		#get n_l
		nl = frac_abun*gl/Z*np.exp(-Tl/Tex)*density
		#reshape
		nl = np.reshape(nl,(nl.shape[0],nl.shape[1],nl.shape[2],1))*np.ones((1,1,1,v_range.shape[0]))
		#get absorption profile(x,y,z,nu)
		kappa_nu = nl*sigma_lu*(1-np.exp(-h*nu_shift/(k*Tgas))) #cm^-1
		return kappa_nu	

	def calc_dtau_nu(kappa_nu_proj,spaxel_sz_pc):
		#Calculate delta tau
		#dtau = kappa(s)ds
		dtau_nu = kappa_nu_proj*spaxel_sz_pc*(3.086E18)
		return dtau_nu

	def calc_T_nu(dtau_nu,Tex_nu,dtau_cont_nu,T_cont_nu):
		#Do the radiative transfer to get dT(nu) and T(nu) for the cold outflowing component only
		T_nu = np.zeros(dtau_nu.shape)
		dT_nu = np.zeros(dtau_nu.shape)
		for i in range(T_nu.shape[2]-2,-1,-1):
			T_nu[:,:,i,:] = T_nu[:,:,i+1,:]*np.exp(-(dtau_nu[:,:,i,:]+dtau_cont_nu[:,:,i,:])) + (Tex_nu[:,:,i,:]+T_cont_nu[:,:,i,:])*(1-np.exp(-(dtau_nu[:,:,i,:]+dtau_cont_nu[:,:,i,:])))
			dT_nu[:,:,i,:] = T_nu[:,:,i,:]-T_nu[:,:,i+1,:]
		return dT_nu,T_nu

	def calc_T_nu_hot(dtau_nu,Tex_nu,dtau_cont_nu,T_cont_nu,dtau_hot_nu,Tex_hot_nu):
		#Do the radiative transfer to get dT(nu) and T(nu) for the cold outflowing & hot components
		#This assumes that the components do not interact
		T_nu = np.zeros(dtau_nu.shape)
		dT_nu = np.zeros(dtau_nu.shape)
		for i in range(T_nu.shape[2]-2,-1,-1):
			dT_c = T_cont_nu[:,:,i,:]*(1-np.exp(-dtau_cont_nu[:,:,i,:]))
			dT_g = Tex_nu[:,:,i,:]*(1-np.exp(-dtau_nu[:,:,i,:]))
			dT_h = Tex_hot_nu[:,:,i,:]*(1-np.exp(-dtau_hot_nu[:,:,i,:]))
			T_nu[:,:,i,:] = T_nu[:,:,i+1,:]*np.exp(-(dtau_nu[:,:,i,:]+dtau_cont_nu[:,:,i,:]+dtau_hot_nu[:,:,i,:])) + dT_c+dT_g+dT_h
			dT_nu[:,:,i,:] = T_nu[:,:,i,:]-T_nu[:,:,i+1,:]
		return dT_nu,T_nu

	def calc_T_nu_hot_fast(dtau_nu,Tex_nu,dtau_cont_nu,T_cont_nu,dtau_hot_nu,Tex_hot_nu,dtau_fast_nu,Tex_fast_nu):
		#Do the radiative transfer to get dT(nu) and T(nu) for the cold (slow & fast) outflowing & hot components
		#This assumes that the components do not interact
		T_nu = np.zeros(dtau_nu.shape)
		dT_nu = np.zeros(dtau_nu.shape)
		for i in range(T_nu.shape[2]-2,-1,-1):
			dT_c = T_cont_nu[:,:,i,:]*(1-np.exp(-dtau_cont_nu[:,:,i,:]))
			dT_g = Tex_nu[:,:,i,:]*(1-np.exp(-dtau_nu[:,:,i,:]))
			dT_h = Tex_hot_nu[:,:,i,:]*(1-np.exp(-dtau_hot_nu[:,:,i,:]))
			dT_f = Tex_fast_nu[:,:,i,:]*(1-np.exp(-dtau_fast_nu[:,:,i,:]))
			T_nu[:,:,i,:] = T_nu[:,:,i+1,:]*np.exp(-(dtau_nu[:,:,i,:]+dtau_cont_nu[:,:,i,:]+dtau_hot_nu[:,:,i,:]+dtau_fast_nu[:,:,i,:])) + dT_c+dT_g+dT_h+dT_f
			dT_nu[:,:,i,:] = T_nu[:,:,i,:]-T_nu[:,:,i+1,:]
		return dT_nu,T_nu


	def plt3d(thisax,x,y,z,data,r_cen,cmap,norm,label,extend,theta,inc,phi,saveon):
		#plot 3D scatter plot for visualization
		#THIS IS SLOW
		c=data.copy()
		#cut away portion to see to center
		cut = 0
		c[r_cen+cut:,r_cen+cut:,:]=np.nan
		c=np.ravel(c)
		ax=thisax
		im=ax.scatter(x,y,z,c=c,s=4,marker='s',depthshade=False,cmap=cmap,norm=norm,rasterized=True,antialiaseds=True)
		
		#add boundaries to guide the eye
		lw=0.5
		col='darkgray'
		ax.plot([cut,cut],[cut,r_cen],[r_cen,r_cen],'--',color=col,linewidth=lw,zorder=10)
		ax.plot([cut,r_cen],[cut,cut],[r_cen,r_cen],'--',color=col,linewidth=lw,zorder=10)
		ax.plot([cut,cut],[cut,r_cen],[-r_cen,-r_cen],'--',color=col,linewidth=lw,zorder=10)
		ax.plot([cut,r_cen],[cut,cut],[-r_cen,-r_cen],'--',color=col,linewidth=lw,zorder=10)
		ax.plot([cut,-r_cen],[r_cen,r_cen],[-r_cen,-r_cen],'--',color=col,linewidth=lw,zorder=10)
		ax.plot([r_cen,r_cen],[cut,-r_cen],[-r_cen,-r_cen],'--',color=col,linewidth=lw,zorder=10)
		ax.plot([cut,cut],[cut,cut],[-r_cen,r_cen],'--',color=col,linewidth=lw,zorder=10)
		ax.plot([r_cen,r_cen],[cut,cut],[-r_cen,r_cen],'--',color=col,linewidth=lw,zorder=10)
		ax.plot([cut,cut],[r_cen,r_cen],[-r_cen,r_cen],'--',color=col,linewidth=lw,zorder=10)
		ax.plot([cut,-r_cen],[r_cen,r_cen],[r_cen,r_cen],'--',color=col,linewidth=lw,zorder=10)
		ax.plot([r_cen,r_cen],[cut,-r_cen],[r_cen,r_cen],'--',color=col,linewidth=lw,zorder=10)
		ax.plot([-r_cen,-r_cen],[r_cen,r_cen],[-r_cen,r_cen],'--',color=col,linewidth=lw,zorder=10)
		ax.plot([r_cen,r_cen],[-r_cen,-r_cen],[-r_cen,r_cen],'--',color=col,linewidth=lw,zorder=10)
		ax.plot([-r_cen,-r_cen],[-r_cen,r_cen],[-r_cen,-r_cen],'--',color=col,linewidth=lw,zorder=10)
		ax.plot([-r_cen,r_cen],[-r_cen,-r_cen],[-r_cen,-r_cen],'--',color=col,linewidth=lw,zorder=10)

		#label plot, set axes, set viewing angle
		ax.set_xlabel('x')
		ax.set_ylabel('y')
		ax.set_zlabel('z')
		ax.set_xlim(-r_cen,r_cen)
		ax.set_ylim(-r_cen,r_cen)
		ax.set_zlim(-r_cen,r_cen)
		#ax.view_init(elev=270-75,azim=180+45)
		ax.view_init(elev=180+2,azim=180+45)
		ax.tick_params(labelsize=7)
		#ax.grid(False)

		#add colorbar
		#fix ugly SymLogNorm tick labels
		if 'SymLogNorm' in str(norm):
			sln=norm
			ct_n=-np.geomspace(-sln.vmin,sln.linthresh,
							   num=int(np.log10(np.abs(sln.vmin/sln.linthresh)))+1)
			ct_p=np.geomspace(sln.linthresh,sln.vmax,
							  num=int(np.log10(np.abs(sln.vmax/sln.linthresh)))+1)
			ct = np.concatenate([ct_n,[0.0],ct_p])
			ct_sign = np.sign(ct)
			ct_log = np.log10(np.abs(ct))
			cb = plt.colorbar(im,ticks=ct,shrink=0.6,pad=0.15,extend=extend)
			ctl = np.empty(ct.shape,dtype='U10')
			for i in range(len(ctl)):
				if np.isinf(ct_log[i])==False:
					ctl[i] = r'$%d^{%d}$' %(ct_sign[i]*10,ct_log[i])
				else:
					ctl[i] = '0'
			cb.ax.set_yticklabels(ctl)
		else:
			cb=plt.colorbar(im,shrink=0.6,pad=0.1,extend=extend)
		cb.set_label(label)
			
		#save individual plot
		if saveon==1:
			if os.path.isdir(path+'Model_'+label+'3D')==False:
				os.system('mkdir '+path+'Model_'+label+'3D')

			extent = ax.get_window_extent().transformed(fig.den_maski_scale_trans.inverted())
			plt.savefig(path+'Model_'+label+'3D/Model'+label+'3D_theta'+str(theta)+'_inc'+str(inc)+'_phi'+str(phi)+'.pdf',den_maski=300,bbox_inches=extent.expanded(1.2,1.2))

	def plt_spec(v_spec,T_spec,theta,inc,phi,saveon,suffix):
		plt.figure(2)
		plt.clf()
		plt.plot(v_spec,T_spec,'k.-')
		ax=plt.gca()
		ax.set_xlabel('V (km s$^{-1}$)')
		ax.set_ylabel('T$_{\mathrm{B}}$ (K)')
		ax.minorticks_on()
		ax2 = ax.twiny()
		ax2.set_xlim(ax.get_xlim())
		ax2.set_xlabel(r'$\nu_{\mathrm{rest}}$ (GHz)')
		xtl = ax.get_xticks()
		xt_min = nu*(1-ax.get_xlim()[0]/c)
		xt_max = nu*(1-ax.get_xlim()[1]/c)
		xt_top = np.arange(np.round(xt_min,1),np.round(xt_max,1)-0.1,-0.1) 
		xtl_top = ['%.1f' %x for x in xt_top]
		xt_top_pos = c*(1-xt_top/nu)
		if xt_top_pos[0] < xtl[0]:
			xt_top_pos = xt_top_pos[1:]
			xtl_top = xtl_top[1:]
		if xt_top_pos[-1] > xtl[-1]:
			xt_top_pos = xt_top_pos[0:-1]
			xtl_top = xtl_top[0:-1]
		ax2.set_xticks(xt_top_pos)
		ax2.set_xticklabels(xtl_top)
		ax2.minorticks_on()
		if saveon==1:
			if os.path.isdir(path+'Model_Spec/')==False:
				os.system('mkdir '+path+'Model_Spec/')
			plt.savefig(path+'Model_Spec/Model_Spec_theta'+str(theta)+'_inc'+str(inc)+'_phi'+str(phi)+suffix+'.pdf',dpi=300,bbox_inches='tight')
		plt.close()

	def plot_datacube(temp_nu,r_cen,T_cont,v_range,theta,inc,phi,suffix):
		#warning: this is slow and may be buggy
		if os.path.isdir(path+'Model_3Dcube/')==False:
			os.system('mkdir '+path+'Model_3Dcube/')

		fig = plt.figure(1)
		plt.clf()
		c=temp_nu.copy()
		c=mask_outside_cont_src(c,r,r_cen)
		c[r_cen+1:,r_cen+1:,:]=np.nan
		c=np.ravel(c)
		ax=fig.add_subplot(111,projection='3d')
		cmap='Spectral_r'
		norm=SymLogNorm(vmin=np.nanmin(c),vmax=T_cont,linthresh=0.1)
		im=ax.scatter(np.ravel(rxnu),np.ravel(rynu),np.ravel(rnu),c=c,s=4,marker='s',depthshade=False,cmap=cmap,norm=norm,rasterized=True)
		lw=0.5
		ax.plot([0,0],[0,r_cen],[np.max(v_range),np.max(v_range)],'--',color='gray',linewidth=lw,zorder=10)
		ax.plot([0,r_cen],[0,0],[np.max(v_range),np.max(v_range)],'--',color='gray',linewidth=lw,zorder=10)
		ax.plot([0,0],[0,r_cen],[-np.max(v_range),-np.max(v_range)],'--',color='gray',linewidth=lw,zorder=10)
		ax.plot([0,r_cen],[0,0],[-np.max(v_range),-np.max(v_range)],'--',color='gray',linewidth=lw,zorder=10)
		ax.plot([0,-r_cen],[r_cen,r_cen],[-np.max(v_range),-np.max(v_range)],'--',color='gray',linewidth=lw,zorder=10)
		ax.plot([r_cen,r_cen],[0,-r_cen],[-np.max(v_range),-np.max(v_range)],'--',color='gray',linewidth=lw,zorder=10)
		ax.plot([0,0],[0,0],[-np.max(v_range),np.max(v_range)],'--',color='gray',linewidth=lw,zorder=10)
		ax.plot([r_cen,r_cen],[0,0],[-np.max(v_range),np.max(v_range)],'--',color='gray',linewidth=lw,zorder=10)
		ax.plot([0,0],[r_cen,r_cen],[-np.max(v_range),np.max(v_range)],'--',color='gray',linewidth=lw,zorder=10)
		ax.plot([0,-r_cen],[r_cen,r_cen],[np.max(v_range),np.max(v_range)],'--',color='gray',linewidth=lw,zorder=10)
		ax.plot([r_cen,r_cen],[0,-r_cen],[np.max(v_range),np.max(v_range)],'--',color='gray',linewidth=lw,zorder=10)
		ax.plot([-r_cen,-r_cen],[r_cen,r_cen],[-np.max(v_range),np.max(v_range)],'--',color='gray',linewidth=lw,zorder=10)
		ax.plot([r_cen,r_cen],[-r_cen,-r_cen],[-np.max(v_range),np.max(v_range)],'--',color='gray',linewidth=lw,zorder=10)
		ax.plot([-r_cen,-r_cen],[-r_cen,r_cen],[-np.max(v_range),-np.max(v_range)],'--',color='gray',linewidth=lw,zorder=10)
		ax.plot([-r_cen,r_cen],[-r_cen,-r_cen],[-np.max(v_range),-np.max(v_range)],'--',color='gray',linewidth=lw,zorder=10)
		ax.set_xlabel('x')
		ax.set_ylabel('y')
		ax.set_zlabel('V (km/s)')
		ax.set_xlim([-r_cen,r_cen])
		ax.set_ylim([-r_cen,r_cen])
		ax.set_zlim([-np.max(v_range),np.max(v_range)])
		ax.view_init(elev=270-75,azim=180+45)
		ax.tick_params(labelsize=7)
		cb=plt.colorbar(im,shrink=0.6,pad=0.15,extend='both')
		cb.set_label('T$_{\mathrm{B}}$ (K)')
		plt.savefig(path+'Model_3Dcube/Model_datacube_theta'+str(theta)+'_inc'+str(inc)+'_phi'+str(phi)+suffix+'.pdf',dpi=300,bbox_inches='tight')
		plt.close()

	def plot_radial_profile(r_cen,rr_pc,Tex_nu,Tex_hot_nu,Tex_fast_nu,T_cont_nu,density,density_hot,density_fast,dtau_nu,dtau_hot_nu,dtau_fast_nu,tau_cont_nu,ssc,line,theta,inc,phi,suffix):
		r_1d = rr_pc[r_cen,r_cen,r_cen:]
		Tex_1d = np.nanmax(Tex_nu[r_cen,r_cen,r_cen:,:],axis=1)
		Tex_hot_1d = np.nanmax(Tex_hot_nu[r_cen,r_cen,r_cen:,:],axis=1)
		Tex_fast_1d = np.nanmax(Tex_fast_nu[r_cen,r_cen,r_cen:,:],axis=1)
		Tcont_1d = np.nanmax(T_cont_nu[r_cen,r_cen,r_cen:,:],axis=1)
		density_1d = density[r_cen,r_cen,r_cen:]
		density_hot_1d = density_hot[r_cen,r_cen,r_cen:]
		density_fast_1d = density_fast[r_cen,r_cen,r_cen:]
		tau_1d = np.nanmax(dtau_nu[r_cen,r_cen,r_cen:,:],axis=1)
		tau_hot_1d = np.nanmax(dtau_hot_nu[r_cen,r_cen,r_cen:,:],axis=1)
		tau_fast_1d = np.nanmax(dtau_fast_nu[r_cen,r_cen,r_cen:,:],axis=1)
		tau_cont_1d = np.nanmax(tau_cont_nu[r_cen,r_cen,r_cen:,:],axis=1)

		if os.path.isdir(path+'Model_Params/Plots/')==False:
			os.system('mkdir '+path+'Model_Params/Plots/')

		fig,ax=plt.subplots(nrows=1,ncols=3,num=2,figsize=(15,3))
		ax[0].axvline(r_cont_src_pc,color='k',linestyle='--')
		ax[0].step(r_1d,density_1d,'b-',label='Cold Gas')
		ax[0].step(r_1d,density_hot_1d,'-',color='gold',label='Hot Gas')
		ax[0].step(r_1d,density_fast_1d,'-',color='rebeccapurple',label='Cold, Fast Gas')
		ax[0].set_ylabel('H$_2$ Density (cm$^{-3}$)')
		ax[0].set_xlabel('Radius (pc)')
		ax[0].set_yscale('log')
		ax[0].set_xlim(left=0)
		ax[0].minorticks_on()
		ax[1].axvline(r_cont_src_pc,color='k',linestyle='--')
		ax[1].step(r_1d,tau_1d,'b-',label='Cold Gas')
		ax[1].step(r_1d,tau_hot_1d,'-',color='gold',label='Hot Gas')
		ax[1].step(r_1d,tau_fast_1d,'-',color='rebeccapurple',label='Cold, Fast Gas')
		ax[1].step(r_1d,tau_cont_1d,'g-',label='Continuum')
		ax[1].set_ylabel('$\\tau$')
		ax[1].set_xlabel('Radius (pc)')
		ax[1].set_yscale('log')
		ax[1].set_xlim(left=0)
		ax[1].minorticks_on()
		ax[2].axvline(r_cont_src_pc,color='k',linestyle='--',label='Continuum\n Source Radius')
		ax[2].step(r_1d,Tex_1d,'b-',label='Cold Gas')
		ax[2].step(r_1d,Tex_hot_1d,'-',color='gold',label='Hot Gas')
		ax[2].step(r_1d,Tex_fast_1d,'-',color='rebeccapurple',label='Cold, Fast Gas')
		ax[2].step(r_1d,Tcont_1d,'g-',label='Continuum')
		ax[2].set_ylabel('Temperature (K)')
		ax[2].set_xlabel('Radius (pc)')
		ax[2].legend(loc='center left',bbox_to_anchor=(1.1,0.5),ncol=1,fontsize=9)
		ax[2].set_ylim(bottom=0)
		ax[2].set_xlim(left=0)
		ax[2].minorticks_on()
		plt.tight_layout()
		plt.savefig(path+'Model_Params/Plots/ModelParams_SSC'+ssc+'_'+line+'_theta'+str(theta)+'_inc'+str(inc)+'_phi'+str(phi)+suffix+'.pdf')
		plt.close()

	def save_model_cube(T_out,bmaj,bmin,spaxel_sz_arcsec,chan_width,v_range,nu,theta,inc,phi,suffix,line,ssc):
		#save the final T(x,y,nu) model as a fits cube
		#can then do moments, channel maps, etc
		#warning: this may be buggy
		from astropy.io import fits
		import getpass

		hdu = fits.PrimaryHDU(T_out.T)

		#add other header keywords
		hdu.header['BMAJ'] = bmaj/3600 #deg
		hdu.header['BMIN'] = bmin/3600 #deg
		hdu.header['BPA'] = 0.0 #deg, nominal
		hdu.header['BTYPE'] = 'Intensity'
		hdu.header['BUNIT'] = 'K'	
		hdu.header['CRVAL1'] = '0'
		hdu.header['CTYPE1'] = 'RA---CAR'
		hdu.header['CRVAL2'] = '0'
		hdu.header['CTYPE2'] = 'DEC--CAR'
		hdu.header['CDELT1'] = spaxel_sz_arcsec/3600 #deg
		hdu.header.comments['CDELT1'] = 'degrees'
		hdu.header['CDELT2'] = spaxel_sz_arcsec/3600 #deg
		hdu.header.comments['CDELT2'] = 'degrees'		
		hdu.header['CUNIT1'] = 'deg'
		hdu.header['CUNIT2'] = 'deg'	
		#hdu.header['CRPIX1'] = '0'
		#hdu.header['CRPIX2'] = '0'
		hdu.header['CTYPE3'] = 'VRAD'
		hdu.header['CRVAL3'] = np.min(v_range)
		hdu.header['CDELT3'] = chan_width
		hdu.header['CRPIX3'] = '0'
		hdu.header['CUNIT3'] = 'km/s'
		hdu.header['RESTFRQ'] = nu*1E9
		hdu.header.comments['RESTFRQ'] = 'Rest Frequency (Hz)'
		hdu.header['SPECSYS'] = 'LSRK'
		hdu.header['OBSERVER'] = getpass.getuser()
		hdu.header['DATE'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
		hdu.header.comments['DATE'] = 'Timestamp of model'
		hdu.header['ORIGIN'] = 'ModelOutflowPCygni.py'

		hdul = fits.HDUList([hdu])
		if os.path.isdir(path+'Model_Cubes/')==False:
			os.system('mkdir '+path+'Model_Cubes/')
		fname=path+'Model_Cubes/ModelCube_SSC'+ssc+'_'+line+'_theta'+str(theta)+'_inc'+str(inc)+'_phi'+str(phi)+suffix+'.fits'
		hdul.writeto(fname,overwrite=True)

	###############################
	##### START MAIN FUNCTION #####
	###############################

	suffix = '_'+cont_type

	#define constants
	c = 2.99792458E5 #km/s
	h = 6.6260755E-27 #erg s
	k = 1.380658E-16 #erg/K
	dist = 3.5 #Mpc, NGC253
	mH2 = 2*1.6733E-24 #g
	M_sun = 2E33 #g

	#open molecular line data
	line_data = pd.read_csv(path+'Band7_HighRes_LineID.csv')
	idx = line_data['Line'].values.tolist().index(line)
	nu = line_data['Freq'][idx] #GHz, line rest frequency


	print('Matching '+line+' in SSC '+ssc+':')
	print('Running model with outflow parameters:')
	print('\ttheta = %1.f deg, inc = %1.f deg, phi = %1.f deg' %(theta,inc,phi))
	if printdiagnostics==True:
	#print the input parameters
		print('\tV_outflow = '+str(v_outflow)+' km/s')
		print('\tsigma_v,FWHM = '+str(v_outflow_FWHM)+' km/s')

		print('Running model with gas parameters:')
		print('\tLine = '+line)
		print('\tT_ex = '+str(T_ex)+' K')
		print('\tn_cen = %.1E cm^-3' %n_cen)

		print('Running model with continuum parameters:')
		print('\tcont model = '+cont_type)
		print('\tT_cont = %1.f K' %T_cont)


	#make a box, size is based on the source size
	box_sz = 2**6+1
	box = np.zeros((box_sz,box_sz,box_sz))
	source_sz_pc = source_sz_arcsec/206265.*dist*1E6 #pc
	box_oversamp_fac = 2.0 #how much larger than continuum source to make box
	box_sz_pc = box_oversamp_fac*source_sz_pc #make the box twice the size of the continuum source
	spaxel_sz_pc = box_sz_pc/box_sz #pc
	source_sz_pix = int(np.ceil(source_sz_pc/spaxel_sz_pc)) #pixels across continuum source
	if np.mod(source_sz_pix,2)==0: #make odd number
		source_sz_pix = source_sz_pix+1
	r_cont_src = (source_sz_pix-1)/2 #radius of the continuum source
	r_cont_src_pc = (source_sz_pc-spaxel_sz_pc)/2

	#make the spectral_box (x,y,z,nu)
	nchan = 2**7+1
	pad = 4.
	vel_rng = np.sqrt(v_outflow**2+v_outflow_FWHM**2)
	v_range = np.linspace(-pad*vel_rng,pad*vel_rng,num=nchan) #km/s, channels
	chan_width = 2*pad*vel_rng/(nchan-1)
	channels_nu = nu*(1-v_range/c) #GHz
	channels_dnu = -nu*v_range/c #GHz
	box_spec = np.zeros((box_sz,box_sz,box_sz,nchan))

	if printdiagnostics==True:
		print('Creating a '+str(box_sz)+' x '+str(box_sz)+' x '+str(box_sz)+' x '+str(nchan)+' box.')
		print('\tThe spatial box is '+str(np.round(box_sz_pc,2))+' pc on a side.')
		print('\tEach spaxel is '+str(np.round(spaxel_sz_pc,3))+' pc on a side.')
		print('\tThe channels are '+str(np.round(chan_width,2))+' km/s wide.')

	###################################
	##### IN THE FRAME OF THE SSC #####
	###################################

	#get radial coordinates
	r_cen = int((box_sz-1)/2) #pixel coordinate of r=0
	r=np.linspace(-r_cen,r_cen,num=box_sz)

	rx, ry, rz = np.meshgrid(r, r, r)
	r_cont_shift=np.abs(r)-r_cont_src
	r_cont_shift[r_cont_shift<0]=0.
	r_cont_shift[r < 0] = -r_cont_shift[r < 0]
	rx_cont_shift, ry_cont_shift, rz_cont_shift = np.meshgrid(r_cont_shift,r_cont_shift,r_cont_shift)
	rxnu, rynu, rnu = np.meshgrid(r,r,v_range)
	rr = np.sqrt(rx**2+ry**2+rz**2) #radial coordinates
	rr_cont_shift = np.sqrt(rx_cont_shift**2+ry_cont_shift**2+rz_cont_shift**2)
	r_pc = np.linspace(-box_sz_pc/2,box_sz_pc/2,num=box_sz) #pc
	rx_pc, ry_pc, rz_pc = np.meshgrid(r_pc, r_pc, r_pc)
	rr_pc = np.sqrt(rx_pc**2+ry_pc**2+rz_pc**2)

	#construct density profile
	density = construct_density(r_cont_src_pc,n_cen,rr,rr_pc,box_sz)
	density = mask_inside_cont_src(density,r,r_cont_src)

	density_fast = construct_density(r_cont_src_pc,n_cen_fast,rr,rr_pc,box_sz)
	density_fast = mask_inside_cont_src(density_fast,r,r_cont_src)

	density_hot = construct_density_hotcomp(r_cont_src_pc,n_cen_hot,rr,rr_pc,box_sz)
	density_hot = mask_outside_cont_src(density_hot,r,r_cont_src)

	#now set up the constant velocity field
	velocity = (box.copy()+1.)*v_outflow
	velocity = mask_inside_cont_src(velocity,r,r_cont_src)

	velocity_fast = (box.copy()+1.)*v_outflow_fast
	velocity_fast = mask_inside_cont_src(velocity_fast,r,r_cont_src)

	#set up the angle at each x,y,z
	#will be needed for l.o.s. velocity projection
	Phi = np.arctan(ry/rx)
	Theta = np.arccos(rz/rr)
	cosPsi = np.cos(Theta)


	#now carve out the outflow
	#i.e. set the density outsite the outflow to 0
	#also set the density inside the continuum source to 0
	outflow_mask = mask_outflow(r,theta,box,source_sz_pix)
	outflow_mask_spec = mask_outflow_spec(r,theta,box_spec,source_sz_pix)

	#######################################
	##### PROJECT TO OBSERVER'S FRAME #####
	#######################################

	#rotate outflow mask
	outflow_mask_proj,outflow_mask_spec_proj=project_outflow(outflow_mask,outflow_mask_spec,inc,phi)

	#mask velocity to projected outflow
	velocity_proj = velocity*outflow_mask_proj
	velocity_fast_proj = velocity_fast*outflow_mask_proj

	#get the total H2 mass in the box
	spaxel_sz_cm = spaxel_sz_pc*3.086E18 #cm
	spaxel_sz_arcsec = spaxel_sz_pc/(dist*1E6)*206265
	spaxel_volume = (spaxel_sz_cm)**3 #cm^3
	density[np.isnan(density)==True]=0.0
	density_fast[np.isnan(density_fast)==True]=0.0
	density_hot[np.isnan(density_hot)==True]=0.0

	#get velocity projected along the line of sight
	velocity_los = line_of_sight_velocity(velocity_proj,cosPsi)
	velocity_fast_los = line_of_sight_velocity(velocity_fast_proj,cosPsi)

	#get the velocity distribution of the ambient gas
	velocity_amb_hot = np.random.normal(loc=0.,scale=v_hot_FWHM/2.355,size=velocity_los.shape)
	velocity_amb_cold = np.random.normal(loc=0.,scale=v_outflow_FWHM/2.355,size=velocity_los.shape)
	velocity_amb_fast = np.random.normal(loc=0.,scale=v_outflow_fast_FWHM/2.355,size=velocity_fast_los.shape)


	#where not outflow, set the velocity to the ambient cold velocity
	velocity_los[outflow_mask_proj==0.]=velocity_amb_cold[outflow_mask_proj==0.]
	velocity_fast_los[outflow_mask_proj==0.]=velocity_amb_fast[outflow_mask_proj==0.]
	
	#inside continuum source (hot gas), set the velocity to the hot gas vel disp
	cont_mask = mask_outside_cont_src(box+1.,r,r_cont_src)
	velocity_los[cont_mask==1.]=velocity_amb_hot[cont_mask==1.]
	velocity_fast_los[cont_mask==1.]=velocity_amb_hot[cont_mask==1.]
	
	#get the mass in each component
	M_hot_gas = np.nansum(density_hot)*spaxel_volume*mH2/M_sun #Msun
	density_cold = mask_fov(density*outflow_mask_proj,r,r_cont_src)
	M_cold_gas = np.nansum(density_cold)*spaxel_volume*mH2/M_sun #Msun
	density_blue = density_cold.copy()
	density_blue[velocity_los >= 0]=np.nan
	M_cold_gas_blue = np.nansum(density_blue)*spaxel_volume*mH2/M_sun #Msun
	M_fast_gas = np.nansum(density_fast)*spaxel_volume*mH2/M_sun #Msun
	density_fast_blue = density_fast.copy()
	density_fast_blue[velocity_fast_los >= 0]=np.nan
	M_fast_gas_blue = np.nansum(density_fast)*spaxel_volume*mH2/M_sun #Msun
	
	if printdiagnostics==True:
		print('Check Masses in Components')
		print('\tH_2 mass in hot gas component = %.1E M_sun = 10^%.1f M_sun' %(M_hot_gas,np.log10(M_hot_gas)))
		print('\tH_2 mass in cold gas component = %.1E M_sun = 10^%.1f M_sun' %(M_cold_gas,np.log10(M_cold_gas)))
		print('\tH_2 mass in blueshifted cold gas component = %.1E M_sun = 10^%.1f M_sun' %(M_cold_gas_blue,np.log10(M_cold_gas_blue)))
		print('\tH_2 mass in fast cold gas component = %.1E M_sun = 10^%.1f M_sun' %(M_fast_gas,np.log10(M_fast_gas)))
		print('\tH_2 mass in blueshifted fast cold gas component = %.1E M_sun = 10^%.1f M_sun' %(M_fast_gas_blue,np.log10(M_fast_gas_blue)))
		

	#get gas excitation temperature as function of nu
	Tex_nu = temperature_ex_nu(T_ex,velocity_los,v_outflow_FWHM,v_range,nu,c)
	Tex_nu = mask_inside_cont_src(Tex_nu,r,r_cont_src)

	Tex_fast_nu = temperature_ex_nu(T_ex,velocity_fast_los,v_outflow_FWHM,v_range,nu,c)
	Tex_fast_nu = mask_inside_cont_src(Tex_fast_nu,r,r_cont_src)

	Tex_hot_nu = temperature_ex_nu(T_hot,velocity_los,v_outflow_FWHM,v_range,nu,c)
	Tex_hot_nu = mask_outside_cont_src(Tex_hot_nu,r,r_cont_src)

	#get kappa_nu
	kappa_nu = abs_coeff_nu(line,v_range,v_outflow_FWHM,T_ex,density,velocity_los,r_cen) #cm^-1
	kappa_fast_nu = abs_coeff_nu(line,v_range,v_outflow_fast_FWHM,T_ex,density_fast,velocity_fast_los,r_cen) #cm^-1
	kappa_hot_nu = abs_coeff_nu(line,v_range,v_outflow_FWHM,T_hot,density_hot,velocity_los,r_cen) #cm^-1
	
	kappa_nu_proj = kappa_nu*outflow_mask_spec_proj
	kappa_fast_nu_proj = kappa_fast_nu*outflow_mask_spec_proj
	kappa_hot_nu_proj = kappa_hot_nu.copy()

	#find local delta_tau at each cell
	dtau_nu = calc_dtau_nu(kappa_nu_proj,spaxel_sz_pc)
	dtau_fast_nu = calc_dtau_nu(kappa_fast_nu_proj,spaxel_sz_pc)
	dtau_hot_nu = calc_dtau_nu(kappa_hot_nu_proj,spaxel_sz_pc)

	#mask the gas temperature and tau to outside the continuum source
	dtau_nu = mask_inside_cont_src(dtau_nu,r,r_cont_src)
	dtau_fast_nu = mask_inside_cont_src(dtau_fast_nu,r,r_cont_src)
	dtau_hot_nu = mask_outside_cont_src(dtau_hot_nu,r,r_cont_src)
	
	#get the continuum tau and T
	T_cont_nu, tau_cont_nu = add_smooth_cont_source(box_spec,rr,T_cont,tau_cont_max,source_sz_pix)
		
	#mask out the continuum outside the continuum source
	T_cont_nu = mask_outside_cont_src(T_cont_nu,r,r_cont_src)
	tau_cont_nu = mask_outside_cont_src(tau_cont_nu,r,r_cont_src)


	#####################################
	##### FIND THE INTENSITY(x,y,z) #####
	#####################################

	#find dT at each cell
	dT_nu, T_nu = calc_T_nu_hot_fast(dtau_nu,Tex_nu,tau_cont_nu,T_cont_nu,dtau_hot_nu,Tex_hot_nu,dtau_fast_nu,Tex_fast_nu)

	if save3dplots==True:
		#get 3D cube where v(spec) = v(los)
		kappa_v = box.copy()
		dtau_v = box.copy()
		dT_v = box.copy()
		T_v = box.copy()
		vspec_v = box.copy()
		Tex_v = box.copy()
		Tex_hot_v = box.copy()
		Tex_fast_v = box.copy()
		for i in range(box_spec.shape[0]):
			for j in range(box_spec.shape[1]):
				for k in range(box_spec.shape[2]):
					idx = np.argmin(np.abs(velocity_los[i,j,k]-v_range))
					kappa_v[i,j,k] = kappa_nu_proj[i,j,k,idx]
					dtau_v[i,j,k] = dtau_nu[i,j,k,idx]+tau_cont_nu[i,j,k,idx]
					dT_v[i,j,k] = dT_nu[i,j,k,idx]
					T_v[i,j,k] = T_nu[i,j,k,idx]
					vspec_v[i,j,k] = v_range[idx]
					Tex_v[i,j,k] = Tex_nu[i,j,k,idx]
					Tex_hot_v[i,j,k] = Tex_hot_nu[i,j,k,idx]
					Tex_fast_v[i,j,k] = Tex_fast_nu[i,j,k,idx]

	###################
	##### OBSERVE #####
	###################

	#extract the temperature at the front of the box
	T_out = T_nu[:,:,0,:]

	#save this as a fits cube
	save_model_cube(T_out,bmaj,bmin,spaxel_sz_arcsec,chan_width,v_range,nu,theta,inc,phi,suffix,line,ssc)

	#mask out cylindrical region around the continuum source
	T_out_mask = mask_fov(T_out,r,r_cont_src)
	T_out_mask[T_out_mask==0.0]=np.nan

	#calculate line-of-sight and outflow masses as checks
	density_tot_los = mask_fov(density+density_hot+density_fast,r,r_cont_src)
	density_tot_los[density_tot_los==0.]=np.nan
	density_outflow = mask_fov(density+density_fast,r,r_cont_src)*outflow_mask_proj
	density_outflow[density_outflow==0.]=np.nan
	density_outflow_blue = density_outflow.copy()
	density_outflow_blue[velocity_los>=0.]=np.nan
	M_H2_los = mH2*spaxel_volume*np.nansum(density_tot_los)/M_sun
	M_H2_outflow = mH2*spaxel_volume*np.nansum(density_outflow)/M_sun
	M_H2_outflow_blue = mH2*spaxel_volume*np.nansum(density_outflow_blue)/M_sun
	
	#calculate line-of-sight column density as a check
	column_H2_tot_los = np.nanmean(np.nansum(density_tot_los,axis=2)*spaxel_sz_cm) #cm^-2
	column_H2_outflow = np.nanmean(np.nansum(density_outflow,axis=2)*spaxel_sz_cm) #cm^-2
	column_H2_outflow_blue = np.nanmean(np.nansum(density_outflow_blue,axis=2)*spaxel_sz_cm) #cm^-2

	m_H2_check = column_H2_tot_los*np.pi*(source_sz_pc/2*3.086E18)**2*mH2/M_sun
	print('### COMPARE COLUMN DENSITY ###')
	print('\tH_2 column density along los = %.1E cm^-2 = 10^%.1f cm^-2' %(column_H2_tot_los,np.log10(column_H2_tot_los)))
	print('\tH_2 column density in blueshifted outflow = %.1E cm^-2 = 10^%.1f cm^-2' %(column_H2_outflow_blue,np.log10(column_H2_outflow_blue)))

	print('### COMPARE MASS ###')
	print('\tTotal H_2 mass along los = %.1E M_sun = 10^%.1f M_sun' %(M_H2_los,np.log10(M_H2_los)))
	print('\tTotal H_2 mass in the blueshifted outflow = %.1E M_sun = 10^%.1f M_sun' %(M_H2_outflow_blue,np.log10(M_H2_outflow_blue)))

	
	#find the spectrum
	T_spec = np.nanmean(T_out_mask,axis=(0,1))


	#plot the resulting spectrum
	if save3dplots==True:
		plt_spec(v_range,T_spec,theta,inc,phi,1,suffix)


	if savemodel==True:
		#save the output spectrum to a file
		if cont_type=='sharpdisk':
			npix_str = str(npix_cont)+' pixels'
			T_cont_str = str(T_cont)+' K'
			tau_cont_str = 'inf'
		elif cont_type=='sharpsphere':
			npix_str = str(npix_cont)+' pixels'
			T_cont_str = str(T_cont)+' K'
			tau_cont_str = str(np.round(np.max(tau_cont_nu),2))
		elif cont_type=='smoothsphere':
			npix_str = 'N/A'
			T_cont_str = str(T_cont)+' K'
			tau_cont_str = str(np.round(np.max(tau_cont_nu),2))
		elif cont_type=='none':
			npix_str = 'N/A'
			T_cont_str = 'N/A'
			tau_cont_str = 'N/A'
		else:
			npix_str = 'N/A'
			T_cont_str = str(T_cont)+' K'
			tau_cont_str = str(np.round(np.max(tau_cont_nu),2))

		tab_dat = np.array([v_range,T_spec]).T

		if os.path.isdir(path+'Model_Spec_Tables/')==False:
			os.system('mdkir '+path+'Model_Spec_Tables/')
		fname = path+'Model_Spec_Tables/ModelSpec_SSC'+ssc+'_'+line+'_theta'+str(theta)+'_inc'+str(inc)+'_phi'+str(phi)+suffix+'.txt'
		
		comments = ('#ssc = '+ssc+'\n#theta = '+str(theta)+' deg\n#inc = '+str(inc)+' deg\n#phi = '+str(phi)+' deg\n'+
			'#Box = '+str(np.round(box_sz_pc,2))+' pc\n#Spaxel = '+str(np.round(spaxel_sz_pc,3))+' pc\n'+
			'#Channel width = '+str(np.round(chan_width,1))+' km/s\n'+
			'#V_outflow = '+str(v_outflow)+' km/s\n#V_outflow_fast = '+str(v_outflow_fast)+' km/s\n#sigma_v,FWHM = '+str(v_outflow_FWHM)+' km/s\n#sigma_v_fast,FWHM = '+str(v_outflow_fast_FWHM)+' km/s\n#sigma_v_hot,FWHM = '+str(v_hot_FWHM)+' km/s\n'+
			'#log10(max density) = '+str(np.round(np.log10(n_cen),1))+' cm^-3\n#log10(max density fast) = '+str(np.round(np.log10(n_cen_fast),1))+' cm^-3\n'+
			'#T_ex_cold = '+str(T_ex)+' K\n#T_ex_hot = '+str(T_hot)+' K\n#log10(max density hot) = '+str(np.round(np.log10(n_cen_hot),1))+' cm^-3\n'+
			'#Continuum Type = '+cont_type+'\n#Continuum radius = '+npix_str+'\n'+
			'#T_cont = '+T_cont_str+'\n#max(tau_cont) = '+tau_cont_str+'\n'+
			'#log10(N_H2_tot) = '+str(np.round(np.log10(column_H2_tot_los),1))+' cm^-2\n#log10(N_H2_blue) = '+str(np.round(np.log10(column_H2_outflow_blue),1))+' cm^-2\n'+
			'#log10(M_H2_tot) = '+str(np.round(np.log10(M_H2_los),1))+' Msun\n#log10(M_H2_blue) = '+str(np.round(np.log10(M_H2_outflow_blue),1))+' Msun\n'+
			'#Version = '+datetime.datetime.now().strftime("%Y-%m-%d %H:%M")+'\n')
		np.savetxt(fname,tab_dat,delimiter='\t',fmt='%.4e',
				   header='#V_los_km/s\tT_B_K',
				   comments=comments)

	################################
	##### PLOT RADIAL PROFILES #####
	################################
	plot_radial_profile(r_cen,rr_pc,Tex_nu,Tex_hot_nu,Tex_fast_nu,T_cont_nu,density,density_hot,density_fast,dtau_nu,dtau_hot_nu,dtau_fast_nu,tau_cont_nu,ssc,line,theta,inc,phi,suffix)

	######################
	##### PLOT IN 3D #####
	######################
	if save3dplots==True:

		#mask the cubes, set values outside to nan
		den_mask, vmax_den_mask, vmin_den_mask = mask_cubes(density,outflow_mask_proj)
		kappa_v_mask, vmax_kappa_v_mask, vmin_kappa_v_mask = mask_cubes(kappa_v,outflow_mask_proj)
		kappa_v_mask[np.isinf(kappa_v_mask)==1]=vmax_kappa_v_mask*100
		vlos_mask, vmax_vlos_mask, vmin_vlos_mask = mask_cubes(velocity_los,outflow_mask_proj)
		vspec_v_mask, vmax_vspec_v_mask, vmin_vspec_v_mask = mask_cubes(vspec_v,outflow_mask_proj)
		vspec_v_mask[vspec_v_mask==0.0]=np.nan
		dT_v_mask, vmax_dT_v_mask, vmin_dT_v_mask = mask_cubes(dT_v,outflow_mask_proj)
		T_v_mask, vmax_T_v_mask, vmin_T_v_mask = mask_cubes(T_v,outflow_mask_proj)
		dtau_v_mask, vmax_dtau_v_mask, vmin_dtau_v_mask = mask_cubes(dtau_v,outflow_mask_proj)

		#mask to FOV
		den_mask_fov = mask_fov(den_mask,r,r_cont_src)
		den_mask_fov[den_mask_fov==0.0]=np.nan
		T_v_mask = mask_fov(T_v_mask,r,r_cont_src)
		T_v_mask[T_v_mask==0.0]=np.nan

		#get coordinates
		x=np.ravel(rx)
		y=np.ravel(ry)
		z=np.ravel(rz)

		#make subplots
		print('Plotting figures...')
		fig = plt.figure(0,figsize=(9,5.5))
		plt.clf()
		data=dT_v_mask.copy()
		vmin_data = vmin_dT_v_mask
		vmax_data = vmax_dT_v_mask
		data[np.isnan(data)==True]=vmin_data
		cmap='Spectral_r'
		norm=LogNorm(vmin=vmin_data,vmax=vmax_data)
		clabel=''
		plt3d(fig.add_subplot(111,projection='3d'),x,y,z,data,r_cen,cmap,
		 		norm,clabel,'neither',theta,inc,phi,0)
		plt.show() 

		
	if savemodel==True:
		#compare against actual spectrum
		os.system('python ComparePCygniModel.py '+str(theta)+' '+str(inc)+' '+str(phi)+' '+path+' --suffix '+cont_type+' --line "'+line+'" --SSC_Number '+ssc+' --saveplot True')
		
	end=time.time()

	if printdiagnostics==True:
		print('\nTook '+str(np.round(end-start,1))+' s')

	return v_range,T_spec


### RUN ###
main(ssc,line,path)