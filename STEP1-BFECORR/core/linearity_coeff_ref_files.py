"""Function file get the interpolate the linearity coefficients and generate a custom file

:History:

Created on Fri 16 Feb 2024

@author: Danny Gasman (KULeuven, Belgium, danny.gasman@kuleuven.be)
"""

# import python modules
import numpy as np
from pandas import read_pickle
from .distortion import d2cMapping
from .funcs import point_source_centroiding,detpixel_trace,detpixel_trace_compactsource
import datetime
from os import listdir
from astropy.io import fits
import os

# main function to call
def gen_custom_linearity(rate_file,dist_dir,num=1,dist_ver='flt7'):
    detector = rate_file[0].header['DETECTOR']
    subband = rate_file[0].header['BAND']
    
    if detector == 'MIRIFULONG':
        print('Cannot generate solution for long wavelength detector')
        
    else:
        band = ['1{}'.format(subband),'2{}'.format(subband)]
        
        # Get linearity reference file
        lin_ref = find_ref_linearity(band[0])
        
        coeffs_save = np.copy(lin_ref['COEFFS'].data)
        
        rate = rate_file['SCI'].data
        
        d2cMaps = {}
        alphadic = {}
        betadic = {}
        
        for b in band:
            
            if b[1:] == 'SHORT':
                subbandl = 'A'
            elif b[1:] == 'MEDIUM':
                subbandl = 'B'
            else:
                subbandl = 'C'
            
            # Get coordinate maps
            d2cMaps[b] = d2cMapping(b[0]+subbandl,dist_dir,slice_transmission='10pc',fileversion = "flt7")
            
            # Centroid input data
            alphadic[b[0]],betadic[b[0]] = return_centroids(rate,b,d2cMaps[b])
            print(alphadic[b[0]],betadic[b[0]])
            
            # Get coefficient file
            coeff_img1,coeff_img2 = open_coeff_ref(b)
            
            # Loop over detector and find solutions
            ypos,xpos = detpixel_trace_compactsource(rate,b+subbandl,d2cMaps[b],offset_slice=0)
        
            sliceID = d2cMaps[b]['sliceMap'][ypos[512],xpos[512]] - 100*int(b[0])
            alpha = np.nanmedian(d2cMaps[b]['alphaMap'][ypos,xpos])

            for slice_off in [-1,0,1]:
                coord=np.where(d2cMaps[b]['sliceMap'][ypos[512],xpos[512]] == sliceID+slice_off)
                ypos,xpos = detpixel_trace(b,d2cMaps[b],sliceID=sliceID+slice_off,alpha_pos=alpha)
                
                for xpixoff in [-2,-1,0,1,2]:
                    for i in range(len(ypos)):
                        alpha_off = d2cMaps[b]['alphaMap'][ypos[i],xpos[i]+xpixoff]-alphadic[b[0]]
                        if np.abs(alpha_off)<=0.4:  
                            if ypos[i] < 10:
                                y = 10
                            elif ypos[i]>1010:
                                y = 1010
                            else:
                                y = ypos[i]

                            coeff1,replace = spline2d_get_curve(coeff_img1,[alpha_off],[y])
                            if replace:
                                coeffs_save[2,ypos[i],xpos[i]+xpixoff] = coeff1[0,0]
                            coeff2,replace = spline2d_get_curve(coeff_img2,[alpha_off],[y])
                            if replace:
                                coeffs_save[3,ypos[i],xpos[i]+xpixoff] = coeff2[0,0]
        
        lin_ref['COEFFS'].data = coeffs_save
        
        lin_ref.writeto('./custom_linearity_ref_{}.fits'.format(num),overwrite=True)
            
def find_nearest_grid(rate_file,dist_dir,dist_ver='flt7'):
    detector = rate_file[0].header['DETECTOR']
    subband = rate_file[0].header['BAND']
    
    if detector == 'MIRIFULONG':
        print('Cannot generate solution for long wavelength detector')
        
    else:
        band = ['1{}'.format(subband)] #,'2{}'.format(subband)]
        
        # Get linearity reference file
        lin_ref = find_ref_linearity(band[0])
        
        coeffs_save = np.copy(lin_ref['COEFFS'].data)
        
        rate = rate_file['SCI'].data
        
        d2cMaps = {}
        alphadic = {}
        betadic = {}
        
        for b in band:
            
            if b[1:] == 'SHORT':
                subbandl = 'A'
            elif b[1:] == 'MEDIUM':
                subbandl = 'B'
            else:
                subbandl = 'C'
            
            # Get coordinate maps
            d2cMaps[b] = d2cMapping(b[0]+subbandl,dist_dir,slice_transmission='10pc',fileversion = "flt7")
            
            # Centroid input data
            alphadic[b[0]],betadic[b[0]] = return_centroids(rate,b,d2cMaps[b])
            print(alphadic[b[0]],betadic[b[0]])

            startdate = datetime.datetime(1900,1,1)
            if b[0] in ['1','2']:
                det = 'MIRIFUSHORT'
            elif b[0] in ['3','4']:
                det = 'MIRIFULONG'
                
            diff = 100
                
            for filename in listdir('./core/LINEARITY/'):
                
                if filename.startswith(det+'_'+subband):

                    hdu = fits.open('./core/LINEARITY/'+filename)
                    datestr = (hdu[0].header['DATE']).split('T')
                    date = datetime.datetime.strptime(datestr[0],'%Y-%m-%d')
                    refalpha = hdu[1].header['ALPHA{}'.format(b[0])]
                    refbeta = hdu[1].header['BETA{}'.format(b[0])]
                    hdu.close()

                    diffalpha = (refalpha-alphadic[b[0]])
                    diffbeta = (refbeta-betadic[b[0]])

                    if np.abs(diffbeta) < 0.03:
                        if np.abs(diffalpha) < diff:
                            reffile = filename
                            alphabest = refalpha
                            betabest = refbeta
                            diff = np.abs(diffalpha)

    return reffile

def return_centroids(rate,b,d2cMaps):
    
    if b[1:] == 'SHORT':
        subband = 'A'
    elif b[1:] == 'MEDIUM':
        subband = 'B'
    else:
        subband = 'C'
    
    lambmin = np.nanmin(d2cMaps['lambdaMap'][np.where(d2cMaps['lambdaMap']!=0)]) # micron
    lambmax = np.nanmax(d2cMaps['lambdaMap']) # micron
    lambcens = np.arange(lambmin,lambmax,(lambmax-lambmin)/1024.)
    lambfwhms = np.ones(len(lambcens))*(2*(lambmax-lambmin)/1024.)
    
    sign_amp2D,alpha_centers2D,beta_centers2D,sigma_alpha2D,sigma_beta2D,bkg_amp2D = point_source_centroiding(b,rate,d2cMaps,spec_grid=[lambcens,lambfwhms],fit='2D')

    alpha_ind=np.where(np.abs(alpha_centers2D)<=np.nanmax(np.abs(d2cMaps['alphaMap'])))[0][:]
    beta_ind=np.where(np.abs(beta_centers2D)<=np.nanmax(np.abs(d2cMaps['betaMap'])))[0][:]

    popt_alpha = np.polyfit(lambcens[alpha_ind],alpha_centers2D[alpha_ind],4)
    alpha = np.poly1d(popt_alpha)

    popt_beta  = np.polyfit(lambcens[beta_ind],beta_centers2D[beta_ind],4)
    beta = np.poly1d(popt_beta)

    return alpha(lambcens[512]),beta(lambcens[512])

def open_coeff_ref(band):
    if band[1:] == 'SHORT':
        subband = 'A'
    elif band[1:] == 'MEDIUM':
        subband = 'B'
    else:
        subband = 'C'
    
    coeff_dic1 = read_pickle('./core/coeff_fits/coeff_img_L2_{}.pickle'.format(band[0]+subband))
    coeff_dic2 = read_pickle('./core/coeff_fits/coeff_img_L3_{}.pickle'.format(band[0]+subband))
    
    return coeff_dic1,coeff_dic2

def find_ref_linearity(band):
    
    startdate = datetime.datetime(1900,1,1)
    
    crdsDir = './crds_cache/references/jwst/miri/'
    # crdsDir = '/Users/danny/crds_cache/references/jwst/miri/'
    
    subband = band[1:]
    
    if band[0] in ['1','2']:
        det = 'MIRIFUSHORT'
    elif band[0] in ['3','4']:
        det = 'MIRIFULONG'
    for filename in listdir(crdsDir):
        if filename.startswith('jwst_miri_linearity'):
            hdu = fits.open(crdsDir+filename)
            datestr = (hdu[0].header['DATE']).split('T')
            date = datetime.datetime.strptime(datestr[0]+' '+datestr[1],'%Y-%m-%d %H:%M:%S.%f')
            refdet = hdu[0].header['DETECTOR']
            refband = hdu[0].header['BAND'] if "BAND" in  hdu[0].header.keys() else 'NON'
            
            if refdet == det:
                if refband == 'N/A':
                    if date > startdate:
                        reffile = filename
                        startdate = date
                elif refband == subband:
                    if date > startdate:
                        reffile = filename
                        startdate = date
            
    ref_file = fits.open(crdsDir+reffile)
    
    return ref_file
        
# Get coefficient per alpha offset
def spline2d_get_curve(coeff_img,x,y):
    
    vals_out = np.empty((len(x),len(y)))
    dic_keys = np.array(list(coeff_img.keys()))
    
    replace = True
    
    #Compute chunks
    for i in range(len(x)):
        for j in range(len(y)):
            try:
                dic_id = np.where((y[j]==dic_keys[:,0]) & (x[i]>=dic_keys[:,1]) & (x[i]<dic_keys[:,2]))[0][0]

                vals_out[i,j]=coeff_img[dic_keys[dic_id,0],dic_keys[dic_id,1],dic_keys[dic_id,2]](x[i])
            except:
                replace = False
                continue
            
    return np.array(vals_out),replace
        
# Used to generate the spline fits, not needed by general user
def spline2d_in_chunks(x,y,z,x_min,x_max,x_space,y_min,y_max,y_space,pad=0.01,fitdeg=3,test=False):
    ny,nx = y_space,x_space # chunks
    pix_arr = np.linspace(y_min,y_max,ny) # pixels
    pos_arr = np.linspace(x_min,x_max,nx) # alpha offsets

    import matplotlib.pyplot as plt
    coeff_img = {}
    
    if test:
        plt.figure()
    
    #Compute chunks
    for i in range(len(pix_arr)-1):
        idy = np.where((y>=pix_arr[i]) & (y<pix_arr[i+1]))[0][:]
        zfity = z[idy]
        for j in range(len(pos_arr)-1):
            idx = np.where((x[idy]>=pos_arr[j]-pad) & (x[idy]<pos_arr[j+1]+pad))[0][:]
            zfitx = zfity[idx]
            if test:
                plt.scatter(x[idy[idx]],zfitx,alpha=0.3)
            
            #Fit chunk
            try:
                id_sorted = np.argsort(x[idy[idx]])
                
                coeffs = np.polyfit(x[idy[idx[id_sorted]]],zfitx[id_sorted],deg=fitdeg)
                model = np.poly1d(coeffs)
                
                if test:
                    plt.plot(x[idy[idx[id_sorted]]],model(x[idy[idx[id_sorted]]]),c='red')
                coeff_img[pix_arr[i],pos_arr[j],pos_arr[j+1]] = model
            except:
                continue
            
    return coeff_img


