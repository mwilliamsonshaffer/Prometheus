#imports
import matplotlib.pyplot as plt
import scipy
#import astropy.io
from astropy.io import fits
#from astropy import units as u
#import astropy.cosmology
##?from astropy import wcs
import numpy as np
from numpy import append
import healpy as hp
#from cosmology import *  # No star imports
from scipy.interpolate import interp1d
#import pixell
from pixell import reproject, enplot, enmap, utils, curvedsky
#import tarfile
#import orphics
#from orphics import catalogs
#import camb
#from camb import model, initialpower 
import time
from mpi4py import MPI
import numpy as np
import os,sys
import argparse
import putils

argparser = argparse.ArgumentParser()
# Read positional argument for case
argparser.add_argument("name", type=str, help="case number")
# Read optional boolean argument for random
argparser.add_argument("-r", "--random", action="store_true", help="random")
argparser.add_argument("-m", "--modelsub", action="store_true", help="model subtraction")
argparser.add_argument("-s", "--sz", action="store_true", help="SZ clusters instead of halo catalog")
# Read optional argument for maximum number of objects
argparser.add_argument("-N", "--N", type=int, default=None, help="number of objects to limit to")
argparser.add_argument("--rmax", type=float, default=10., help="maximum radius in arcmin")
argparser.add_argument("--decmin", type=float, default=-50., help="minimum dec for randoms in degrees")
argparser.add_argument("--decmax", type=float, default=20., help="maximum dec for randoms in degrees")
# Read optional argument for number of randoms
argparser.add_argument("--nrand", type=int, default=50000, help="number of randoms")
args = argparser.parse_args()

print(args)

random = args.random
print(random)

nrand = args.nrand
print(nrand)

Nmax = args.N
print(Nmax)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
ntasks = comm.Get_size()





ACTsim = '/data5/sims/websky/dr5_clusters/TOnly_ACTNoise_cmb-tsz_la145_CAR.fits'
imap = enmap.read_map(ACTsim)

if args.modelsub:
    raise NotImplementedError
    model = '/data5/sims/websky/dr5_clusters/model_MFMF_pass2_cmb-tsz_f150.fits' # is this the right path?
    modelmap = enmap.read_map(model)
    imap = imap - modelmap


# Converts alm files to an enmap
def read_enmap_alm(file):
    shape, wcs = enmap.fullsky_geometry(res=0.5 * utils.arcmin, proj='car')
    iheal = hp.read_alm(file)
    iheal = iheal.astype(np.float32)
    return curvedsky.alm2map(iheal,enmap.empty(shape,wcs,dtype=np.float32))

websky_map = read_enmap_alm('lensed_alm.fits')

# Loads catalog of real halo data
def load_fits(fits_file,column_names,hdu_num=1,Nmax=None):
    hdu = fits.open(fits_file)
    columns = {}
    for col in column_names:
        columns[col] = hdu[hdu_num].data[col][:Nmax]
    hdu.close()
    return columns

if random:
    # Creates coordinates for random locations in the lensed CMB map
    np.random.seed(0)

    lower_dec = args.decmin  # Lower bound (inclusive)
    upper_dec = args.decmax  # Upper bound (exclusive)

    # Generate an array of random numbers for theta
    cosDEC = np.random.uniform(np.cos(lower_dec), np.cos(upper_dec), size=(50000,))
    DEC = np.arccos(cosDEC)

    # Generate an array of random numbers for phi
    RA = np.random.uniform(0, 360., size=(50000,)) * utils.degree

else:
    if not(args.sz):
        SZcat = '/data5/sims/websky/dr5_clusters/ACTSim_CMB-T_cmb-tsz_MFMF_pass2_mass.fits'
        columnnames = 'M200m', 'redshift', 'RADeg', 'decDeg'
        function = load_fits(SZcat, columnnames, hdu_num=1,Nmax=None)
        mass = function['M200m'] * 1 *10**14
        zs = function['redshift']
        RA = function['RADeg'] * utils.degree
        DEC = function['decDeg'] * utils.degree
    else:
        # variables for Halo catalog
        omegab = 0.049
        omegac = 0.261
        omegam = omegab + omegac
        h      = 0.68
        ns     = 0.965
        sigma8 = 0.81
        c = 3e5
        H0 = 100*h
        nz = 100000
        z1 = 0.0
        z2 = 6.0
        za = np.linspace(z1,z2,nz)
        dz = za[1]-za[0]
        H = lambda z: H0*np.sqrt(omegam*(1+z)**3+1-omegam)
        dchidz = lambda z: c/H(z)
        chia = np.cumsum(dchidz(za))*dz
        zofchi = interp1d(chia,za)
        rho = 2.775e11*omegam*h**2 # Msun/Mpc^3d

        # Opens the first N terms in the Halos Catalog
        f=open('halos.pksc')
        N = 1000000
        Nhalo = np.fromfile(f, count=3, dtype=np.int32)[0]
        catalog=np.fromfile(f,count=N*10,dtype=np.float32)

        # Converts catalog data into a useful form
        catalog=np.reshape(catalog,(N,10))
        x  = catalog[:,0];  y = catalog[:,1];  z = catalog[:,2] # Mpc (comoving)
        vx = catalog[:,3]; vy = catalog[:,4]; vz = catalog[:,5] # km/sec
        R  = catalog[:,6] # Mpc

        # convert to mass, comoving distance, radial velocity, redshfit, RA and DEc
        mass    = 4*np.pi/3.*rho*R**3        # this is M200m (mean density 200 times mean) in Msun
        chi      = np.sqrt(x**2+y**2+z**2)    # Mpc
        vrad     = (x*vx + y*vy + z*vz) / chi # km/sec
        zs = zofchi(chi)  
        RA  = hp.vec2ang(np.column_stack((x,y,z)))[0] # in radians
        DEC = hp.vec2ang(np.column_stack((x,y,z)))[1]



    
def thumbnail(enmap, iRA, iDEC, imass, iz, rescale=True):
    
    coords = np.column_stack(([iDEC], [iRA]))
    
    image = reproject.thumbnails(enmap, coords, r=args.rmax * utils.arcmin)
    if image.shape[0]!=1: raise ValueError
    image = image[0]

    if rescale and not(random):
        mean_factor_value = mean_factor(imass, iz)
        wmap = image.copy()
        wmap.wcs.wcs.cdelt *= mean_factor_value
        image = wmap.project(image.shape, image.wcs)
    
    return image

# Calculates the magnitude of the average gradient of a thumbnail
def gradient(i, enmap, mass_list, redshift_list, RA_list, dec_list, random=False, rescale=False):
    thumb = thumbnail(i, enmap, mass_list, redshift_list, RA_list, dec_list, random=random, rescale=rescale)
    y_grad, x_grad = np.gradient(thumb)

    x_ave = np.mean(x_grad)
    y_ave = np.mean(y_grad)

    mag = np.sqrt(x_ave**2 + y_ave**2)
    angle = np.arctan2(y_ave, x_ave)

    return mag, angle

def averaged_map(n, enmap, mass_list, redshift_list, RA_list, dec_list, random = False, rescale=True):
    image_list = [thumbnail(i, enmap, mass_list, redshift_list, RA_list, dec_list, random=random, rescale=rescale) for i in range(n)]
    thumbnails = np.stack(image_list, axis=0)
    avg_thumbnail = np.mean(thumbnails, axis=0)
    return avg_thumbnail

def rotate_thumbnail(i, enmap, mass_list, redshift_list, RA_list, dec_list, random=False, rescale=True):
    
    thumbnail_image = thumbnail(i, enmap, mass_list, redshift_list, RA_list, dec_list, random=random, rescale=rescale)
    
    angle = gradient(i, enmap, mass_list, redshift_list, RA_list, dec_list, random=random, rescale=False)[1]
    
    rotated_thumbnail = scipy.ndimage.rotate(thumbnail_image, np.degrees(angle), reshape=False, mode='constant', cval=np.nan)
    
    return rotated_thumbnail

def averaged_map_rotated(n, enmap, mass_list, redshift_list, RA_list, dec_list, random=False, rescale=True, threshold=1000):
    image_list_rotated = []

    for i in range(n):
        rotated_thumbnail = rotate_thumbnail(i, enmap, mass_list, redshift_list,
                    RA_list, dec_list, random=random, rescale=rescale)
        grad = gradient(i, enmap, mass_list, redshift_list, RA_list, dec_list, random=random, rescale=False)[0]

        # Check for threshold exceeding using NumPy
        if np.any(np.abs(rotated_thumbnail) > threshold):
            continue

        image_list_rotated.append(rotated_thumbnail)

    # Stack the images into a single NumPy array
    stacked_images_rotated = np.stack(image_list_rotated, axis=0)

    # Compute the weighted average of the stacked thumbnails
    return np.average(stacked_images_rotated, axis=0)

print('testing')

if not(random):
    mass,zs, RA, DEC = putils.cluster_selection(mass, zs, RA, DEC)

stack = averaged_map_rotated(imap, mass, zs, RA, DEC)

enmap.write_map(f'stack_{args.name}.fits', stack)
plt.imshow(stack)
plt.savefig('stack.png')