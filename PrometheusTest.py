#imports
import matplotlib.pyplot as plt
import scipy
from astropy.io import fits
import numpy as np
import healpy as hp
from scipy.interpolate import interp1d
#import pixell
from pixell import reproject, enmap, utils, curvedsky
import time
from mpi4py import MPI
import numpy as np
import os,sys
import argparse
import putils
from p_tqdm import p_map
from functools import partial

argparser = argparse.ArgumentParser()
# Read positional argument for case
argparser.add_argument("name", type=str, help="case number")
# Read optional boolean argument for random
argparser.add_argument("-r", "--random", action="store_true", help="random")
argparser.add_argument("-w", "--websky", action="store_true", help="use websky sim map")
argparser.add_argument("-m", "--modelsub", action="store_true", help="model subtraction")
argparser.add_argument("-s", "--sz", action="store_true", help="SZ clusters instead of halo catalog")
# Read optional argument for maximum number of objects
argparser.add_argument("-N", "--N", type=int, default=None, help="number of objects to limit to")
argparser.add_argument("--rmax", type=float, default=10., help="maximum radius in arcmin")
argparser.add_argument("--decmin", type=float, default=-50., help="minimum dec for randoms in degrees")
#note: dec max is set to 20 in cluster selection
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
ACTmap = enmap.read_map(ACTsim)

# Converts alm files to an enmap
def read_enmap_alm(file):
    shape, wcs = enmap.fullsky_geometry(res=0.5 * utils.arcmin, proj='car')
    iheal = hp.read_alm(file)
    iheal = iheal.astype(np.float64)
    return curvedsky.alm2map(iheal,enmap.empty(shape,wcs,dtype=np.float64))

websky_map = read_enmap_alm('lensed_alm.fits')

if args.websky:
    mymap = websky_map

else:
    mymap = ACTmap

if args.modelsub:
    raise NotImplementedError
    model = '/data5/sims/websky/dr5_clusters/model_MFMF_pass2_cmb-tsz_f150.fits' # is this the right path?
    modelmap = enmap.read_map(model)
    imap = imap - modelmap

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

    lower_dec = np.deg2rad(args.decmin)  # Lower bound (inclusive)
    upper_dec = np.deg2rad(args.decmax)  # Upper bound (exclusive)

    # Generate an array of random numbers for theta
    # cosDEC = np.random.uniform(np.cos(lower_dec), np.cos(upper_dec), size=(args.nrand,))
    # DEC = np.arccos(cosDEC)
    DEC = np.random.uniform(lower_dec, upper_dec, size=(args.nrand,))
    #this for some reason is in radians

    # Generate an array of random numbers for phi
    RAdeg = np.random.uniform(0, 360., size=(args.nrand,))
    # in radians
    RA = np.deg2rad(RAdeg)

    # coords = RA, DEC
    coords = np.column_stack((DEC[:args.nrand], RA[:args.nrand]))

    mass = np.zeros(args.nrand)
    zs = np.zeros(args.nrand)

else:
    if (args.sz):
        SZcat = '/data5/sims/websky/dr5_clusters/ACTSim_CMB-T_cmb-tsz_MFMF_pass2_mass.fits'
        columnnames = 'M200m', 'redshift', 'RADeg', 'decDeg'
        function = load_fits(SZcat, columnnames, hdu_num=1,Nmax=None)
        mass = function['M200m'] * 1 *10**14
        zs = function['redshift']
        RA = np.deg2rad(function['RADeg'])
        DEC = np.deg2rad(function['decDeg'])
        #Note: These are originally in DEGREES. However I think the smallest degrees are listed first
        #now i have converted to radians

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
        RA = hp.vec2ang(np.column_stack((x,y,z)))[0] # in RADIANS
        DEC = hp.vec2ang(np.column_stack((x,y,z)))[1]
    mass, zs, RA, DEC = putils.cluster_selection(mass, zs, RA, DEC)
    coords = np.column_stack((DEC[:args.N], RA[:args.N]))

def take_thumbnails(icoords):
    image = reproject.thumbnails(mymap, icoords, r=args.rmax * utils.arcmin)
    return image

all_thumbnails = p_map(partial(take_thumbnails),coords)

def rescale_thumbnails(thumbnails, imass, iz):
    if not random:
        i_factor_value = putils.get_factor_i(imass, iz)
        wmap = thumbnails.copy()
        wmap.wcs.wcs.cdelt *= i_factor_value
        thumbnails = wmap.project(thumbnails.shape, thumbnails.wcs)
    return thumbnails

all_rescaled_thumbnails = p_map(partial(rescale_thumbnails), all_thumbnails, mass, zs)

def gradient(thumbnails):
    y_grad, x_grad = np.gradient(thumbnails)

    x_ave = np.mean(x_grad)
    y_ave = np.mean(y_grad)

    mag = np.sqrt(x_ave**2 + y_ave**2)
    angle = np.arctan2(y_ave, x_ave)

    return mag, angle

all_gradients = p_map(partial(gradient), all_thumbnails)


def rotate_thumbnails(thumbnail_gradients, thumbnails):
    angle = thumbnail_gradients[1]
    rotated_thumbnails = scipy.ndimage.rotate(thumbnails, np.degrees(angle), reshape=False, mode='constant', cval=np.nan)
    return rotated_thumbnails

all_rotated_thumbnails = p_map(partial(rotate_thumbnails), all_gradients, all_rescaled_thumbnails)


# # def check_threshold(rotated_thumbnails, threshold=1000):
# #     image_list_rotated = []

# #     # Check for threshold exceeding using NumPy
# #     if np.any(np.abs(rotated_thumbnails) > threshold):
# #         rotated_thumbnails = np.array(rotated_thumbnails)
# #         print(rotated_thumbnails.shape)
# #         image_list_rotated.append(rotated_thumbnails)

# #     return image_list_rotated

# # threshold_rotated_thumbnails = np.array(p_map(partial(check_threshold), all_rotated_thumbnails))
# # print(threshold_rotated_thumbnails)

def stack_thumbnails(thumbnails):
    return np.stack(thumbnails, axis=0)

stack_of_thumbnails = p_map(partial(stack_thumbnails), all_rotated_thumbnails)
print("Shape of one thumbnail in stack:", (np.array(stack_of_thumbnails[0])).shape)
print("Shape of stack of thumbnails:", (np.array(stack_of_thumbnails)).shape)

def average_stack(stack):
    return np.mean(stack, axis=0)

# averaged_thumbnail = p_map(average_stack, stack_of_thumbnails)
averaged_thumbnail = average_stack(stack_of_thumbnails)
print("Shape of averaged thumbnail:", (np.array(averaged_thumbnail)).shape)

enmap.write_map(f'stack_{args.name}.fits', averaged_thumbnail)
plt.imshow(averaged_thumbnail)
plt.colorbar()
plt.savefig(f'{args.name}.png')
