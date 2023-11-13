#imports
import matplotlib.pyplot as plt
import scipy
import astropy.io
from astropy.io import fits
from astropy import units as u
import astropy.cosmology
from astropy import wcs
import numpy as np
from numpy import append
import healpy as hp
from cosmology import *
from scipy.interpolate import *
import pixell
from pixell import reproject, enplot, enmap, utils
import tarfile
import orphics
from orphics import catalogs
import camb
from camb import model, initialpower 
import hmvec
import time
from mpi4py import MPI
import numpy as np
import os,sys

# from pixell.utils
def allgather(a, comm):
    """Convenience wrapper for Allgather that returns the result
    rather than needing an output argument."""

    # Create an instance of the Cosmology class (if needed)
    cosmo_instance = hmvec.Cosmology()

    ACTsim = '/data5/sims/websky/dr5_clusters/TOnly_ACTNoise_cmb-tsz_la145_CAR.fits'
    ACTsim_enmap = enmap.read_map(ACTsim)
    ACTsim_map = ACTsim_enmap.astype(np.float64)

    model = '/data5/sims/websky/dr5_clusters/model_MFMF_pass2_cmb-tsz_f150.fits'
    model_enmap = enmap.read_map(model)
    model_map = model_enmap.astype(np.float64)

    # Opens fits files that are in alm form
    def read_healpy_alm(file):
        A1 = hp.read_alm(file)
        NSIDE1 = 1024
        return hp.alm2map(A1,NSIDE1)

        # Converts alm files to an enmap
    def read_enmap_alm(file):
        shape, wcs = enmap.fullsky_geometry(res=0.5 * utils.arcmin, proj='car')
        iheal = read_healpy_alm(file)
        iheal = iheal.astype(np.float32)
        return reproject.healpix2map(iheal,shape,wcs)

    websky_map = read_enmap_alm('lensed_alm.fits')

    # Loads catalog of real halo data
    def load_fits(fits_file,column_names,hdu_num=1,Nmax=None):
        hdu = fits.open(fits_file)
        columns = {}
        for col in column_names:
            columns[col] = hdu[hdu_num].data[col][:Nmax]
        hdu.close()
        return columns

    SZcat = '/data5/sims/websky/dr5_clusters/ACTSim_CMB-T_cmb-tsz_MFMF_pass2_mass.fits'
    columnnames = 'M200m', 'redshift', 'RADeg', 'decDeg'
    function = load_fits(SZcat, columnnames, hdu_num=1,Nmax=None)
    SZcat_mass = function['M200m'] * 1 *10**14
    SZcat_redshift = function['redshift']
    SZcat_RA = function['RADeg'] * utils.degree
    SZcat_dec = function['decDeg'] * utils.degree

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
    webskycat_mass    = 4*np.pi/3.*rho*R**3        # this is M200m (mean density 200 times mean) in Msun
    chi      = np.sqrt(x**2+y**2+z**2)    # Mpc
    vrad     = (x*vx + y*vy + z*vz) / chi # km/sec
    webskycat_redshift = zofchi(chi)  
    webskycat_RA  = hp.vec2ang(np.column_stack((x,y,z)))[0] # in radians
    webskycat_dec = hp.vec2ang(np.column_stack((x,y,z)))[1]

    def cluster_selection(mass_list, redshift_list,
                        RA_list, dec_list, dec_constraint= np.max(SZcat_dec), 
                        mass_constraint=2e14, redshift_constraint=0.05):
        
        # Create boolean masks for mass and redshift constraints
        dec_mask = dec_list <= dec_constraint
        mass_mask = mass_list >= mass_constraint
        redshift_mask = redshift_list >= redshift_constraint

        # Combine the masks to select the valid entries
        valid_mask = mass_mask & redshift_mask & dec_mask

        # Apply the mask to each array
        selected_masses = mass_list[valid_mask]
        selected_redshifts = redshift_list[valid_mask]
        selected_RA = RA_list[valid_mask]
        selected_dec = dec_list[valid_mask]

        return np.asarray(selected_masses), np.asarray(selected_redshifts), np.asarray(selected_RA), np.asarray(selected_dec)

    # Creates coordinates for random locations in the lensed CMB map
    np.random.seed(0)

    lower_dec = 0  # Lower bound (inclusive)
    upper_dec = np.pi  # Upper bound (exclusive)

    # Generate an array of random numbers for theta
    random_dec = np.random.uniform(lower_dec, upper_dec, size=(50000,))

    lower_RA = np.min(SZcat_dec)  # Lower bound (inclusive)
    upper_RA = np.max(SZcat_dec)  # Upper bound (exclusive)

    # Generate an array of random numbers for phi
    random_RA = np.random.uniform(lower_RA, upper_RA, size=(50000,))

    def R_from_M(i, mass_list, redshift_list, delta=200): 
        return float(3.*mass_list[i]/4./np.pi/delta/cosmo_instance.rho_matter_z(redshift_list[i]))**(1./3.)

    def R_array(n, mass_list, redshift_list):
        return [R_from_M(i, mass_list, redshift_list) for i in range(n)]

    def distance_array(n, redshift_list):
        return [cosmo_instance.results.angular_diameter_distance(redshift) for redshift in redshift_list]

    def get_theta(mass_list, redshift_list):
        angular_distances = cosmo_instance.results.angular_diameter_distance(redshift_list)
        
        theta_values = np.zeros_like(redshift_list)
        non_zero_indices = angular_distances != 0
        
        # Calculate theta values for non-zero angular distances using a loop
        for i in range(len(mass_list)):
            if non_zero_indices[i]:
                theta = R_from_M(i, mass_list, redshift_list) / angular_distances[i]
                theta_deg = np.rad2deg(theta)
                theta_arc = theta_deg * 60
                theta_values[i] = theta_arc
        
        return theta_values

    def get_factor_i(mass_list, redshift_list, theta_out = 4):
        return theta_out/get_theta(mass_list, redshift_list)

    def mean_factor(mass_list, redshift_list):
        return np.mean(get_factor_i(mass_list, redshift_list))
        
    def thumbnail(i, enmap, mass_list, redshift_list, RA_list, dec_list, random=False, rescale=True):
        if random:
            RA = random_RA
            dec = random_dec
        else:
            cluster_data = cluster_selection(mass_list, redshift_list, RA_list, dec_list)
            RA = cluster_data[2]
            dec = cluster_data[3]

        coords = np.column_stack((dec, RA))
        coords_copy = coords.copy()  # Make a copy if you intend to modify coords in the future

        image = reproject.thumbnails(enmap, coords[i], r=10 * utils.arcmin)

        if rescale:
            mean_factor_value = mean_factor(mass_list, redshift_list)
            wmap = image.copy()
            wmap.wcs.wcs.cdelt *= mean_factor_value
            omap = wmap.project(image.shape, image.wcs)
        else:
            omap = image

        return omap

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

    n_webskycat= len(cluster_selection(webskycat_mass, webskycat_redshift,
                        webskycat_RA, webskycat_dec))

    plt.imshow(averaged_map_rotated(n_webskycat, websky_map, webskycat_mass, webskycat_redshift, webskycat_RA, webskycat_dec)
            -averaged_map_rotated(50000, websky_map, webskycat_mass, webskycat_redshift, webskycat_RA, webskycat_dec, random=True))
    plt.colorbar()
    plt.savefig('/home3/mayaws/Maya/case1')

    plt.imshow(averaged_map_rotated(n_webskycat, ACTsim_map, webskycat_mass, webskycat_redshift, webskycat_RA, webskycat_dec)
            -averaged_map_rotated(50000, ACTsim_map, webskycat_mass, webskycat_redshift, webskycat_RA, webskycat_dec, random=True))
    plt.colorbar()
    plt.savefig('/home3/mayaws/Maya/case2')

    plt.imshow(averaged_map_rotated(n_webskycat, (ACTsim_map-model_map), webskycat_mass, webskycat_redshift, webskycat_RA, webskycat_dec)
            -averaged_map_rotated(50000, (ACTsim_map-model_map), webskycat_mass, webskycat_redshift, webskycat_RA, webskycat_dec, random=True))
    plt.colorbar()
    plt.savefig('/home3/mayaws/Maya/case3')

    plt.imshow(averaged_map_rotated(n_webskycat, (ACTsim_map- averaged_map(webskycat_mass, webskycat_redshift, webskycat_RA, webskycat_dec)), webskycat_mass, webskycat_redshift, webskycat_RA, webskycat_dec)
            -averaged_map_rotated(50000, (ACTsim_map- averaged_map(webskycat_mass, webskycat_redshift, webskycat_RA, webskycat_dec)), webskycat_mass, webskycat_redshift, webskycat_RA, webskycat_dec, random=True))
    plt.colorbar()
    plt.savefig('/home3/mayaws/Maya/case4')

    a   = np.asarray(a)
    res = np.zeros((comm.size,)+a.shape,dtype=a.dtype)
    if np.issubdtype(a.dtype, np.string_):
        comm.Allgather(a.view(dtype=np.uint8), res.view(dtype=np.uint8))
    else:
        comm.Allgather(a, res)
    return res

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
ntasks = comm.Get_size()

if ntasks<=1: raise ValueError

my_value = np.asarray([rank,])

print(rank,ntasks)

all_values = allgather(my_value,comm)
sum = all_values.sum()
if not(sum==(ntasks*(ntasks-1)/2)): raise ValueError
print(all_values)
print("Success.")
