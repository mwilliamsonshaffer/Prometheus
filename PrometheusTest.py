#imports
import matplotlib.pyplot as plt
import scipy
from astropy.io import fits
import numpy as np
import healpy as hp
from scipy.interpolate import interp1d
from pixell import reproject, enmap, utils as u, curvedsky
import numpy as np
import argparse
import putils
from p_tqdm import p_map
from functools import partial
import sys
import pandas as pd
from orphics import maps, cosmology

argparser = argparse.ArgumentParser()
# Read positional argument for case
argparser.add_argument("name", type=str, help="case number")
# Read optional arguments
argparser.add_argument("-x", "--unrotated", action="store_true", help="don't rotate stack")
argparser.add_argument("-r", "--random", action="store_true", help="random")
argparser.add_argument("-w", "--websky", action="store_true", help="use websky sim map")
argparser.add_argument("-q", "--qmap", action="store_true", help="use q sim map")
argparser.add_argument("-u", "--umap", action="store_true", help="use u sim map")
argparser.add_argument("-m", "--modelsub", action="store_true", help="model subtraction")
argparser.add_argument("-a", "--actlike", action="store_true", help="Act like clusters in halo catalog")
argparser.add_argument("-s", "--sz", action="store_true", help="SZ clusters instead of halo catalog")
argparser.add_argument("-i", "--rescale", action="store_true", help="i_factor rescale for random thumbnails")
argparser.add_argument("-n", "--noise", action="store_true", help="add beam and white noise")
# Read optional argument for maximum number of objects
argparser.add_argument("-N", "--N", type=int, default=13000, help="number of objects to limit to")
argparser.add_argument("--rmax", type=float, default=10., help="maximum radius in arcmin")
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

if (args.websky) or (args.qmap) or (args.umap):
    print("Reading Websky map")

    shape, wcs = enmap.fullsky_geometry(res=0.5 * u.arcmin, proj='car')
    new_shape = (3,) + shape

    # new_shape = (3, 21601, 43200)
    teb_alm = hp.read_alm('lensed_alm.fits', hdu=(1,2,3))
    print(teb_alm.shape)
    iqu_map = curvedsky.alm2map(teb_alm,enmap.empty(new_shape,wcs,dtype=np.float64),spin=[0,2])
    tmap = iqu_map[0]
    qmap = iqu_map[1]
    umap = iqu_map[2]

    if (args.qmap):
        mymap = qmap
    elif (args.umap):
        mymap = umap
    else: 
        mymap = tmap 

    lower_dec = -np.pi/2
    upper_dec = np.pi/2
    upper_ra = 2*np.pi

else:
    print("Reading ACT map")

    ACTsim = '/data5/sims/websky/nemo-sim-kit7/sim-kit_NemoWebSky_CustomLensedCMB_tenToA0Tuned_ACT-DR5-2Pass_2degTiles/sim-maps/simMap_f150.fits'
    # dr5_clusters/TOnly_ACTNoise_cmb-tsz_la145_CAR.fits'
    ACTmap = enmap.read_map(ACTsim)

    mymap = ACTmap
    lower_dec = np.deg2rad(-60)
    upper_dec = np.deg2rad(20)
    upper_ra = 2* np.pi

    if (args.modelsub):
        model = '/data5/sims/websky/nemo-sim-kit7/sim-kit_NemoWebSky_CustomLensedCMB_tenToA0Tuned_ACT-DR5-2Pass_2degTiles/NemoWebSky_CustomLensedCMB_tenToA0Tuned_ACT-DR5-2Pass_2degTiles/clusterModelMaps/clusterModelMap_f150.fits'
        #dr5_clusters/model_MFMF_pass2_cmb-tsz_f150.fits' # is this the right path?
        modelmap = enmap.read_map(model)
        mymap = mymap - modelmap

if (args.noise):

    print("Adding noise")
    nlevel = 1.0
    fwhm = 1.5
    
    def apply_beam(imap): 
        # map2alm of the maps, almxfl(alm, beam_1d) to convolve with beam, alm2map to convert back to map
        nside = 8192
        alm_lmax = nside * 3
        bfunc = lambda x: maps.gauss_beam(fwhm, x)  
        imap_alm = curvedsky.map2alm(imap, lmax=alm_lmax)
        beam_convoloved_alm = curvedsky.almxfl(imap_alm, bfunc)
        return curvedsky.alm2map(beam_convoloved_alm, enmap.empty(imap.shape, imap.wcs))

    white_noise = maps.white_noise(mymap.shape, mymap.wcs, noise_muK_arcmin=nlevel)
    mymap = apply_beam(mymap) + white_noise

print("Loading Catalogs")

def load_fits(fits_file,column_names,hdu_num=1,Nmax=None):
    hdu = fits.open(fits_file)
    columns = {}
    for col in column_names:
        columns[col] = hdu[hdu_num].data[col][:Nmax]
    hdu.close()
    return columns

if random:
    print("Generating random coordinates")

    DEC = np.random.uniform(lower_dec, upper_dec, size=(args.nrand,))
    #in radians

    RA = np.random.uniform(0, upper_ra, size=(args.nrand,))
    # in radians

    # coords = dec, ra
    coords = np.column_stack((DEC[:args.nrand], RA[:args.nrand]))

    mass = np.zeros(args.nrand)
    zs = np.zeros(args.nrand)
    
else:

    if (args.sz):
        print("Loading SZ Catalog")

        SZcat = '/data5/sims/websky/nemo-sim-kit7/sim-kit_NemoWebSky_CustomLensedCMB_tenToA0Tuned_ACT-DR5-2Pass_2degTiles/NemoWebSky_CustomLensedCMB_tenToA0Tuned_ACT-DR5-2Pass_2degTiles/NemoWebSky_CustomLensedCMB_tenToA0Tuned_ACT-DR5-2Pass_2degTiles_mass.fits'
        # dr5_clusters/ACTSim_CMB-T_cmb-tsz_MFMF_pass2_mass.fits'
        columnnames = 'M200m', 'redshift', 'RADeg', 'decDeg', 'SNR'
        function = load_fits(SZcat, columnnames, hdu_num=1,Nmax=None)

        mass = function['M200m'] * 1 *10**14
        zs = function['redshift']
        RA = np.deg2rad(function['RADeg'])
        DEC = np.deg2rad(function['decDeg'])
        SNR = function['SNR']

    elif (args.actlike):
        print("Loading ACT-Like Halo Catalog")

        actlike_cat = '/data5/sims/websky/nemo-sim-kit7/websky_halo_snr5p5.txt'
        data = pd.read_csv(actlike_cat, sep=" ", header=None)
        data.columns = ['RADeg', 'decDeg', 'redshift', 'mass']

        mass = data['mass']* 1 *10**14
        zs = data['redshift']
        RA = np.deg2rad(data['RADeg'])
        DEC = np.deg2rad(data['decDeg'])
        SNR = np.zeros(len(mass))

    else:
        print("Loading Full Halo Catalog")

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

        catalog=np.reshape(catalog,(N,10))
        x  = catalog[:,0];  y = catalog[:,1];  z = catalog[:,2] # Mpc (comoving)
        vx = catalog[:,3]; vy = catalog[:,4]; vz = catalog[:,5] # km/sec
        R  = catalog[:,6] # Mpc

        # convert to mass, comoving distance, radial velocity, redshfit, RA and DEc
        mass    = 4*np.pi/3.*rho*R**3        # this is M200m (mean density 200 times mean) in Msun
        chi      = np.sqrt(x**2+y**2+z**2)    # Mpc
        vrad     = (x*vx + y*vy + z*vz) / chi # km/sec
        zs = zofchi(chi)  
        theta = hp.vec2ang(np.column_stack((x,y,z)))[0] # in RADIANS
        # 90-dec
        DEC = np.pi/2 - theta
        RA = hp.vec2ang(np.column_stack((x,y,z)))[1]
        SNR = np.zeros(len(mass))

    print("Selecting Coordinates")

    # i_factor = putils.get_factor_i(mass, zs)
    mass, zs, RA, DEC, SNR = putils.cluster_selection(mass, zs, RA, DEC, lower_dec, upper_dec, upper_ra, SNR)
    coords = np.column_stack((DEC[:args.N], RA[:args.N]))
    # np.savetxt(f'i_factor_{args.name}.txt', i_factor)
    
print("Taking thumbnails")

def take_thumbnails(icoords):
    image = reproject.thumbnails(mymap, icoords, r=args.rmax * u.arcmin, res= 0.5 * u.arcmin)
    return image

all_thumbnails = p_map(partial(take_thumbnails),coords)

if (args.noise):
    print("Applying Wiener Filter")

    def default_theory(lpad=9000,root="cosmo2017_10K_acc3"):
        cambRoot = f"/home3/mayaws/orphics/data/{root}"
        return cosmology.loadTheorySpectraFromCAMB(cambRoot,unlensedEqualsLensed=False,useTotal=False,TCMB = 2.7255e6,lpad=lpad,get_dimensionless=False)

    theory = default_theory()
    ells = np.arange(8000)
    cltt = theory.lCl('TT',ells)
    clee = theory.lCl('EE', ells)

    if (args.qmap) or (args.umap):
        def wiener_filter(thumbnail):
            nell = (nlevel * np.pi / 180. / 60. / maps.gauss_beam(ells, fwhm)) ** 2
            wiener = clee / (clee + 2*nell/ maps.gauss_beam(ells, fwhm)**2)
            taper, w2 = maps.get_taper(thumbnail.shape, thumbnail.wcs)
            fthumb = enmap.fft(thumbnail * taper, normalize='phys')
            modlmap_values = thumbnail.modlmap()
            wiener_interp = maps.interp(ells, wiener)(modlmap_values)
            fthumb = fthumb * wiener_interp
            filtered_thumbnail = enmap.ifft(fthumb, normalize='phys').real
            return filtered_thumbnail
        
        def inverse_variance_filter(thumbnail):
            nell = (nlevel * np.pi / 180. / 60. / maps.gauss_beam(ells, fwhm)) ** 2
            inverse = 1 / (clee + 2*nell/ maps.gauss_beam(ells, fwhm)**2)
            taper, w2 = maps.get_taper(thumbnail.shape, thumbnail.wcs)
            fthumb = enmap.fft(thumbnail * taper, normalize='phys')
            modlmap_values = thumbnail.modlmap()
            wiener_interp = maps.interp(ells, inverse)(modlmap_values)
            fthumb = fthumb * wiener_interp
            filtered_thumbnail = enmap.ifft(fthumb, normalize='phys').real
            return filtered_thumbnail
            
    else:
        def wiener_filter(thumbnail):
            nell = (nlevel * np.pi / 180. / 60. / maps.gauss_beam(ells, fwhm)) ** 2
            wiener = cltt / (cltt + nell)
            taper, w2 = maps.get_taper(thumbnail.shape, thumbnail.wcs)
            fthumb = enmap.fft(thumbnail * taper, normalize='phys')
            modlmap_values = thumbnail.modlmap()
            wiener_interp = maps.interp(ells, wiener)(modlmap_values)
            fthumb = fthumb * wiener_interp
            filtered_thumbnail = enmap.ifft(fthumb, normalize='phys').real
            return filtered_thumbnail
        
        def inverse_variance_filter(thumbnail):
            nell = (nlevel * np.pi / 180. / 60. / maps.gauss_beam(ells, fwhm)) ** 2
            inverse = 1 / (cltt + nell)
            taper, w2 = maps.get_taper(thumbnail.shape, thumbnail.wcs)
            fthumb = enmap.fft(thumbnail * taper, normalize='phys')
            modlmap_values = thumbnail.modlmap()
            wiener_interp = maps.interp(ells, inverse)(modlmap_values)
            fthumb = fthumb * wiener_interp
            filtered_thumbnail = enmap.ifft(fthumb, normalize='phys').real
            return filtered_thumbnail
    
    all_wiener_thumbnails = p_map(partial(wiener_filter), all_thumbnails)
    all_inverse_thumbnails = p_map(partial(inverse_variance_filter), all_thumbnails)

else:
    all_wiener_thumbnails = all_thumbnails
    all_inverse_thumbnails = all_thumbnails


if not (args.unrotated):
    print("Calculating gradient")

    def gradient(thumbnails):
        y_grad, x_grad = np.gradient(thumbnails)

        x_ave = np.mean(x_grad)
        y_ave = np.mean(y_grad)

        mag = np.sqrt(x_ave**2 + y_ave**2)
        angle = np.arctan2(y_ave, x_ave)

        return mag, angle

    if (args.noise):
        all_gradients = p_map(partial(gradient), all_wiener_thumbnails)
        all_gradients = np.array(all_gradients)
    else:
        all_gradients = p_map(partial(gradient), all_thumbnails)
        all_gradients = np.array(all_gradients)

    print("Rotating thumbnails")

    def rotate_thumbnails(thumbnail_gradients, thumbnails):
        angle = thumbnail_gradients[1]
        rotated_thumbnails = scipy.ndimage.rotate(thumbnails, np.degrees(angle), reshape=False, cval=np.nan)
        return rotated_thumbnails

    all_rotated_thumbnails = p_map(partial(rotate_thumbnails), all_gradients, all_inverse_thumbnails)

else:
    all_rotated_thumbnails = all_thumbnails

print("Stacking thumbnails")

def stack_thumbnails(thumbnails):
    return np.stack(thumbnails, axis=0)

stack_of_thumbnails = p_map(partial(stack_thumbnails), all_rotated_thumbnails)

print((np.array(stack_of_thumbnails)).shape)

print("Averaging stack")

def average_stack(stack):
    magnitudes = all_gradients[:, 0]
    weighted_average = np.average(stack, axis=0, weights=magnitudes)
    return weighted_average

averaged_thumbnail = average_stack(stack_of_thumbnails)
print("Shape of averaged thumbnail:", (np.array(averaged_thumbnail)).shape)


def mask_outside_circle(image, radius=20):
    # Create a circular mask
    center_x = image.shape[1] // 2
    center_y = image.shape[0] // 2
    y, x = np.ogrid[:image.shape[0], :image.shape[1]]
    mask = ((x - center_x)**2 + (y - center_y)**2) > radius**2
    
    # Apply the mask to the image and set values outside the circle to NaN
    masked_image = np.copy(image)
    masked_image[mask] = np.nan
    
    return masked_image

masked_thumbnail = mask_outside_circle(averaged_thumbnail)

print("Writing enmap")
enmap.write_map(f'stack_{args.name}.fits', masked_thumbnail)
print(f'stack_{args.name}.fits')

print("Reading enmap")
result = enmap.read_map(f'stack_{args.name}.fits')
print("Plotting enmap")
plt.imshow(result)

plt.colorbar()
# plt.clim(-30, 30)
plt.show()
plt.savefig(f'{args.name}.png')