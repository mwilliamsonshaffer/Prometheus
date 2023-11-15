import numpy as np
import hmvec

# Create an instance of the Cosmology class (if needed)
cosmo_instance = hmvec.Cosmology()


def cluster_selection(mass_list, redshift_list,
                    RA_list, dec_list, 
                    mass_constraint=2e14, redshift_constraint=0.05):
    
    # Create boolean masks for mass and redshift constraints
    mass_mask = mass_list >= mass_constraint
    redshift_mask = redshift_list >= redshift_constraint

    # Combine the masks to select the valid entries
    valid_mask = mass_mask & redshift_mask 

    # Apply the mask to each array
    selected_masses = mass_list[valid_mask]
    selected_redshifts = redshift_list[valid_mask]
    selected_RA = RA_list[valid_mask]
    selected_dec = dec_list[valid_mask]

    return np.asarray(selected_RA), np.asarray(selected_dec)


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
