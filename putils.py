import numpy as np
import hmvec

# Create an instance of the Cosmology class (if needed)
cosmo_instance = hmvec.Cosmology()

def R_from_M(mass_list, redshift_list, delta=200): 
    return [float(3.*mass/4./np.pi/delta/cosmo_instance.rho_matter_z(redshift))**(1./3.) for mass, redshift in zip(mass_list, redshift_list)]

# def R_array(mass_list, redshift_list):
#     return [R_from_M(mass, redshift) for mass, redshift in zip(mass_list, redshift_list)]

def distance_array(redshift_list):
    return [cosmo_instance.results.angular_diameter_distance(redshift) for redshift in redshift_list]

def get_theta(mass_list, redshift_list):
    angular_distances = cosmo_instance.results.angular_diameter_distance(redshift_list)

    theta = R_from_M(mass_list, redshift_list) / angular_distances
    theta_deg = np.rad2deg(theta)
    theta_arc = theta_deg * 60
    
    return theta_arc

def get_factor_i(mass_list, redshift_list, theta_out=4):
    factor_i = theta_out / get_theta(mass_list, redshift_list)
    return factor_i

def cluster_selection(mass, zs, RA, DEC, lower_dec, upper_dec, upper_ra, i_factor,
                      mass_constraint=2e14, redshift_constraint=0.05, upper_i_factor= 1.5, lower_i_factor= .5):
    # Create boolean masks for mass and redshift constraints
    #dec constaint was added for websky halo catalog where dec goes up to 360, but using act map
    dec_mask = (DEC <= upper_dec) & (DEC >= lower_dec)
    ra_mask = RA <= upper_ra
    mass_mask = mass >= mass_constraint
    redshift_mask = zs >= redshift_constraint
    i_factor_mask = (i_factor <= upper_i_factor) & (i_factor >= lower_i_factor)

    # Combine the masks to select the valid entries
    valid_mask = dec_mask & ra_mask & mass_mask & redshift_mask & i_factor_mask

    # Apply the mask to each array
    selected_masses = mass[valid_mask]
    selected_redshifts = zs[valid_mask]
    selected_RA = RA[valid_mask]
    selected_dec = DEC[valid_mask]
    selected_i_factor = i_factor[valid_mask]

    return selected_masses, selected_redshifts, selected_RA, selected_dec, selected_i_factor

#check i factor values with histogram
#exclude for factor is smaller than .5 or greater than 1.5



# import numpy as np
# import hmvec

# # Create an instance of the Cosmology class (if needed)
# cosmo_instance = hmvec.Cosmology()


# def cluster_selection(mass, zs,
#                     RA, DEC, 
#                     mass_constraint=2e14, redshift_constraint=0.05):
    
#     # Create boolean masks for mass and redshift constraints
#     mass_mask = mass >= mass_constraint
#     redshift_mask = zs >= redshift_constraint

#     # Combine the masks to select the valid entries
#     valid_mask = mass_mask & redshift_mask 

#     # Apply the mask to each array
#     selected_masses = mass[valid_mask]
#     selected_redshifts = zs[valid_mask]
#     selected_RA = RA[valid_mask]
#     selected_dec = DEC[valid_mask]

#     return np.asarray(selected_masses), np.asarray(selected_redshifts), np.asarray(selected_RA), np.asarray(selected_dec)

# def R_from_M(i, mass_list, redshift_list, delta=200): 
#     return float(3.*mass_list[i]/4./np.pi/delta/cosmo_instance.rho_matter_z(redshift_list[i]))**(1./3.)

# def R_array(mass_list, redshift_list):
#     return [R_from_M(i, mass_list, redshift_list) for i in range(len(mass_list))]

# def distance_array(redshift_list):
#     return [cosmo_instance.results.angular_diameter_distance(redshift) for redshift in redshift_list]

# def get_theta(mass_list, redshift_list):
#     angular_distances = cosmo_instance.results.angular_diameter_distance(redshift_list)
    
#     non_zero_indices = angular_distances != 0
    
#     # Calculate theta values for non-zero angular distances
#     theta_values = R_from_M(np.arange(np.size(mass_list)), mass_list, redshift_list) / angular_distances[non_zero_indices]
#     theta_deg = np.rad2deg(theta_values)
#     theta_arc = theta_deg * 60
    
#     return theta_arc


# def get_factor_i(mass_list, redshift_list, theta_out=4):
#     return theta_out / get_theta(mass_list, redshift_list)

# def mean_factor(mass_list, redshift_list):
#     return np.mean(get_factor_i(mass_list, redshift_list))



# # def R_from_M(mass_list, redshift_list, delta=200): 
# #     return float(3.*mass_list[i]/4./np.pi/delta/cosmo_instance.rho_matter_z(redshift_list))**(1./3.)

# # def R_array(mass_list, redshift_list):
# #     return [R_from_M(mass_list, redshift_list)]

# # def distance_array(redshift_list):
# #     return [cosmo_instance.results.angular_diameter_distance(redshift) for redshift in redshift_list]

# # def get_theta(mass_list, redshift_list):
# #     angular_distances = cosmo_instance.results.angular_diameter_distance(redshift_list)
    
# #     theta_values = np.zeros_like(redshift_list)
# #     non_zero_indices = angular_distances != 0
    
# #     # Calculate theta values for non-zero angular distances using a loop
# #     for i in range(np.size(mass_list)):
# #         if np.any(non_zero_indices):
# #             theta = R_from_M(i, mass_list, redshift_list) / angular_distances[i]
# #             theta_deg = np.rad2deg(theta)
# #             theta_arc = theta_deg * 60
# #             theta_values[i] = theta_arc
    
# #     return theta_values

# # def get_factor_i(mass_list, redshift_list, theta_out = 4):
# #     return theta_out/get_theta(mass_list, redshift_list)

# # def mean_factor(mass_list, redshift_list):
# #     return np.mean(get_factor_i(mass_list, redshift_list))