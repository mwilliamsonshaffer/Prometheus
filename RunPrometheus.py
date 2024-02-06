import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
from pixell import enmap
import sys
import subprocess, os
my_env = os.environ.copy()
my_env['DISABLE_MPI'] = 'true'

print("Running first stack")

# Define the arguments for the first run
args_run1 = ["python", "PrometheusTest.py", "case0_select", "--websky"]

# Run the first script with the specified arguments
result_run1 = subprocess.run(args_run1, capture_output=False, text=True, env= my_env)

sys.exit()

print("Reading first map")

map1 = "/home3/mayaws/Maya/stack_case0_select.fits"
result1 = enmap.read_map(map1)

print("Running second stack")

# Define the arguments for the second run
args_run2 = ["python", "PrometheusTest.py", "case0_random", "-r","--websky"]

# Run the second script with the specified arguments
result_run2 = subprocess.run(args_run2, capture_output=False, text=True)

print("Reading second map")

map2 = "/home3/mayaws/Maya/stack_case0_random.fits"
result2 = enmap.read_map(map2)

# Perform the subtraction or other desired operations on the FITS file data
composite_data = result1 - result2

# # Save the composite data to a new FITS file
# composite_filename = "/home3/mayaws/Maya/compositetest1.fits"  # Update this with the desired output filename
# fits.writeto(composite_filename, composite_data, overwrite=True)

# Display the result if applicable
plt.imshow(composite_data)
plt.colorbar()
plt.show()
plt.savefig('case0.png')


# import subprocess
# import matplotlib.pyplot as plt

# # Define the arguments for the first run
# args_run1 = ["python", "PrometheusTest.py", "test1", "--N 100"]

# # Run the first script with the specified arguments


# # Define the arguments for the second run
# args_run2 = ["python", "PrometheusTest.py", "randomtest1", "-r", "--nrand 100"]

# # Run the second script with the specified arguments


# composite_thumbnail = subprocess.run(args_run1) - subprocess.run(args_run2)

# plt.imshow(composite_thumbnail)
# plt.colorbar()
# plt.savefig('testy.png')
