import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
from pixell import enmap
import sys
import subprocess, os
import argparse
my_env = os.environ.copy()
my_env['DISABLE_MPI'] = 'true'

argparser = argparse.ArgumentParser()

argparser.add_argument("name", type=str, help="case number")

# Read positional argument for case
argparser.add_argument("name1", type=str, help="case number")
# Read optional boolean argument for random
argparser.add_argument("-u1", "--unrotated1", action="store_true", help="don't rotate stack")
argparser.add_argument("-r1", "--random1", action="store_true", help="random")
argparser.add_argument("-w1", "--websky1", action="store_true", help="use websky sim map")
argparser.add_argument("-m1", "--modelsub1", action="store_true", help="model subtraction")
argparser.add_argument("-s1", "--sz1", action="store_true", help="SZ clusters instead of halo catalog")
# Read optional argument for maximum number of objects
argparser.add_argument("-N1", "--N1", type=int, default=13000, help="number of objects to limit to")
# Read optional argument for number of randoms
argparser.add_argument("--nrand1", type=int, default=50000, help="number of randoms")

# Read positional argument for case
argparser.add_argument("name2", type=str, help="case number")
# Read optional boolean argument for random
argparser.add_argument("-u2", "--unrotated2", action="store_true", help="don't rotate stack")
argparser.add_argument("-r2", "--random2", action="store_true", help="random")
argparser.add_argument("-w2", "--websky2", action="store_true", help="use websky sim map")
argparser.add_argument("-m2", "--modelsub2", action="store_true", help="model subtraction")
argparser.add_argument("-s2", "--sz2", action="store_true", help="SZ clusters instead of halo catalog")
# Read optional argument for maximum number of objects
argparser.add_argument("-N2", "--N2", type=int, default=13000, help="number of objects to limit to")
# Read optional argument for number of randoms
argparser.add_argument("--nrand2", type=int, default=50000, help="number of randoms")

# Read positional argument for case
argparser.add_argument("name3", type=str, help="case number")
# Read optional boolean argument for random
argparser.add_argument("-u3", "--unrotated3", action="store_true", help="don't rotate stack")
argparser.add_argument("-r3", "--random3", action="store_true", help="random")
argparser.add_argument("-w3", "--websky3", action="store_true", help="use websky sim map")
argparser.add_argument("-m3", "--modelsub3", action="store_true", help="model subtraction")
argparser.add_argument("-s3", "--sz3", action="store_true", help="SZ clusters instead of halo catalog")
# Read optional argument for maximum number of objects
argparser.add_argument("-N3", "--N3", type=int, default=13000, help="number of objects to limit to")
# Read optional argument for number of randoms
argparser.add_argument("--nrand3", type=int, default=50000, help="number of randoms")

args = argparser.parse_args()
print(args)

args_run1 = []

if args.unrotated1:
    args_run1.append("-u")
if args.random1:
    args_run1.append("-r")
if args.websky1:
    args_run1.append("-w")
if args.modelsub1:
    args_run1.append("-m")
if args.sz1:
    args_run1.append("-s")
if args.N1:
    args_run1.extend(["-N", str(args.N1)])
if args.nrand1:
    args_run1.extend(["--nrand", str(args.nrand1)])
    
print("Running first stack")

# Define the arguments for the first run
run1 = ["python", "PrometheusTest.py", f'{args.name1}']
run1.extend(args_run1)

# # Run the first script with the specified arguments
result_run1 = subprocess.run(run1, capture_output=False, text=True, env=my_env)

print("Reading first map")

map1 = f'stack_{args.name1}.fits'
result1 = enmap.read_map(map1)

print("Running second stack")

args_run2 = []

if args.unrotated2:
    args_run2.append("-u")
if args.random2:
    args_run2.append("-r")
if args.websky2:
    args_run2.append("-w")
if args.modelsub2:
    args_run2.append("-m")
if args.sz2:
    args_run2.append("-s")
if args.N2:
    args_run2.extend(["-N", str(args.N2)])
if args.nrand2:
    args_run2.extend(["--nrand", str(args.nrand2)])

# Define the arguments for the second run
run2 = ["python", "PrometheusTest.py", f'{args.name2}']
run2.extend(args_run2)

result_run2 = subprocess.run(run2, capture_output=False, text=True, env=my_env)

print("Reading second map")

map2 = f'stack_{args.name2}.fits'
result2 = enmap.read_map(map2)

print("Running third stack")

args_run3 = []

if args.unrotated3:
    args_run3.append("-u")
if args.random3:
    args_run3.append("-r")
if args.websky3:
    args_run3.append("-w")
if args.modelsub3:
    args_run3.append("-m")
if args.sz3:
    args_run3.append("-s")
if args.N3:
    args_run3.extend(["-N", str(args.N3)])
if args.nrand3:
    args_run2.extend(["--nrand", str(args.nrand3)])

# Define the arguments for the second run
run3 = ["python", "PrometheusTest.py", f'{args.name3}']
run3.extend(args_run3)

result_run3 = subprocess.run(run3, capture_output=False, text=True, env=my_env)

print("Reading third map")

map3 = f'stack_{args.name3}.fits'
result3 = enmap.read_map(map3)

print("Creating composite map")

# Perform the subtraction or other desired operations on the FITS file data
composite_data = result1 - result2 - result3

# Display the result if applicable
plt.imshow(composite_data)
plt.colorbar()
plt.show()
plt.savefig(f'{args.name}.png')

