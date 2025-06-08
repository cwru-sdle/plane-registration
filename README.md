# Intro

This repo has the scripts used to identify the x,y,z position of a defect from a cross section, such as metallography, EBSD, TEM, etc from the part

NOTE: The center is in the top left corner for images

NOTE: Will add CAD and cli processing to this repo, since it's fairly small

NOTE: Will add instructions for adding the docker image

# Saving Scripts

## [Defect From Parts](coordinates.py)

Here we are finding the xyz coordiantes from the metlalography samples. This is a little confusing so bear with me:

1. We load the following packages:

   1. roifil - to open the coordinates files from imagej
   2. os - for pathing
   3. numpy - for numeric operations
   4. cv2 - for momentums
   5. scipy - for root solving
   6. jupyter notebook - to run a script as a notebook
2. We load the measurements for the part edge lengths.

   1. We load every line and calculate their length in the folder and save it as a list of tuples
   2. We find the roi labeled 1 mm and calculate the factor to convert pixel length to millimeters
   3. Convert the line lengths into millimeters
3. Calculate alpha from the lengths

   1. Define the system of equations, which here is delta z, x_center and tan(alpha)
   2. Calculate theta
   3. Apply apha to the measurements. Here 1 is always the left side of the face any 2 is always the right side
4. Load the enscribining ellipses

   1. If the roi is a default, it is appended to a list, otherwise, it is put into a dictionary
   2. We find the roi labeled 1 mm and calculate the factor to convert pixel length to millimeters
   3. The center point is found by averaging the positions of all holes
   4. phi is calculate from these center points

# all-pyrometer

Here we are converting the pcd files to parquet files. This is needed because it lets us save a session as a single file, saves us a lot of space, and supports incrimental writes and reads. That last part is important because some of the sessions are over 100 gigabytes when uncompressed, which is more than we can load into RAM. This is also why we use parques with pyarrow instead of csvs with pandas. Pandas requires loading the entire dataframe into memory, while pyarrow allows for rows to be written to a file as groups, which can then be loaded into memory seperately. Here, each group corresponds to an individual pcd file, which are eigther saved as individuals layers within the print, or individual layers within a part, depending on the setting of the machine / files.
However, pyarrow does not have all of the features of a dataframe object, like joins, groupings, filtering, etc. Therefore, pyarrow handles all of the writes / reads, then pandas / polars is used for processing the batched data.

# Representations

The columns from the point clouds are:

['t',
 'x',
 'y',
 'z',
 'sensor0',
 'sensor1',
 'sensor2',
 'sensor3',
 'state0',
 'state1',
 'layer',
 'session',
 'config',
 'job']

# Sequences

# stgnn

A directed, homogenous, spatio temporal graphiacl nerual network is the most comprehensive way to represent this data.
