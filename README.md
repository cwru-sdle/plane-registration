# plane-registration

This repo has the scripts used to identify the x,y,z position of a defect from a cross section, such as metallography, EBSD, TEM, etc from the part

NOTE: The center is in the top left corner for images

# Coordinates

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
   5.
