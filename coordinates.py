# %%
'''Import Python Packages'''
import roifile
import os
import numpy as np
import cv2
import scipy
# %%
'''Load Lines'''
directory = "/home/anthony-lino/repos/plane-registration/samples/sample2/20/rois/"
lines = []
for file in os.listdir(directory):
    if ".roi" in file:
        roi = roifile.roiread(directory+file)
        array = np.array(roi.coordinates())
        lines.append(
            (
            np.linalg.norm(array[0]-array[1]),
            file,
            )
        )
for line in lines:
    if '1mm' in line[1]:
        mm_scale = 1/line[0]
lines_lengths = []
for line in lines:
    lines_lengths.append(
        (
            line[0]*mm_scale,
            line[1].replace(".roi","")

        )
    )
m_lengths = {name: val for val, name in lines_lengths}
# %%
'''Calculating Alpha'''
def alpha_equation(alpha,m1,m2, theta,triangle1,b):
    delta_z = np.abs((m1 - m2) * np.tan(theta)*np.cos(alpha))
    x_center = np.abs((m1 - m2) *np.cos(alpha))/2
    if triangle1:
        x = np.cos(alpha)*m1
        if m1>m2:
            x_center=-x_center
    else:
        x = np.cos(alpha)*m2
        if m2>m1:
            x_center=-x_center
    return np.tan(alpha)-(delta_z/2)/((b/2)+x+x_center)
L = (100 - b)/2
b = 50
h = 100
THETA = np.arctan((100/2)/(50/2))
alpha = {
    'top': scipy.optimize.fsolve(alpha_equation, x0=0, args=(m_lengths['top_1'],m_lengths['top_2'], THETA,True,b)),
    'bottom': scipy.optimize.fsolve(alpha_equation, x0=0, args=(m_lengths['bottom_1'],m_lengths['bottom_2'], THETA,True,b)),
    'left': scipy.optimize.fsolve(alpha_equation, x0=0, args=(m_lengths['left_1'],m_lengths['left_2'], THETA,True,b)),
    'right':scipy.optimize.fsolve(alpha_equation, x0=0, args=(m_lengths['right_1'],m_lengths['right_2'], THETA,True,b))
}
# %%
'''Load Ellipses'''
directory= "/home/anthony-lino/repos/plane-registration/samples/sample2/500/rois/"
centroid = []
holes = []
for file in os.listdir(directory):
    roi = roifile.roiread(directory+file)
    array = np.array(roi.coordinates())
    if '-' in file:
        M = cv2.moments(array)
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        centroid.append(np.array([cx, cy]))
    elif '1mm' in file:
        mm_scale - 1/np.linalg.norm(array[0]-array[1])
    else:
        M = cv2.moments(array)
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        holes.append(
            (
                np.array([cx, cy]),
                file.replace(".roi","")
            )
        )
holes = {name: val for val, name in holes}
# %%
'''Calculate Image Angles and Center'''
image_deltas= {
    'top':holes['top_right'] - holes['top_left'],
    'bottom':holes['bottm_right'] - holes['bottom_left'],
    'left':holes['bottom_left'] - holes['top_left'],
    'right':holes['bottm_right']- holes['top_right']
    }
image_angles = {
    'top':np.arctan(image_deltas['top'][1]/image_deltas['top'][0]),
    'bottom':np.arctan(image_deltas['bottom'][1]/image_deltas['bottom'][0]),
    'left':np.arctan(image_deltas['left'][0]/image_deltas['left'][1]),
    'right':np.arctan(image_deltas['right'][0]/image_deltas['right'][1])
}
center = np.mean([
    holes['bottom_left'],
    holes['bottm_right'],
    holes['middle'],
    holes['top_right'],
    holes['top_left']
    ], axis=0
)
# %%
'''z center'''
def z_equation(alpha,m1,m2,L):
    x1 = (m1*np.cos(alpha))-L
    x2 = (m2*np.cos(alpha))-L
    z1 = np.tan(THETA)
    z2 = (x2+L)*np.sin(alpha)
    return np.mean([z1,z2])
z_positions = {
    'left':z_equation(alpha=alpha['left'],m1=m_lengths['left_1'],m2=m_lengths['left_2'],L=(100 - b)/2),
    'right':z_equation(alpha=alpha['right'],m1=m_lengths['right_1'],m2=m_lengths['right_2'],L=(100 - b)/2),
    'bottom':z_equation(alpha=alpha['bottom'],m1=m_lengths['bottom_1'],m2=m_lengths['bottom_2'],L=(100 - b)/2),
    'top':z_equation(alpha=alpha['top'],m1=m_lengths['top_1'],m2=m_lengths['top_2'],L=(100 - b)/2),
}
z_center = np.mean([
    z_positions['left'],
    z_positions['right'],
    z_positions['top'],
    z_positions['bottom']
    ], axis=0
)
# %%
'''Print Results'''
print("ANGLES\n"+"-"*10)
print(alpha)
print("COORDINATES\n"+"-"*10)
print(center_values)
print("IMAGE_ANGLES\n"+"-"*10)
print(image_angles)
#print("DEFECTS\n"+"-"*10)
#print(centroid)