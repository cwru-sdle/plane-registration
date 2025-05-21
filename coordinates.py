# %%
'''Import Python Packages'''
import roifile
import os
import numpy as np
import cv2
import scipy
import pandas as pd
data_directory = "/home/anthony-lino/repos/plane-registration/data/"
# %%
'''Load Lines'''
directory = data_directory+"microscope/sample2/20/rois/"
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
        mm_scale_20x = 1/line[0]
lines_lengths = []
for line in lines:
    lines_lengths.append(
        (
            line[0]*mm_scale_20x,
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
b = 5
h = 10
L = (10 - b)/2
THETA = np.arctan((h/2)/(b/2))
alpha = {
    'top': scipy.optimize.fsolve(alpha_equation, x0=0, args=(m_lengths['top_1'],m_lengths['top_2'], THETA,True,b)),
    'bottom': scipy.optimize.fsolve(alpha_equation, x0=0, args=(m_lengths['bottom_1'],m_lengths['bottom_2'], THETA,True,b)),
    'left': scipy.optimize.fsolve(alpha_equation, x0=0, args=(m_lengths['left_1'],m_lengths['left_2'], THETA,True,b)),
    'right':scipy.optimize.fsolve(alpha_equation, x0=0, args=(m_lengths['right_1'],m_lengths['right_2'], THETA,True,b))
}
# %%
'''Load Ellipses'''
directory= data_directory+"microscope/sample2/500/rois/"
centroid_500x = []
holes = []
for file in os.listdir(directory):
    roi = roifile.roiread(directory+file)
    array = np.array(roi.coordinates())
    if '-' in file:
        M = cv2.moments(array)
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        centroid_500x.append(np.array([cx, cy]))
    elif '1mm' in file:
        mm_scale_500x = 1/np.linalg.norm(array[0]-array[1])
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
phi = np.mean([
    image_angles['top'],
    image_angles['bottom'],
    image_angles['left'],
    image_angles['right']
    ], axis=0
)
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
def z_equation(alpha,m1,m2,L=(10 - b)/2,theta=np.radians(60)):
    x1 = (m1*np.cos(alpha))-L
    x2 = (m2*np.cos(alpha))-L
    z1 = np.tan(theta)*x1
    z2 = np.tan(theta)*x2
    return np.mean([z1,z2])
z_positions = {
    'left':z_equation(alpha=alpha['left'],m1=m_lengths['left_1'],m2=m_lengths['left_2'],),
    'right':z_equation(alpha=alpha['right'],m1=m_lengths['right_1'],m2=m_lengths['right_2']),
    'bottom':z_equation(alpha=alpha['bottom'],m1=m_lengths['bottom_1'],m2=m_lengths['bottom_2']),
    'top':z_equation(alpha=alpha['top'],m1=m_lengths['top_1'],m2=m_lengths['top_2']),
}
z_center = np.mean([
    z_positions['left'],
    z_positions['right'],
    z_positions['top'],
    z_positions['bottom']
    ], axis=0
)
#%%
'''Find Position'''
def z_position(position, alpha, beta, z_center,phi):
    x_CAD = position[0]*np.cos(phi)+position[1]*np.sin(phi)
    y_CAD = -position[0]*np.sin(phi)+position[1]*np.cos(phi)
    z = z_center + x_CAD * np.tan(alpha) + y_CAD * np.tan(beta)
    return [x_CAD, y_CAD, z]

# Apply the transformation
position_list = list(map(
    lambda x: z_position((x - center) * mm_scale_500x, alpha['top'], alpha['left'], z_center,phi),
    centroid_500x
))

# Convert to DataFrame
df = pd.DataFrame(position_list, columns=['x', 'y', 'z'])
df['z'] = df['z'].apply(lambda x: x[0]) # Saves as float not list

# Save to CSV (or any other format)
df.to_csv(data_directory+'csvs/500x_positions.csv', index=False)
# %%
directory = data_directory + "microscope/sample2/150/"
zip_path = os.path.join(directory, "RoiSet.zip")
centroid_150x = []
centers_150x = []
for roi in roifile.roiread(zip_path):
    if roi.name=="1mm":
        array = np.array([[roi.x1, roi.y1],[roi.x2, roi.y2]])
        mm_scale_150x = 1/np.linalg.norm(array[0]-array[1])
    
    elif '-' in roi.name:
        center_x = (roi.left + roi.right) / 2
        center_y = (roi.top + roi.bottom) / 2
        centroid_150x.append(np.array([center_x,center_y]))
    elif 'top' in roi.name or 'bottom' in roi.name or 'middle' in roi.name:
        center_x = (roi.left + roi.right) / 2
        center_y = (roi.top + roi.bottom) / 2
        centers_150x.append(
            (
                np.array(
                    [center_x,center_y]
                ),
                roi.name
            )
        )        
    else:
        print("ROI type not specifically handled.")
centers_150x= {name:val for val, name in centers_150x}
#%%
center_150x = np.mean(
    [
        centers_150x['top_left'],
        centers_150x['top_right'],
        centers_150x['middle'],
        centers_150x['bottom_right'],
        centers_150x['bottom_left']
    ], axis=0
)
#%%
'''Find Position'''
def z_position(position, alpha, beta, z_center,phi):
    x_CAD = position[0]*np.cos(phi)+position[1]*np.sin(phi)
    y_CAD = -position[0]*np.sin(phi)+position[1]*np.cos(phi)
    z = z_center + x_CAD * np.tan(alpha) + y_CAD * np.tan(beta)
    return [x_CAD, y_CAD, z]

# Apply the transformation
position_list = list(map(
    lambda x: z_position((x - center_150x) * mm_scale_150x, alpha['top'], alpha['left'], z_center,phi),
    centroid_150x
))

# Convert to DataFrame
df = pd.DataFrame(position_list, columns=['x', 'y', 'z'])
df['z'] = df['z'].apply(lambda x: x[0]) # Saves as float not list

# Save to CSV (or any other format)
df.to_csv(data_directory+'csvs/150x_positions.csv', index=False)