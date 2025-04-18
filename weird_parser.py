# %%
import numpy as np
import matplotlib.pyplot as plt
import csv

# Path to the CSV file
csv_file = 'monochromatic_high_mag_2_Optical.csv'

def extract_monochrome_r_image(file_path):
    with open(file_path, 'r') as f:
        reader = csv.reader(f)i
        capture = False
        pixel_rows = []
        for row in reader:
            if not row:
                continue
            first_cell = row[0].strip()
            if first_cell == "Monochrome (R)":
                capture = True
                continue
            elif first_cell in {"Monochrome (G)", "Monochrome (B)"}:
                break  # Stop reading when a new monochrome section begins
            elif capture:
                # Convert the entire row to integers, skip empty cells
                pixel_row = [int(val) for val in row if val.strip().isdigit()]
                if pixel_row:
                    pixel_rows.append(pixel_row)
        return np.array(pixel_rows, dtype=np.uint8)

# Extract only Monochrome (R) data
img_array = extract_monochrome_r_image(csv_file)

# Display the image
plt.imshow(img_array, cmap='gray', vmin=0, vmax=255)
plt.title("Monochrome (R) Image")
plt.axis('off')
plt.show()
# %%
import matplotlib.image
matplotlib.image.imsave('monochrome_r_image.png', img_array, cmap='gray', vmin=0, vmax=255)
# %%
import read_roi 
x = read_roi.read_roi_zip('RoiSet.zip')
def get_leng

# %%
import scipy.optimize
import math

def equation(xm1,xm2, theta, alpha,triangle1,b):
    delta_z = abs((xm1 - xm2) * math.tan(theta)*math.cos(alpha))
    x0 = abs((xm1 - xm2) *math.cos(alpha))/2
    if triangle1:
        x = math.cos(alpha)*xm1
        if xm1>xm2:
            x0=-x0
    else:
        x = math.cos(alpha)*xm2
        if xm2>xm1:
            x0=-x0
    return math.tan(alpha)-(delta_z/2)/((b/2)+x+x0)
scipy.optimize.fsolve(equation, x0=0, args=(1,1.1, 60, alpha,True,0))
# %%
import scipy.optimize
import math
import numpy as np

def equation_to_solve(alpha, xm1, xm2, theta, triangle1, b):
    """Function whose root we want to find.
    
    Args:
        alpha: The angle we're solving for (in radians)
        xm1, xm2: Position parameters
        theta: Angle in degrees
        triangle1: Boolean flag determining which formula to use
        b: Distance parameter
        
    Returns:
        The residual value that should be zero when alpha is correct
    """
    # Convert theta from degrees to radians
    theta_rad = math.radians(theta)
    
    # Calculate delta_z and x0
    delta_z = abs((xm1 - xm2) * math.tan(theta_rad) * math.cos(alpha))
    x0 = abs((xm1 - xm2) * math.cos(alpha)) / 2
    
    # Determine x and adjust x0 based on triangle1
    if triangle1:
        x = math.cos(alpha) * xm1
        if xm1 > xm2:
            x0 = -x0
    else:
        x = math.cos(alpha) * xm2
        if xm2 > xm1:
            x0 = -x0
    
    # The equation we're trying to solve
    return math.tan(alpha) - (delta_z / 2) / ((b / 2) + x + x0)

# Set the parameters
xm1 = 1.0
xm2 = 1.1
theta = 60  # degrees
triangle1 = True
b = 0.0

# Initial guess for alpha (in radians)
initial_alpha = 0.1

# Solve for alpha
result = scipy.optimize.fsolve(
    func=lambda a: equation_to_solve(a, xm1, xm2, theta, triangle1, b),
    x0=initial_alpha
)

# Print the results
alpha_radians = result[0]
alpha_degrees = math.degrees(alpha_radians)
print(f"Alpha solution (radians): {alpha_radians}")
print(f"Alpha solution (degrees): {alpha_degrees}")

# Verify the solution
residual = equation_to_solve(alpha_radians, xm1, xm2, theta, triangle1, b)
print(f"Residual value (should be close to zero): {residual}")

# You can also try different initial guesses to verify convergence
initial_guesses = [-0.5, 0.0, 0.1, 0.5, 1.0]
for guess in initial_guesses:
    result = scipy.optimize.fsolve(
        func=lambda a: equation_to_solve(a, xm1, xm2, theta, triangle1, b),
        x0=guess
    )
    print(f"Initial guess: {guess}, Solution: {result[0]} rad, {math.degrees(result[0])} deg")