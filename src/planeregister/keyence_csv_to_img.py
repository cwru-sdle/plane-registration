# %%
import numpy as np
import matplotlib.pyplot as plt
import csv
import matplotlib.image

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
if __name__ == "__main__":
    matplotlib.image.imsave('monochrome_r_image.png', img_array, cmap='gray', vmin=0, vmax=255)