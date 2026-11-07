from PIL import Image
import numpy as np
import math
import matplotlib.pyplot as plt

# value for undefined pixels
UNDEFINED = -1.0

# --- IMAGE SMOOTHING using GRADIENT MASK ---

gaussian_mask = np.array([
    [1, 1, 2, 2, 2, 1, 1],
    [1, 2, 2, 4, 2, 2, 1],
    [2, 2, 4, 8, 4, 2, 2],
    [2, 4, 8, 16, 8, 4, 2],
    [2, 2, 4, 8, 4, 2, 2],
    [1, 2, 2, 4, 2, 2, 1],
    [1, 1, 2, 2, 2, 1, 1]
])

mask_sum = 140

# mask pixel offsets (i = rows, j = cols)
i_offset, j_offset = gaussian_mask.shape
i_offset = i_offset // 2
j_offset = j_offset // 2


def smoothing(image_matrix):
    result = np.full((i_pixels, j_pixels), UNDEFINED, dtype=np.float64) # border pixels are UNDEFINED

    for i in range(i_offset, i_pixels - i_offset):
        for j in range(j_offset, j_pixels - j_offset):
            current_area = image_matrix[i - i_offset : i + i_offset + 1, j - j_offset : j + j_offset + 1]
            result[i][j] = np.sum(current_area * gaussian_mask) / mask_sum

    return result


# 2. --- GRADIENT MAGNITUDE using PREWITTS OPERATOR ---

gx = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
])
gy = np.array([
    [-1, -1, -1],
    [0, 0, 0],
    [1, 1, 1]
])

grad_offset = 1


def gradient_operation(image_matrix):
    grad_magnitude = np.full((i_pixels, j_pixels), UNDEFINED, dtype=np.float64)
    grad_angle     = np.full((i_pixels, j_pixels), UNDEFINED, dtype=np.float64)

    normalization_factor = 3 * math.sqrt(2)

    for i in range(grad_offset, i_pixels - grad_offset):
        for j in range(grad_offset, j_pixels - grad_offset):
            current_area = image_matrix[i - grad_offset : i + grad_offset + 1, j - grad_offset : j + grad_offset + 1]

            if np.any(current_area == UNDEFINED): # if any of the 3x3 neighbors are undefined, skip the pixel
                continue

            horizontal_grad = np.sum(current_area * gx)
            vertical_grad   = np.sum(current_area * gy)
            grad_magnitude[i, j] = math.sqrt(horizontal_grad ** 2 + vertical_grad ** 2) / normalization_factor

            # atan2 returns (-180, 180] --> normalize to [0, 360) for NMS
            grad_angle[i, j] = math.degrees(math.atan2(vertical_grad, horizontal_grad)) % 360.0

    return grad_magnitude, grad_angle


# 3. --- NON-MAXIMA SUPPRESSION ---

window_offset = 1


def non_maxima_suppression(grad_magnitude, grad_angle):
    nms_grad_magnitude = grad_magnitude.copy()

    for i in range(window_offset, i_pixels - window_offset):
        for j in range(window_offset, j_pixels - window_offset):

            # zero output if the center or any 3x3 neighbor is undefined
            if np.any(grad_magnitude[i-1:i+2, j-1:j+2] == UNDEFINED):
                nms_grad_magnitude[i, j] = 0
                continue

            a = grad_angle[i, j]

            # find which sector the angle falls into and assign the two neighbors
            if (22.5 <= a <= 67.5) or (202.5 <= a <= 247.5): # sector 1
                n1 = (i-1, j+1)
                n2 = (i+1, j-1)
            elif (67.5 <= a <= 112.5) or (247.5 <= a <= 292.5): # sector 2
                n1 = (i-1, j)
                n2 = (i+1, j)
            elif (112.5 <= a <= 157.5) or (292.5 <= a <= 337.5): # sector 3
                n1 = (i-1, j-1)
                n2 = (i+1, j+1)
            else: # sector 0
                n1 = (i, j-1)
                n2 = (i, j+1)

            if (grad_magnitude[i, j] <= grad_magnitude[n1]) or (grad_magnitude[i, j] <= grad_magnitude[n2]):
                # if the center pixel is not a local maximum set to zero
                nms_grad_magnitude[i, j] = 0

    nms_grad_magnitude[nms_grad_magnitude == UNDEFINED] = 0 # output is zero at locations with undefined magnitude values.
    return nms_grad_magnitude


# 4. --- SIMPLE THRESHOLDING to produce BINARY EDGE MAPS ---

def thresholding(nms_grad_magnitude, percentile, prefix):

    # zero magnitude pixels are excluded when calculating the percentile
    filtered_nms = nms_grad_magnitude[nms_grad_magnitude != 0]
    threshold = np.percentile(filtered_nms, percentile)

    # create binary edge map using threshold and save to file
    binary_edges = (nms_grad_magnitude >= threshold).astype(np.uint8) * 255
    Image.fromarray(binary_edges, mode='L').save(f'export-images/{prefix}_binary-edge-map-{percentile}.bmp')

    return threshold


# --- OUTPUT HELPERS ---

def save_display_image(matrix, path):
    # zero out UNDEFINED for display images and clip to [0,255]
    display = matrix.copy()
    display[display == UNDEFINED] = 0
    display = np.clip(display, 0, 255)
    Image.fromarray(display.astype(np.uint8), mode='L').save(path)


def process_image(input_path, photo_name):
    global i_pixels, j_pixels
    export_dir = 'export-images'

    input_image = Image.open(input_path).convert('L')
    image_matrix = np.array(input_image, dtype=np.float64)
    i_pixels, j_pixels = image_matrix.shape

    # 1. Gaussian smoothing
    smooth_image_matrix = smoothing(image_matrix)
    save_display_image(smooth_image_matrix, f'{export_dir}/{photo_name}_smooth-image.bmp')

    # 2. Prewitt gradient 
    grad_magnitude, grad_angle = gradient_operation(smooth_image_matrix)
    save_display_image(grad_magnitude, f'{export_dir}/{photo_name}_gradient-image.bmp')

    # 3. Non-maxima suppression
    nms_grad_magnitude = non_maxima_suppression(grad_magnitude, grad_angle)
    save_display_image(nms_grad_magnitude, f'{export_dir}/{photo_name}_nms-gradient-image.bmp')

    # 4. Binary edge maps at 20th, 45th, 70th percentiles
    for p in (20, 45, 70):
        t = thresholding(nms_grad_magnitude, p, photo_name)
        print(f'[{photo_name}] {p}th percentile threshold = {t:.4f}')

    # 5. Histogram of NMS magnitude
    plt.figure()
    plt.hist(nms_grad_magnitude[nms_grad_magnitude > 0].flatten(),bins=256, range=(0, 255))
    plt.xlabel('Gradient Magnitude')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of NMS Magnitude ({photo_name})')
    plt.savefig(f'{export_dir}/{photo_name}_histogram.png')
    plt.close()


def main():

    process_image('input-images/Barbara.bmp', 'Barbara')
    process_image('input-images/Goldhill.bmp', 'Goldhill')
    process_image('input-images/Peppers.bmp', 'Peppers')

main()