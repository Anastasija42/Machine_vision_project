import numpy as np
import cv2

rows = 6
cols = 8

square_size = 100

img_width = cols * square_size + 2 * square_size
img_height = rows * square_size + 2 * square_size

checkerboard_img = np.full((img_height, img_width), 255, dtype=np.uint8)

start_x = square_size
start_y = square_size

for i in range(rows):
    for j in range(cols):
        if (i + j) % 2 == 0:
            pass
        else:
            top_left = (j * square_size + start_x, i * square_size + start_y)
            bottom_right = (top_left[0] + square_size, top_left[1] + square_size)

            cv2.rectangle(checkerboard_img, top_left, bottom_right, 0, -1)

output_filename = f'checkerboard_{cols}x{rows}.png'
cv2.imwrite(output_filename, checkerboard_img)

print(f"Checkerboard saved as '{output_filename}'")
print(f"Image dimensions: {img_width}x{img_height} pixels")
print(f"Inner corners: ({cols-1}, {rows-1})")

cv2.imshow('Checkerboard', checkerboard_img)
cv2.waitKey(0)
cv2.destroyAllWindows()