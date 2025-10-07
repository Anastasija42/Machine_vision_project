import numpy as np
import cv2

camera_matrix = np.array([[1.03349288e+03, 0.0, 1.01793889e+03],
                          [0.0, 1.03091699e+03, 5.51814421e+02],
                          [0.0, 0.0, 1.0]], dtype=np.float32)

dist_coeffs = np.array([[-0.10050491, 0.05342321, 0.00650947, -0.00234044, -0.0294389]], dtype=np.float32)


world_pts = np.array([
    [0, 0, 0],              # Point 1: Origin of the board
    [175, 0, 0],            # Point 2: Corner along the X-axis (7 squares * 25mm/square)
    [0, 125, 0],            # Point 3: Corner along the Y-axis (5 squares * 25mm/square)
    [175, 125, 0]           # Point 4: Opposite corner
], dtype=np.float32)


image_pts = np.array([
    [458, 292],             # Pixel coords for World Point 1
    [850, 305],             # Pixel coords for World Point 2
    [328, 514],             # Pixel coords for World Point 3
    [711, 532]              # Pixel coords for World Point 4
], dtype=np.float32)


world_pts_2d = world_pts[:, :2]
homography_matrix, _ = cv2.findHomography(image_pts, world_pts_2d)

print("--- Homography Matrix ---")
print(homography_matrix)


def pixel_to_world(pixel_coord):
    """
    Transforms a pixel coordinate to a real-world coordinate using the homography matrix.
    """
    pixel_homogeneous = np.array([[pixel_coord[0]], [pixel_coord[1]], [1]])
    world_homogeneous = np.dot(homography_matrix, pixel_homogeneous)
    scale_factor = world_homogeneous[2, 0]
    world_x = world_homogeneous[0, 0] / scale_factor
    world_y = world_homogeneous[1, 0] / scale_factor

    return world_x, world_y

test_pixel = (600, 420)
real_world_position = pixel_to_world(test_pixel)

print(f"\nPixel coordinate {test_pixel} corresponds to real-world position:")
print(f"X = {real_world_position[0]:.2f} mm")
print(f"Y = {real_world_position[1]:.2f} mm (on the Z=0 plane)")
