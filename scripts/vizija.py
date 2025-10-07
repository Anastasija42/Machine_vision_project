import cv2
import numpy as np
import itertools
import time

ARUCO_ID_TO_FIND = 23
ARUCO_REAL_SIZE_MM = 35.0

TARGET_WIDTH_PX = 190
WIDTH_TOLERANCE = 0.2
MINIMUM_LENGTH_RATIO = 5.0
IOU_THRESHOLD = 0.3

LOWER_WOOD = np.array([8, 80, 50])
UPPER_WOOD = np.array([30, 255, 255])

def get_pixel_to_mm_ratio(image, marker_id, real_marker_size_mm):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    corners, ids, rejected = detector.detectMarkers(image)
    if ids is not None:
        for i, current_id in enumerate(ids):
            if current_id == marker_id:
                marker_corners = corners[i][0]
                width_px = np.linalg.norm(marker_corners[0] - marker_corners[1])
                height_px = np.linalg.norm(marker_corners[1] - marker_corners[2])
                avg_size_px = (width_px + height_px) / 2.0
                return avg_size_px / real_marker_size_mm
    return None

def create_precise_mask(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    color_mask = cv2.inRange(hsv_image, LOWER_WOOD, UPPER_WOOD)
    kernel_open = np.ones((9, 9), np.uint8)
    mask_no_noise = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    contours, _ = cv2.findContours(mask_no_noise, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(mask_no_noise)
    for cnt in contours:
        if cv2.contourArea(cnt) > 500:
            cv2.drawContours(final_mask, [cnt], -1, 255, thickness=cv2.FILLED)
    return final_mask

def get_corners_from_mask(mask, epsilon_factor=0.01):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None, None
    main_contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(main_contour, True)
    approx_corners = cv2.approxPolyDP(main_contour, perimeter * epsilon_factor, True)
    return approx_corners, main_contour

def is_rectangle(points, tolerance=0.1):
    if len(points) != 4: return False
    distances = sorted([np.linalg.norm(p1 - p2) for p1, p2 in itertools.combinations(points.reshape(-1, 2), 2)])
    diagonal1, diagonal2 = distances[-1], distances[-2]
    return abs(diagonal1 - diagonal2) < (diagonal1 * tolerance)

def is_filled(rect_points, original_mask, threshold=0.85):
    mask_rect = np.zeros_like(original_mask)
    cv2.fillPoly(mask_rect, [np.int32(rect_points)], 255)
    area_rect = np.count_nonzero(mask_rect)
    if area_rect == 0: return False
    intersection = cv2.bitwise_and(original_mask, mask_rect)
    return (np.count_nonzero(intersection) / float(area_rect)) > threshold

def are_angles_parallel(angle1, angle2, tolerance=10.0):
    diff = abs(angle1 - angle2) % 90
    return diff < tolerance or abs(diff - 90) < tolerance

def check_dimensions_and_modularity(points, target_width, width_tolerance, min_ratio):
    _, (w, h), _ = cv2.minAreaRect(points)
    side_a, side_b = sorted((w, h))
    ratio_a = side_a / target_width
    is_modular = abs(ratio_a - round(ratio_a)) < width_tolerance
    if not is_modular: return False
    if round(ratio_a) == 1 and (side_b / side_a) < min_ratio: return False
    return True

def calculate_iou(rect1_pts, rect2_pts, shape):
    mask1 = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask1, [np.int32(rect1_pts)], 255)
    mask2 = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask2, [np.int32(rect2_pts)], 255)
    intersection = np.count_nonzero(cv2.bitwise_and(mask1, mask2))
    union = np.count_nonzero(cv2.bitwise_or(mask1, mask2))
    return intersection / float(union) if union > 0 else 0

def remove_overlaps(rectangles, iou_threshold, shape):
    if not rectangles: return []
    rectangles.sort(key=lambda r: cv2.contourArea(np.array(r)), reverse=True)
    final_rects = []
    for rect in rectangles:
        has_overlap = any(calculate_iou(rect, final_rect, shape) > iou_threshold for final_rect in final_rects)
        if not has_overlap:
            final_rects.append(rect)
    return final_rects

def split_rectangle(rect_points, n_splits):
    if n_splits <= 1: return [rect_points]
    center, (w, h), angle = cv2.minAreaRect(rect_points)
    is_width_longer = w > h
    long_dim, short_dim = (w, h) if is_width_longer else (h, w)
    new_long_dim = long_dim / n_splits

    angle_rad = np.deg2rad(angle)
    vec_w = np.array([np.cos(angle_rad), np.sin(angle_rad)])
    vec_h = np.array([-np.sin(angle_rad), np.cos(angle_rad)])
    split_axis_vector = vec_w if is_width_longer else vec_h

    sub_rects = []
    start_offset = -long_dim / 2 + new_long_dim / 2
    for i in range(n_splits):
        offset = start_offset + i * new_long_dim
        new_center = np.array(center) + offset * split_axis_vector
        new_size = (new_long_dim, short_dim) if is_width_longer else (short_dim, new_long_dim)
        sub_rects.append(np.int0(cv2.boxPoints((tuple(new_center), new_size, angle))))
    return sub_rects

def analyze_and_visualize_shapes(image, shapes, px_per_mm):
    """Analizira finalne oblike, konvertuje u mm i iscrtava na sliku."""
    object_data_list = []
    overlay = image.copy()
    colors = [(0, 255, 0), (255, 255, 0), (0, 165, 255), (255, 0, 255)]

    for i, shape in enumerate(shapes):
        cv2.fillPoly(overlay, [np.int32(shape)], colors[i % len(colors)])

        rect = cv2.minAreaRect(shape)
        (x_px, y_px), (width_px, height_px), angle = rect

        length_mm = max(width_px, height_px) / px_per_mm
        width_mm = min(width_px, height_px) / px_per_mm

        if width_px < height_px:
            angle += 90

        object_data = {
            "center_mm": (x_px / px_per_mm, y_px / px_per_mm),
            "duzina_mm": length_mm,
            "sirina_mm": width_mm,
            "ugao": angle
        }
        object_data_list.append(object_data)

    alpha = 0.5
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    for i, shape in enumerate(shapes):
        cv2.polylines(image, [np.int32(shape)], True, (255, 255, 255), 2)

        obj_data = object_data_list[i]
        center_px = tuple(np.int0(cv2.minAreaRect(shape)[0]))
        info_text = f"L:{obj_data['duzina_mm']:.1f} W:{obj_data['sirina_mm']:.1f} A:{obj_data['ugao']:.1f}"
        cv2.putText(image, info_text, (center_px[0] - 80, center_px[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return object_data_list, image

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Greška: Nije moguće otvoriti kameru.")
        return

    print("Pokrećem kombinovanu real-time analizu... Pritisni 'q' za izlaz.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        output_image = frame.copy()

        px_per_mm = get_pixel_to_mm_ratio(frame, ARUCO_ID_TO_FIND, ARUCO_REAL_SIZE_MM)

        if px_per_mm:
            cv2.putText(output_image, f"Ratio: {px_per_mm:.2f} px/mm", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            sample_mask = create_precise_mask(frame)

            main_corners, main_contour = get_corners_from_mask(sample_mask)

            decomposed_shapes = []
            if main_corners is not None and len(main_corners) >= 4:
                all_corners = main_corners.reshape(-1, 2)
                _, _, main_angle = cv2.minAreaRect(main_contour)

                candidate_rects = []
                if len(all_corners) <= 15:
                    for points in itertools.combinations(all_corners, 4):
                        quad = np.array(points)
                        if (is_rectangle(quad) and
                            is_filled(quad, sample_mask) and
                            are_angles_parallel(main_angle, cv2.minAreaRect(quad)[2]) and
                            check_dimensions_and_modularity(quad, TARGET_WIDTH_PX, WIDTH_TOLERANCE, MINIMUM_LENGTH_RATIO)):
                            candidate_rects.append(quad)

                final_rects = remove_overlaps(candidate_rects, IOU_THRESHOLD, sample_mask.shape)

                for shape in final_rects:
                    _, (w, h), _ = cv2.minAreaRect(shape)
                    width_multiple = round(min(w, h) / TARGET_WIDTH_PX)
                    if width_multiple > 1:
                        decomposed_shapes.extend(split_rectangle(shape, int(width_multiple)))
                    else:
                        decomposed_shapes.append(shape)

            detected_objects, output_image = analyze_and_visualize_shapes(output_image, decomposed_shapes, px_per_mm)

            if detected_objects:
                print("\033[H\033[J", end="")
                print("--- DETEKTOVANI OBJEKTI ---")
                for i, obj in enumerate(detected_objects):
                    print(f"Objekat #{i+1}: Centar(X,Y): ({obj['center_mm'][0]:.1f}, {obj['center_mm'][1]:.1f}) mm | "
                          f"L:{obj['duzina_mm']:.1f}mm | W:{obj['sirina_mm']:.1f}mm | Ugao:{obj['ugao']:.1f} deg")
        else:
            cv2.putText(output_image, "ARUCO MARKER NIJE VIDLJIV!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Real-time Full Analysis", output_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()