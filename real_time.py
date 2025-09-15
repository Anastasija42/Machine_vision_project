import cv2
import numpy as np


def send_data_to_robot(serial_port, object_data):
    msg = (f"{object_data['center_mm'][0]},{object_data['center_mm'][1]},"
           f"{object_data['duzina_mm']},{object_data['sirina_mm']},"
           f"{object_data['ugao']}\n")

    try:
        print(f"Slanje poruke: {msg.strip()}")
        # serial_port.write(msg.encode('utf-8'))
    except Exception as e:
        print(f"Greška pri slanju podataka: {e}")



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
                pixel_per_mm = avg_size_px / real_marker_size_mm
                return pixel_per_mm # Vraćamo izračunat odnos
    return None # Vraćamo None ako marker nije pronađen u ovom frejmu

def segment_objects(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_wood = np.array([10, 100, 120])
    upper_wood = np.array([25, 255, 255])
    mask = cv2.inRange(hsv_image, lower_wood, upper_wood)
    kernel_close = np.ones((21, 21), np.uint8)
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    contours, _ = cv2.findContours(closed_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros(closed_mask.shape, dtype=np.uint8)
    min_area_threshold = 1000
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area_threshold:
            cv2.drawContours(final_mask, [cnt], -1, (255), thickness=cv2.FILLED)
    dist_transform = cv2.distanceTransform(final_mask, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    sure_bg = cv2.dilate(final_mask, np.ones((3,3),np.uint8), iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    cv2.watershed(image, markers)
    return markers

def analyze_and_get_object_data(markers_image, original_image, px_per_mm):
    object_data_list = []
    output_image = original_image.copy()
    labels = np.unique(markers_image)
    for label in labels:
        if label <= 1: continue
        mask = np.zeros(markers_image.shape, dtype="uint8")
        mask[markers_image == label] = 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) < 500: continue
            rect = cv2.minAreaRect(c)
            (x_rect, y_rect), (width_px, height_px), angle = rect
            M = cv2.moments(c)
            cX, cY = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])) if M["m00"] != 0 else (int(x_rect), int(y_rect))
            if px_per_mm and px_per_mm > 0:
                duzina_mm = max(width_px, height_px) / px_per_mm
                sirina_mm = min(width_px, height_px) / px_per_mm
                if width_px < height_px: angle += 90
                object_data = { "center_px": (cX, cY), "center_mm": (round(cX / px_per_mm, 2), round(cY / px_per_mm, 2)), "duzina_mm": round(duzina_mm, 2), "sirina_mm": round(sirina_mm, 2), "ugao": round(angle, 2) }
                object_data_list.append(object_data)
            box = cv2.boxPoints(rect); box = np.intp(box)
            cv2.drawContours(output_image, [box], 0, (0, 255, 0), 2)
            cv2.circle(output_image, (cX, cY), 7, (255, 0, 0), -1)
            cv2.putText(output_image, f"ID:{label-1}", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return object_data_list, output_image

# --- GLAVNI PROGRAM ZA REAL-TIME RAD ---
if __name__ == "__main__":
    ARUCO_ID_TO_FIND = 23
    ARUCO_REAL_SIZE_MM = 35 # Primer! Zamenite sa vašom vrednošću!

    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Greška: Nije moguće otvoriti kameru.")
        exit()

    # Petlja koja se izvršava za svaki frejm sa kamere
    while True:
        # Čitanje jednog frejma
        ret, frame = cap.read()
        if not ret:
            print("Ne mogu da dobijem frejm. Izlazim...")
            break

        px_per_mm_ratio = get_pixel_to_mm_ratio(frame, ARUCO_ID_TO_FIND, ARUCO_REAL_SIZE_MM)

        segmented_markers = segment_objects(frame)

        if px_per_mm_ratio:
            detected_objects, output_frame = analyze_and_get_object_data(
                segmented_markers, frame, px_per_mm_ratio
            )
            if detected_objects:
                print("\033[H\033[J", end="")
                print("--- DETEKTOVANI OBJEKTI (real-time) ---")
                for i, obj in enumerate(detected_objects):
                    print(f"\nObjekat #{i+1}:")
                    print(f"  Centar (X,Y): {obj['center_mm']} mm | Ugao: {obj['ugao']} deg")

        else:
            output_frame = frame
            cv2.putText(output_frame, "ARUCO MARKER NIJE VIDLJIV!", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Real-time detekcija", output_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

