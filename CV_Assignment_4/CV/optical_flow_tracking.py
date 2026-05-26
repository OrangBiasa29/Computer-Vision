import cv2
import numpy as np

# UTILITIES
def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image, 1.0

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    elif height is None:
        r = width / float(w)
        dim = (width, int(h * r))
    else:
        r = min(width / float(w), height / float(h))
        dim = (int(w * r), int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)
    return resized, r

# LOAD VIDEO
video_path = "traffic.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Cannot open video")
    exit()

# READ FIRST FRAME
ret, first_frame = cap.read()

if not ret:
    print("ERROR: Cannot read first frame")
    exit()

first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# SELECT ROI
display_frame, scale = resize_with_aspect_ratio(
    first_frame,
    width=960,
    height=720
)

cv2.namedWindow("Select Vehicle", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Select Vehicle", display_frame.shape[1], display_frame.shape[0])

bbox_display = cv2.selectROI("Select Vehicle", display_frame, False)
cv2.destroyAllWindows()

if bbox_display == (0, 0, 0, 0):
    print("ERROR: No ROI selected")
    cap.release()
    exit()

x, y, w, h = [int(v / scale) for v in bbox_display]

print("ROI selected (original coords):", (x, y, w, h))


# INITIALIZE OPTICAL FLOW TRACKING

# Detect corners in the initial ROI
roi = first_gray[y:y+h, x:x+w]
corners = cv2.goodFeaturesToTrack(
    roi,
    maxCorners=100,
    qualityLevel=0.01,
    minDistance=10
)

if corners is None:
    print("ERROR: No corners detected in ROI")
    cap.release()
    exit()

# Adjust corner coordinates to full frame
p0 = corners.copy()
p0[:, 0, 0] += x
p0[:, 0, 1] += y

# Lucas-Kanade parameters
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

old_gray = first_gray.copy()
old_points = p0.copy()

print(f"Detected {len(old_points)} feature points")

# OPTICAL FLOW TRACKING

cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
max_frames = int(fps * 30)  # 20 seconds
frame_count = 0

cv2.namedWindow("Optical Flow Tracking", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Optical Flow Tracking", display_frame.shape[1], display_frame.shape[0])

while True:

    ret, frame = cap.read()

    if not ret:
        break
    
    frame_count += 1
    if frame_count > max_frames:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow
    new_points, status, error = cv2.calcOpticalFlowPyrLK(
        old_gray,
        gray,
        old_points,
        None,
        **lk_params
    )

    # Filter good points
    if new_points is not None and status is not None:
        good_new = new_points[status == 1]
        good_old = old_points[status == 1]

        # Draw tracking points
        for new, old in zip(good_new, good_old):
            a, b = new.ravel()
            c, d = old.ravel()
            
            cv2.circle(frame, (int(a), int(b)), 5, (0, 255, 0), -1)
            cv2.line(frame, (int(c), int(d)), (int(a), int(b)), (0, 255, 0), 2)

        # Calculate bounding box from tracked points
        if len(good_new) > 0:
            x_coords = good_new[:, 0]
            y_coords = good_new[:, 1]
            
            bbox_x = int(np.min(x_coords))
            bbox_y = int(np.min(y_coords))
            bbox_w = int(np.max(x_coords) - bbox_x)
            bbox_h = int(np.max(y_coords) - bbox_y)
            
            # Draw bounding box
            cv2.rectangle(
                frame,
                (bbox_x, bbox_y),
                (bbox_x + bbox_w, bbox_y + bbox_h),
                (0, 255, 0),
                3
            )

        # Update for next frame
        old_gray = gray.copy()
        old_points = good_new.reshape(-1, 1, 2)

    cv2.putText(
        frame,
        "Optical Flow Tracking",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )
    
    cv2.putText(
        frame,
        f"Tracked points: {len(old_points) if new_points is not None else 0}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    display_frame, _ = resize_with_aspect_ratio(frame, width=960, height=720)
    cv2.imshow("Optical Flow Tracking", display_frame)

    key = cv2.waitKey(30)

    if key == 27:
        break

    frame_count += 1

print(f"Processed {frame_count} frames")

cap.release()
cv2.destroyAllWindows()
