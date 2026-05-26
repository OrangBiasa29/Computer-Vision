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

template = first_gray[y:y+h, x:x+w]

print("ROI selected (original coords):", (x, y, w, h))

# TEMPLATE TRACKING
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
max_frames = int(fps * 30)  # 30 seconds
frame_count = 0

cv2.namedWindow("Template Tracking", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Template Tracking", display_frame.shape[1], display_frame.shape[0])

while True:

    ret, frame = cap.read()

    if not ret:
        break
    
    frame_count += 1
    if frame_count > max_frames:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    result = cv2.matchTemplate(
        gray,
        template,
        cv2.TM_CCOEFF_NORMED
    )

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    cv2.rectangle(
        frame,
        top_left,
        bottom_right,
        (0, 255, 0),
        3
    )

    cv2.putText(
        frame,
        "Template Tracking",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    display_frame, _ = resize_with_aspect_ratio(frame, width=960, height=720)
    cv2.imshow("Template Tracking", display_frame)

    key = cv2.waitKey(30)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()