import cv2
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# -----------------------------------------------------------------------------
# HARRIS CORNER FUNCTION
# -----------------------------------------------------------------------------
def detect_harris_corners(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    harris = cv2.cornerHarris(gray, 2, 3, 0.04)
    harris = cv2.dilate(harris, None)

    img[harris > 0.01 * harris.max()] = [0, 0, 255]

    # Extract corner points
    points = np.argwhere(harris > 0.01 * harris.max())
    points = np.flip(points, axis=1)
    points = np.float32(points)

    return img, points


# -----------------------------------------------------------------------------
# OPTICAL FLOW FUNCTION (Automatic from Harris points)
# -----------------------------------------------------------------------------
def apply_optical_flow(video_path, points):
    cap = cv2.VideoCapture(video_path)

    ret, old_frame = cap.read()
    if not ret:
        print("❌ Could not read the video for optical flow.")
        return

    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    lk_params = dict(winSize=(21, 21), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    mask = np.zeros_like(old_frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        new_points, status, _ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, points, None, **lk_params)

        for i, (new, old) in enumerate(zip(new_points, points)):
            if status[i]:
                x_new, y_new = new.ravel()
                x_old, y_old = old.ravel()

                mask = cv2.line(mask, (int(x_old), int(y_old)), (int(x_new), int(y_new)), (0, 255, 0), 2)
                frame = cv2.circle(frame, (int(x_new), int(y_new)), 4, (0, 0, 255), -1)

        output = cv2.add(frame, mask)
        cv2.imshow("Optical Flow Tracking", output)

        key = cv2.waitKey(10)
        if key == 27:
            break

        old_gray = frame_gray.copy()
        points = new_points.reshape(-1, 1, 2)

    cap.release()
    cv2.destroyAllWindows()


# -----------------------------------------------------------------------------
# MULTI-OBJECT MANUAL POINT SELECTION
# -----------------------------------------------------------------------------
selected_points = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_points.append((x, y))
        cv2.circle(param, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Select Points", param)


# -----------------------------------------------------------------------------
# MULTI-OBJECT TRACKING USING OPTICAL FLOW
# -----------------------------------------------------------------------------
def track_multiple_objects(video_path):
    global selected_points
    selected_points = []

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()

    if not ret:
        print("❌ Unable to read video for multi-object tracking.")
        return

    clone = frame.copy()
    cv2.imshow("Select Points", clone)
    cv2.setMouseCallback("Select Points", click_event, clone)

    print("👉 Click points on the video frame to track. Press ENTER when done.")

    while True:
        key = cv2.waitKey(1)
        if key == 13:  # ENTER key
            break

    cv2.destroyWindow("Select Points")

    if len(selected_points) == 0:
        print("⚠ No points selected. Cancelling.")
        return

    points = np.array(selected_points, dtype=np.float32).reshape(-1, 1, 2)

    old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lk_params = dict(winSize=(21, 21), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    mask = np.zeros_like(frame)

    while True:
        ret, new_frame = cap.read()
        if not ret:
            break

        new_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)

        new_points, status, _ = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, points, None, **lk_params)

        for i, (new, old) in enumerate(zip(new_points, points)):
            if status[i]:
                x_new, y_new = new.ravel()
                x_old, y_old = old.ravel()
                mask = cv2.line(mask, (int(x_old), int(y_old)), (int(x_new), int(y_new)), (255, 0, 0), 2)
                new_frame = cv2.circle(new_frame, (int(x_new), int(y_new)), 6, (0, 255, 255), -1)

        output = cv2.add(new_frame, mask)
        cv2.imshow("Multi-Object Tracking Optical Flow", output)

        old_gray = new_gray.copy()
        points = new_points.reshape(-1, 1, 2)

        if cv2.waitKey(10) & 27 == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


# -----------------------------------------------------------------------------
# MAIN PROGRAM
# -----------------------------------------------------------------------------
Tk().withdraw()

# Step 1 — Choose image/video for Harris
file1 = askopenfilename(title="Select image or video for Harris Corner Detection",
                        filetypes=[("Media files", "*.jpg *.png *.mp4 *.avi *.mov *.mkv")])

is_image = file1.lower().endswith((".jpg", ".jpeg", ".png"))
is_video = file1.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))

# Step 2 — Harris corner detection
if is_image:
    img = cv2.imread(file1)
    harris_img, corner_points = detect_harris_corners(img)
    cv2.imshow("Harris Corners (Image)", harris_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

elif is_video:
    cap = cv2.VideoCapture(file1)
    ret, frame = cap.read()
    cap.release()

    harris_img, corner_points = detect_harris_corners(frame)
    cv2.imshow("Harris Corners (Video First Frame)", harris_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Step 3 — Ask for video for automatic optical flow
file2 = askopenfilename(title="Select video for Optical Flow Tracking",
                        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")])

apply_optical_flow(file2, corner_points.reshape(-1, 1, 2))

# Step 4 — Ask for video for multi-object tracking
file3 = askopenfilename(title="Select video for MULTI-OBJECT Tracking",
                        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")])

track_multiple_objects(file3)
