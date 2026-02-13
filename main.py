import cv2, os, time
import numpy as np
from post_process_lines import post_process_lines

def post_process_lines_2(lines, slope_thresh=0.2, dist_thresh=1):
    """
    Removes horizontal lines and merges lines that are close and have similar slopes.
    
    Parameters:
        lines: np.array of shape (N,1,4) from HoughLinesP
        slope_thresh: lines with |slope| < slope_thresh are considered horizontal and removed
        dist_thresh: maximum distance between lines to consider them close for merging
    
    Returns:
        merged_lines: np.array of merged lines, shape (M,1,4)
    """
    if lines is None:
        return np.array([]).reshape(0, 1, 4)
    
    # Step 1: Filter out horizontal lines
    filtered_lines = []
    for line in lines[:, 0]:
        x1, y1, x2, y2 = line
        if x2 - x1 == 0:
            slope = np.inf
        else:
            slope = (y2 - y1) / (x2 - x1)
        if abs(slope) >= slope_thresh:  # keep non-horizontal lines
            filtered_lines.append([x1, y1, x2, y2])
    
    if len(filtered_lines) == 0:
        return np.array([]).reshape(0, 1, 4)
    
    # merge lines that have similar slope
    # calculate slope of all lines and store somewhere.
    # for every slope check if there exists any similar slope, if yes, add a new line in merged list that extends from line1 to line2 and slope is avg of both slopes

    slopes = {}
    for line in lines[:, 0]:
        x1, y1, x2, y2 = line
        slope = (y2-y1)/(x2-x1)
        slopes[slope] = (x1, y1, x2, y2)

        
    merged_lines = []
    sorted_slopes = dict(sorted(slopes.items()))
    
    for slope in slopes:
        pass



    # return np.array(merged_lines).reshape(-1, 1, 4)
    return np.array(filtered_lines).reshape(-1, 1, 4)

# Remove the horizontal lines that are detected
def post_process_lines_1(lines):
    if lines is None:
        return lines
    filtered_lines = []
    for line in lines[:, 0]:
        x1, y1, x2, y2 = line
        m = (y2-y1)/(x2-x1)
        # print(m)
        if not (m < 0.2 and m > -0.2):
            filtered_lines.append([x1,y1,x2,y2])

    return np.array(filtered_lines).reshape(-1, 1, 4)

def process_image(filename):

    img = cv2.imread("input/"+filename)
    if img is None:
        print("no image")
        return

    # 1. Converting to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Applying Gaussian Blur
    blur = cv2.GaussianBlur(gray, (5,5), 1.4)

    # 3. Canny Edge Detection
    edges = cv2.Canny(
    blur,
    threshold1=50,
    threshold2=150
    )

    # 4. Masking to Region of Interest
    masked_edges = region_selection(edges)

    # 5. Hough Transform
    lines = cv2.HoughLinesP(
    masked_edges,
    rho=1,
    theta=np.pi/180,
    threshold=150,
    minLineLength=1,
    maxLineGap=500
    )


    # Count lines
    if lines is not None:
        num_linesP = len(lines)
    else:
        num_linesP = 0

    # This removes horizontal lines (lines that are not road lanes)
    lines = post_process_lines_1(lines)

    lines, stats = post_process_lines(lines)
    # print(stats)

    # 6. Drawing Lines on top of image
    line_img = img.copy()
    h, w = img.shape[:2]
    tire_offset = 30
    centre = (w//2, h//2)
    #7. Drawing tire lines
    right_tyre_line = (w//2 + 300, h//2 + 300, w//2 + 100, h//2 + 100)
    left_tyre_line = (w//2 - 300, h//2 + 300, w//2 - 100, h//2 + 100)

    # Get line for the right and left lane
    right_line = get_leftmost_in_right_half(img, lines)
    left_line = get_rightmost_in_left_half(img, lines)
    # print("right_line" , right_line)
    # print("left_line" , left_line)
    # print("left_tyre" , left_tyre_line)
    # print("right_tyre" , right_tyre_line)

    if right_line is None or left_line is None:
        cv2.putText(line_img, "One or more Lanes Not Detected", (50,150), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 3, cv2.LINE_AA)
    else:
        if is_violation(left_tyre_line, right_tyre_line, left_line, right_line):
            # print("Violation Happening-------------------------")
            cv2.putText(line_img, "Lane Violation", (50,200), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 3, cv2.LINE_AA)
        else:
            cv2.putText(line_img, "No Lane Violation", (50,200), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 3, cv2.LINE_AA)


        cv2.putText(line_img, "Lanes Detected", (50,150), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 3, cv2.LINE_AA)

    # Printing Tyre Lines
    cv2.line(line_img, (int(w//2 + 300), int(h//2 + 300) ), (int(w//2 + 100), int(h//2 +100)), (0, 0, 0), 2)
    cv2.line(line_img, (int(w//2 - 300), int(h//2 + 300) ), (int(w//2 - 100), int(h//2 +100)), (0, 0, 0), 2)
    
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            cv2.line(line_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)


    # 7. Saving image
    cv2.imwrite(f"output/{filename}_output.jpg", line_img)
    # cv2.imwrite("edges.jpg", edges)
    # cv2.imwrite("hough_lines.jpg", line_img)

    # 8. Displaying Image
    # cv2.imshow("Edges", edges)
    # cv2.imshow("Hough Lines", line_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def is_violation(left_tyre_line, right_tyre_line, left_line, right_line):
    if ((left_line[0] + left_line[2]) / 2) > ((left_tyre_line[0] + left_tyre_line[2]) /2):
        return True
    if ((right_line[0] + right_line[2]) / 2) < ((right_tyre_line[0] + right_tyre_line[2]) /2):
        return True
    return False

def get_leftmost_in_right_half(img, lines):
    if lines is None:
        return None
    
    h, w = img.shape[:2]
    center_x = w // 2

    right_lines = [
        L for L in lines[:, 0]
        if ((L[0] + L[2]) / 2) > center_x
    ]

    if not right_lines:
        return None
    return min(right_lines, key = lambda L: ((L[0] + L[2]) / 2))


def get_rightmost_in_left_half(img, lines):
    if lines is None:
        return None
    
    h, w = img.shape[:2]
    center_x = w // 2

    left_lines = [
        L for L in lines[:, 0]
        if ((L[0] + L[2]) / 2) < center_x
    ]

    if not left_lines:
        return None
    return max(left_lines, key = lambda L: ((L[0] + L[2]) / 2))

def side_of_line(A, B, P):
    # This function determines which side of a line, a point lies on
    # If returned value = 0 point is on line, if ret value > 0 , point lies on left of line, if ret value < 0, point lies on right
    return (B[0]-A[0])*(P[1]-A[1]) - (B[1]-A[1])*(P[0]-A[0])

def region_selection(img):
    h, w = img.shape[:2]
    mask = np.zeros_like(img)
    mask[h//2:h, :] = 255

    return cv2.bitwise_and(img, mask)




# process_image('sample.jpg')

i = 0
start_time = time.perf_counter()
no_of_files = len(os.listdir('input'))
for filename in os.listdir('input'):
    # if i == 5:
    #     break
    # i+=1
        # print("---------------- Processing File ----------------")
    process_image(filename)
    
end_time = time.perf_counter()
print(f"Processed {no_of_files} in {end_time-start_time} seconds")
print(f"Rate: {no_of_files/(end_time-start_time)} files per second")
