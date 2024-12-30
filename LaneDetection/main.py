import cv2
import numpy as np
from cv2 import cvtColor

cam = cv2.VideoCapture('Lane Detection Test Video 01 1.mp4')

width = int(1920/3)
height = int(1080/4)

left_top = 0
right_top = 0
left_bottom = 0
right_bottom = 0

while True:
    ret, frame = cam.read()
    # ret (bool): return code of the 'read' operation
    # frame (array): The actual frame as an array
    #               Height x Width x 3 (3 colors, BGR) if color image
    #               Height x Width if Grayscale

    if ret is False:
        break
#2) Shrink the frame
    frame =  cv2.resize(frame, (width, height))
    main_frame = np.copy(frame)


#3) Grayscale
    # My method
    # new_frame = np.zeros((height,width), dtype = np.uint8)
    #
    # for i in range(0, frame.shape[0]):         # frame.shape returns a tuple of (height, width)
    #     for j in range(0, frame.shape[1]):
    #         B, G, R = frame[i,j]
    #         GRAYSCALE = R * 0.3 + G * 0.59 + B * 0.11 # NTSC (National Television System Committee) formula
    #         new_frame[i,j] = GRAYSCALE
    #
    # frame = new_frame

    frame = cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('Original', frame)

#4) Trapezoid

    trapezoid_frame = np.zeros((height, width), dtype = np.uint8)

    #trapezoid points - where we draw it
    upper_left = (int(width*0.45), int(height * 0.77))
    upper_right = (width - int(width*0.45), int(height * 0.77))
    lower_left = (0,height)
    lower_right = (width,height)

    trapezoidPoints = np.array([upper_right, upper_left, lower_left, lower_right], dtype = np.int32)
                                    # points in trigonometrical order, because that's how fillConvexPoly works

    cv2.fillConvexPoly(trapezoid_frame, trapezoidPoints, 1)
    # cv2.imshow('Trapezoid', black_frame * 255)

    frame = trapezoid_frame * frame # crop the trapezoid from the frame
                                    # this is element-wise multiplication, that is element by element

    cv2.imshow('Cropped', frame)

#5) Top-down view

    #the extremities of the window
    screen_upper_right = (width, 0)
    screen_upper_left = (0, 0)
    screen_lower_left = (0, height)
    screen_lower_right = (width, height)

    #points in trigonometrical order
    screenPoints = np.array([screen_upper_right, screen_upper_left, screen_lower_left, screen_lower_right],
                            dtype = np.float32)

    trapezoidPoints = np.float32(trapezoidPoints) #convert to float32

    # get the matrix to stretch the frame to the screen points
    stretch_matrix = cv2.getPerspectiveTransform(trapezoidPoints, screenPoints)
        # returns a transformation matrix that maps every point (x,y) from
        # the original image to a new point (x', y') from the transformed image

    #apply the transformation to stretch the image
    frame = cv2.warpPerspective(frame, stretch_matrix, dsize=(width, height))
        # the transformation is applied for every point from the original image

    cv2.imshow('Top down view', frame)


#6) Blur - used to differentiate between the lane markings and not the lane markings
    # To blur an image - make every pixel to be the average of its neighbors
    # more blur - more neighbors and less blur - fewer neighbors

    # The blur help to mark the outlines (more uniform)  and to reduce unnecessary details (ex: shadows)

    frame = cv2.blur(frame, ksize=(5,5))
        # ksize = the size of the area between the pixel
        # 5x5 -> 2 neighbors on every direction, 24 in total
    '''
        1   1   1   1   1
        1   1   1   1   1
        1   1   p   1   1
        1   1   1   1   1
        1   1   1   1   1
        
        p - the pixel we compute
    '''

    cv2.imshow('Blurred', frame)


#7) Edge detection
    '''
    Sobel filter is a filtering method used to detect edges within an image by identifying and highlighting
    coarse changes in pixel intensity based on the 1st derivative. 
    '''

    sobel_vertical = np.float32([[-1, -2, -1],
                                 [0, 0, 0],
                                 [1, 2, 1]])

    sobel_horizontal = np.transpose(sobel_vertical)
    '''
    -1 0 1
    -2 0 2
    -1 0 1
    '''

    frame = np.float32(frame)

    filter_matrix_vertical = cv2.filter2D(frame, -1, kernel=sobel_vertical)
    filter_matrix_horizontal = cv2.filter2D(frame, -1, kernel=sobel_horizontal)

    final_matrix = np.sqrt((filter_matrix_vertical ** 2) + (filter_matrix_horizontal ** 2)) # it is done element by element

    frame = cv2.convertScaleAbs(final_matrix) # Scales, computes absolute values and converts the result to 8-bit

    cv2.imshow('Edge detection', frame)

#8) Binarize the frames

    # Method 1
    # for rows in range(0,frame.shape[0]):
    #     for cols in range(0,frame.shape[1]):
    #         if(frame[rows][cols] < int(255/2)):
    #             frame[rows,cols] = 0
    #         else:
    #             frame[rows,cols] = 255

    # Method 2
    # def binarize(n):
    #     threshold = 80
    #     if n > threshold:
    #         return 255
    #     else:
    #         return 0
    #
    # vectorized_binarize = np.vectorize(binarize)

    # frame = vectorized_binarize(frame)
    #
    # frame = np.uint8(frame)

    _, frame = cv2.threshold(frame,  80, 255, cv2.THRESH_BINARY)

    cv2.imshow('Binarized', frame)

#9)  coordinates of street markings on each side of the road

    # frameCopy = np.copy(frame)

    # blackout first and last 5%
    frame[:,0:int(width * 0.05)] = 0
    frame[:,width - int(width * 0.05):] = 0

    cv2.imshow("Blackout first and last 5% of cols", frame)

    whiteDotsCoordsLeft = np.argwhere(frame[:,0:int(width/2)]) # return coords in form of (y,x)
    whiteDotsCoordsRight = np.argwhere(frame[:,int(width/2)+1:])
    # argwhere -> returns an array where every row contains the coordinates of a
    # position in the initial matrix where the corresponding element satisfied our condition (x>1)

    x_coords_left = whiteDotsCoordsLeft[:, 1]
    y_coords_left = whiteDotsCoordsLeft[:, 0]

    x_coords_right = int(width/2) + whiteDotsCoordsRight[:, 1] # + width/2 because argwhere treats frameCopy[:,int(width/2)+1:] from index 0
    y_coords_right = whiteDotsCoordsRight[:, 0]

#10) Find the lines that detect the edges of the lane
    '''
    The function gives us the line (a polynomial of degree 1) that best passes 
    through the points determined by x_list and y_list.
    '''
    right_side = np.polynomial.polynomial.polyfit(x_coords_right, y_coords_right, deg=1)
    left_side = np.polynomial.polynomial.polyfit(x_coords_left, y_coords_left, deg=1)

    # Coordinates of the lines
    left_top_y = 0
    left_top_x = (left_top_y - left_side[0])/left_side[1]

    left_bottom_y = height-1
    left_bottom_x = (left_bottom_y - left_side[0])/left_side[1]

    right_top_y = 0
    right_top_x =  (right_top_y - right_side[0])/right_side[1]

    right_bottom_y = height-1
    right_bottom_x = (right_bottom_y - right_side[0])/right_side[1]

    if -(10**8) <= left_top_x <= 10**8:
        if -(10**8) <= left_bottom_x <= 10**8:
            if -(10**8) <= right_top_x <= 10**8:
                if -(10**8) <= right_bottom_x <= 10**8:
                    left_top = (int(left_top_x), int(left_top_y))
                    right_top = (int(right_top_x), int(right_top_y))
                    left_bottom = (int(left_bottom_x), int(left_bottom_y))
                    right_bottom = (int(right_bottom_x), int(right_bottom_y))

    frame = cv2.line(frame, left_top, left_bottom, (200, 0, 0), 5)
    frame = cv2.line(frame, right_top, right_bottom, (100, 0, 0), 5)


    cv2.imshow("Draw lines", frame)


#11) Final visualization

    #a) blank frame
    empty_frame_left = np.zeros((height, width), dtype=np.uint8)

    #b) draw the line
    empty_frame_left = cv2.line(empty_frame_left, left_top, left_bottom, (255, 0, 0), 3)

    #c) map the top-down to the screen points (the opposite of #5)
    stretch_matrix = cv2.getPerspectiveTransform(screenPoints, trapezoidPoints)

    #d) warp the perspective
    empty_frame_left = cv2.warpPerspective(empty_frame_left, stretch_matrix, dsize=(width, height))

    #e) get the coordinates of the white pixels for the original image
    left = np.argwhere(empty_frame_left)

    # cv2.imshow("Final visualization LEFT", empty_frame_left)

    empty_frame_right = np.zeros((height, width), dtype=np.uint8)

    empty_frame_right = cv2.line(empty_frame_right, right_top, right_bottom, (255, 0, 0), 3)

    empty_frame_right = cv2.warpPerspective(empty_frame_right, stretch_matrix, dsize=(width, height))

    right = np.argwhere(empty_frame_right)

    # cv2.imshow("Final visualization RIGHT", empty_frame_right)

    #color the lines
    main_frame[left[:,0], left[:,1]] = (50, 50, 250)
    main_frame[right[:, 0], right[:, 1]] = (250, 0, 250)

    cv2.imshow("Final visualization COLOR", main_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): # bitwise AND to make sure we get just the ASCII code of the key pressed
        break

cam.release()
cv2.destroyAllWindows()