import cv2
import numpy as np

# Usage
def solution(image_path):
    image= cv2.imread(image_path)
    ######################################################################
    ######################################################################
    #####  WRITE YOUR CODE BELOW THIS LINE ###############################

    padding=[10,10,10,10]

    # Get the dimensions of the original image
    # height, width, channels = image.shape
    height=image.shape[0]
    width=image.shape[1]
    number_channels=image.shape[2]

    # Calculate the new dimensions including padding
    # new_height = height + top_padding + bottom_padding
    # new_width = width + left_padding + right_padding
    new_height = height + padding[0]+padding[1]
    new_width = width + padding[2] + padding[3]

    # Create a blank canvas with the new dimensions and the same number of channels
    # canvas = np.zeros((new_height, new_width, channels), dtype=np.uint8)
    canvas = np.zeros((new_height, new_width, number_channels), dtype=np.uint8)

    # Paste the original image onto the canvas at the desired position (e.g., with padding)
    canvas[padding[0]:padding[0]+height, padding[2]:padding[2]+width, :] = image
    # plt.imshow(canvas)
    # plt.show()
    image=canvas
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # print(gray.shape)
    # plt.imshow(gray,cmap="gray")
    # plt.imshow(image)
    # plt.show()
    # plt.show()
    # Apply binary thresholding to isolate the quadrilateral
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    # Find contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(contours)
    for contour in contours:
    # Approximate the contour to a polygon with fewer vertices (a quadrilateral)
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        # If the approximation has exactly 4 vertices (quadrilateral)
        if len(approx) == 4:
            # Store the vertices of the quadrilateral
            quadrilateral_vertices = approx
            sorted_vertices = sorted(approx, key=lambda x: x[0][0] + x[0][1])
            approx=sorted_vertices
    if len(quadrilateral_vertices) == 4:
        # print("Coordinates of the quadrilateral vertices:")
        # print(quadrilateral_vertices)
        newmatrix=quadrilateral_vertices[:, 0]
        # print(newmatrix)
        finalInput=[[1,1],[1,1],[1,1],[1,1]]
        B=sorted(newmatrix, key=lambda x: (x[0],x[1]))
        C=sorted(newmatrix,key=lambda x: (x[1],x[0]))
        if B[0][0]==B[1][0]:
            finalInput[0]=B[0]
            finalInput[1]=B[1]
            if B[2][1]<B[3][1]:
                finalInput[3]=B[2]
                finalInput[2]=B[3]
            else:
                finalInput[2]=B[2]
                finalInput[3]=B[3]
        elif B[2][0]==B[3][0]:
            finalInput[3]=B[2]
            finalInput[2]=B[3]
            if B[0][1]<B[1][1]:
                finalInput[0]=B[0]
                finalInput[1]=B[1]
            else:
                finalInput[1]=B[0]
                finalInput[0]=B[1]
        elif C[0][1]==C[1][1]:
            finalInput[0]=C[0]
            finalInput[3]=C[1]
            if C[2][0]<C[3][0]:
                finalInput[1]=C[2]
                finalInput[2]=C[3]
            else:
                finalInput[2]=C[2]
                finalInput[1]=C[3]
        elif C[2][1]==C[3][1]:
            finalInput[1]=C[2]
            finalInput[2]=C[3]
            if C[0][0]<C[1][0]:
                finalInput[0]=C[0]
                finalInput[3]=C[1]
            else:
                finalInput[3]=C[0]
                finalInput[0]=C[1]
        elif isequal(B[0][0],B[1][0]):
            finalInput[0]=B[0]
            finalInput[1]=B[1]
            if B[2][1]<B[3][1]:
                finalInput[3]=B[2]
                finalInput[2]=B[3]
            else:
                finalInput[2]=B[2]
                finalInput[3]=B[3]
        elif isequal(B[2][0],B[3][0]):
            finalInput[3]=B[2]
            finalInput[2]=B[3]
            if B[0][1]<B[1][1]:
                finalInput[0]=B[0]
                finalInput[1]=B[1]
            else:
                finalInput[1]=B[0]
                finalInput[0]=B[1]
        elif isequal(C[0][1],C[1][1]):
            finalInput[0]=C[0]
            finalInput[3]=C[1]
            if C[2][0]<C[3][0]:
                finalInput[1]=C[2]
                finalInput[2]=C[3]
            else:
                finalInput[2]=C[2]
                finalInput[1]=C[3]
        else :
            finalInput[1]=C[2]
            finalInput[2]=C[3]
            if C[0][0]<C[1][0]:
                finalInput[0]=C[0]
                finalInput[3]=C[1]
            else:
                finalInput[3]=C[0]
                finalInput[0]=C[1]
        output_points=[[0,0],[0,599],[599,599],[599,0]]
        # print(finalInput)
        M = cv2.getPerspectiveTransform(np.float32(finalInput),np.float32(output_points))
        out = cv2.warpPerspective(image,M,(600, 600),flags=cv2.INTER_CUBIC)
        # plt.imshow(image)
        # plt.show()
        # plt.imshow(out)
        # plt.show()
        # out=cv2.cvtColor(out,cv2.COLOR_BGR2RGB)
        # out=image
        # plt.imshow(out)
        # plt.show()
        image=out
    else:
        image= cv2.imread(image_path)
        new_size=(600,600)
        resized_image = cv2.resize(image, new_size)
        return resized_image









    ######################################################################

    return image
def isequal(a,b):
    if abs(a-b)<=10:
        return True
    else:
        return False