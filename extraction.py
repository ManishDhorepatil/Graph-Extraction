import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the input image
img_1 = cv2.imread('graph 1.png')
img = cv2.resize(img_1, (500, 500))

# Create a window to display the image
cv2.namedWindow('image')

# Define a list to store the clicked points
points = []

# Define a mouse callback function
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        if len(points) >= 6:
            # Fit a quadratic curve to the clicked points using polynomial regression
            coeffs = np.polyfit([p[0] for p in points], [p[1] for p in points], 2)

            # Generate a set of x values
            x_values = np.arange(0, img.shape[1], 1)

            # Evaluate the quadratic curve at the x values
            y_values = np.polyval(coeffs, x_values)

            # Plot the input image and the fitted curve
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.plot(x_values, y_values, color='red', linewidth=1)
            plt.show()


# Display the image and wait for user input
cv2.imshow('image', img)
cv2.setMouseCallback('image', click_event)
cv2.waitKey(0)

# Close the window
cv2.destroyAllWindows()
