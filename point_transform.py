import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


import cv2 as cv
import argparse

fig,ax = plt.subplots()

img_pts = []
plt_image = []

# mouse callback function
def mouse_handler(event):
	if event.dblclick:
		ax.plot(event.xdata, event.ydata, markersize=10, c='red', marker='x', clip_on=False)
		img_pts.append([event.xdata, event.ydata])
		plt.show()

def order_points(pts):
	# create 4 by 2 Matrix to store the coordinates
	coordinates = np.zeros((4,2))

	# top-left will have the smallest coordinate, bottom-right will have the largest
	sums = pts.sum(axis = 1)
	coordinates[0] = pts[np.argmin(sums)]
	coordinates[2] = pts[np.argmax(sums)]

	# computing the difference
	diff = np.diff(pts, axis = 1)
	coordinates[1] = pts[np.argmin(diff)]
	coordinates[3] = pts[np.argmax(diff)]

	return coordinates

def transform(image, pts):
	rect_pts = order_points(pts)
	(tl, tr, br, bl) = rect_pts

	# now compute the width and height of the new image (pythagorean theorem)
	width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

	# subtract 1 in case the width is out of range
	max_width = max(int(width_a), int(width_b)) - 1
	max_height = max(int(height_a), int(height_b)) -1

	# using the new dimenstion, construct the new desired coords
	dst = np.array([[0, 0], [max_width, 0], [max_width, max_height], [0, max_height]])

	# compute it
	new_image = cv.getPerspectiveTransform(np.float32(rect_pts), np.float32(dst))
	# warp the image into new frame
	warped = cv.warpPerspective(image, new_image, (max_width, max_height))

	return warped

def image_pts(img):
	global plt_image
	global img_pts

	image = cv.imread(img)
	plt_image = plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))

	cid = fig.canvas.mpl_connect('button_press_event', mouse_handler)
	plt.show()

	return np.array(img_pts)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Generate Extrinsic Calibration")
    parser.add_argument('-i', '--img', required=True)
    args = parser.parse_args()
    transformed_img = transform(cv.imread(args.img),image_pts(args.img))
    cv.imshow("title", transformed_img)
    # cv.imwrite("out.png", transformed_img)
    cv.waitKey(0)

