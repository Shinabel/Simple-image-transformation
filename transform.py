import numpy
import cv2
import argparse

def order_points(pts):
	# create 4 by 2 Matrix to store the coordinates
	coordinates = numpy.zeros((4,2))

	# top-left will have the smallest coordinate, bottom-right will have the largest
	sums = pts.sum(axis = 1)
	coordinates[0] = pts[numpy.argmin(sums)]
	coordinates[2] = pts[numpy.argmax(sums)]

	# computing the difference
	diff = numpy.diff(pts, axis = 1)
	coordinates[1] = pts[numpy.argmin(diff)]
	coordinates[3] = pts[numpy.argmax(diff)]

	return coordinates


def transform(image, pts):
	rect_pts = order_points(pts)
	(tl, tr, br, bl) = rect_pts

	# now compute the width and height of the new image (pythagorean theorem)
	width_a = numpy.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	width_b = numpy.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	height_a = numpy.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	height_b = numpy.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

	# subtract 1 in case the width is out of range
	max_width = max(int(width_a), int(width_b)) - 1
	max_height = max(int(height_a), int(height_b)) -1

	# using the new dimenstion, construct the new desired coords
	dst = numpy.array([[0, 0], [max_width, 0], [max_width, max_height], [0, max_height]])

	# compute it
	new_image = cv2.getPerspectiveTransform(numpy.float32(rect_pts), numpy.float32(dst))
	# warp the image into new frame
	warped = cv2.warpPerspective(image, new_image, (max_width, max_height))

	return warped

def main(args):
	image = cv2.imread(args.image)
	pts = numpy.array(eval(args.coords))

	new_image = transform(image, pts)

	cv2.imshow("Original", image)
	cv2.imshow("Transformed", new_image)
	cv2.waitKey(0)
	

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-image',
        help='path to the image file'
        )
    parser.add_argument(
        '-coords',
        help='list of coordinations'
        )

    args = parser.parse_args()
    main(args)