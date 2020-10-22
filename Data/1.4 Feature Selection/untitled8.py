import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True , help="path to input directory of images")
args = vars(ap.parse_args())
image = cv2.imread(args["images"])
cv2.imshow("lallo",image)
cv2.waitKey(0)
cv2.destroyAllWindows()