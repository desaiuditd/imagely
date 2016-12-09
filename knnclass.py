from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os

def image_to_feature_vector(image, size=(64, 64)):
	# resize the image to a fixed size, then flatten the image into
	# a list of raw pixel intensities
	return cv2.resize(image, size).flatten()

def extract_color_histogram(image, bins=(8, 8, 8)):
	# extract a 3D color histogram from the HSV color space using
	# the supplied number of `bins` per channel
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
		[0, 256, 0, 256, 0, 256])

	# handle normalizing the histogram if we are using OpenCV 2.4.X
	if imutils.is_cv2():
		hist = cv2.normalize(hist)

	# otherwise, perform "in place" normalization in OpenCV 3 (I
	# personally hate the way this is done
	else:
		cv2.normalize(hist, hist)

	# return the flattened histogram as the feature vector
	return hist.flatten()


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-k", "--neighbors", type=int, default=9,
	help="# of nearest neighbors for classification")
ap.add_argument("-j", "--jobs", type=int, default=-1,
	help="# of jobs for k-NN distance (-1 uses all available cores)")
args = vars(ap.parse_args())

imagePaths = list(paths.list_images(args["dataset"]))


rawImg = []
features = []
labels = []


for (i, imagePath) in enumerate(imagePaths):
	image = cv2.imread(imagePath)
	label = imagePath.split(os.path.sep)[-1].split(".")[0]
	hist = extract_color_histogram(image)
	rawImg.append(pixels)
	features.append(hist)
	labels.append(label)
	
	if i > 0 and i % (len(imagePaths)/100) == 0:
		fin = (float(i)/len(imagePaths))*100
		print("processed {}%".format(fin))

rawImg = np.array(rawImg)
features = np.array(features)
labels = np.array(labels)
print("RAM occupied by pixels matrix: {:.2f}MB".format(
	rawImg.nbytes / (1024 * 1000.0)))
print("RAM occupied by features matrix: {:.2f}MB".format(
	features.nbytes / (1024 * 1000.0)))


(trainRI, testRI, trainRL, testRL) = train_test_split(
	rawImg, labels, test_size=0.25, random_state=42)
(trainFeat, testFeat, trainLabels, testLabels) = train_test_split(
	features, labels, test_size=0.25, random_state=42)

print("evaluating raw pixel accuracy...")
model = KNeighborsClassifier(n_neighbors=args["neighbors"],
	n_jobs=args["jobs"])
model.fit(trainRI, trainRL)
acc = model.score(testRI, testRL)
print("raw pixel accuracy: {:.2f}%".format(acc * 100))

print("evaluating histogram accuracy...")
model = KNeighborsClassifier(n_neighbors=args["neighbors"],
	n_jobs=args["jobs"])
model.fit(trainFeat, trainLabels)
acc = model.score(testFeat, testLabels)
tracc = model.score(trainFeat, trainLabels)
print("Test histogram accuracy: {:.2f}%".format(acc * 100))
print("train histogram accuracy: {:.2f}%".format(tracc * 100))