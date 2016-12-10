
from __future__ import print_function
from PIL import Image as IMAGE
import random
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves import cPickle as pickle

train_filename = 'NewData'
num_classes = 5
np.random.seed(133)


def get_all_file_names(root, force=False):
    data_folders = [
        os.path.join(root, d) for d in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, d))]
    if len(data_folders) != num_classes:
        raise Exception(
            'Expected %d folders, one per class. Found %d instead.' % (
                num_classes, len(data_folders)))
    print(data_folders)
    return data_folders

train_folders = get_all_file_names(train_filename)

def rename_files(train_folders):
    for folder in train_folders:
        image_files = os.listdir(folder)
        i = 0
        type = folder.split("/")[1]
        print (type)
        for image in image_files:
            i = i + 1
            image_name = folder+'/'+image
            command = "cp " + folder + "/"+ image +" NewData/"+ type + "/" + type + str(i) + ".png"
            # print(command)
            # print(image_name)
            if(i % 100 == 0):
                print (i)
            os.system(command)

rename_files(train_folders)


def change_to_grayscale(train_folders):
    for folder in train_folders:
        image_files = os.listdir(folder)
        i = 0
        type = folder.split("/")[1]
        print (type)
        for image in image_files:
            try:
                i = i + 1
                if (i % 100 == 0):
                    print(i)
                # os.system(command)
                image_file = os.path.join(folder, image)
                img = IMAGE.open(fp=image_file).convert('LA')
                img.save("FinalData1/" + type + "/" + type + str(i) + ".png")
            except IOError as e:
                print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
            except e :
                k = e

change_to_grayscale(train_folders)

from resizeimage import resizeimage
def resize_image(train_folders):
    for folder in train_folders:
        print (folder)
        image_files = os.listdir(folder)
        i = 0
        j = 0
        type = folder.split("/")[1]
        print (type)
        print (len(image_files))
        for image in image_files:
            try:
                i = i + 1
                image_name = folder + '/' + image
                if (i % 1000 == 0):
                    print(i)
                image_file = os.path.join(folder, image)
                image_path = "NewData/" + type + "/" + type + str(i) + ".png"
                with open(image_name, 'r+b') as f:
                    with IMAGE.open(f) as image:
                        cover = resizeimage.resize_cover(image, [200, 100])
                        cover.save(image_path)
            except IOError as e:
                k = e
            except:
                print (image_path)

resize_image(train_folders)

# def make_test_data(train_folders, count):
#     for folder in train_folders:
#         image_files = os.listdir(folder)
#         i = 0
#         type = folder.split("/")[1]
#         print (type)
#         for image in image_files:
#             image_name = folder + '/' + image
#             command = "mv " + folder + "/" + image + " TestData/" + type + "/" + type + str(i) + ".png"
#             if(i % 100 == 0) :
#                 print (i)
#             # print(command)
#             # print(image_name)
#             if (i < 3000):
#                 i = i + 1
#                 os.system(command)
#             else :
#                 break
#
# make_test_data(train_folders, 3000)
#
# test_filename = "TestData"
# test_folders = get_all_file_names(train_filename)

image_size = 255
pixel_depth = 255.0

def load_image_dataset(folder):
    image_files = os.listdir(folder)
    dataset = np.ndarray(shape=(len(image_files), image_size, image_size), dtype=np.float32)
    print(folder)
    num_images = 0
    for image in image_files:
        image_file = os.path.join(folder, image)
        try:
            image_data = (ndimage.imread(image_file).astype(float) -
                          pixel_depth / 2) / pixel_depth
            c = image_data.reshape(image_size, image_size)
            if c.shape != (image_size, image_size ):
                command = "rm " + image_file
                os.system(command)
                continue
            dataset[num_images, :, :] = c
            num_images = num_images + 1
        except IOError as e:
            print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

    dataset = dataset[0:num_images, :, :]
    return dataset


def make_pickle_files(data_folders):
    dataset_names = []
    for folder in data_folders:
        set_filename = folder + '.pickle'
        dataset_names.append(set_filename)
        print('Pickling %s.' % set_filename)
        dataset = load_image_dataset(folder)
        print('Dataset tensor: ', dataset.shape)
        print('Mean of Dataset: ', np.mean(dataset))
        print('Standard deviation of Dataset: ', np.std(dataset))
        try:
            with open(set_filename, 'wb') as f:
                pickle.dump(dataset, f)
        except Exception as e:
            print('Unable to save data to', set_filename, ':', e)

    return dataset_names


train_datasets = make_pickle_files(train_folders)

def make_arrays(nb_rows, img_size, img_width):
    if nb_rows:
        dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
        labels = np.ndarray(nb_rows, dtype=np.int32)
    else:
        dataset, labels = None, None
    return dataset, labels


def make_datasets_all_classes(pickle_files, train_size, valid_size=0):
    num_classes = len(pickle_files)
    valid_dataset, valid_labels = make_arrays(valid_size, image_size, image_size_width)
    train_dataset, train_labels = make_arrays(train_size, image_size, image_size_width)
    vsize_per_class = valid_size // num_classes
    tsize_per_class = train_size // num_classes

    start_v, start_t = 0, 0
    end_v, end_t = vsize_per_class, tsize_per_class
    end_l = vsize_per_class + tsize_per_class
    for label, pickle_file in enumerate(pickle_files):
        try:
            with open(pickle_file, 'rb') as f:
                letter_set = pickle.load(f)
                # let's shuffle the letters to have random validation and training set
                np.random.shuffle(letter_set)
                if valid_dataset is not None:
                    valid_letter = letter_set[:vsize_per_class, :, :]
                    valid_dataset[start_v:end_v, :, :] = valid_letter
                    valid_labels[start_v:end_v] = label
                    start_v += vsize_per_class
                    end_v += vsize_per_class
                    valid_test_letter = letter_set[vsize_per_class:tsize_per_class, : , :]
                    train_dataset[start_t:end_t, : , :] = valid_test_letter
                    test_labels[start_t:end_t] = label
                    start_t += tsize_per_class
                    end_t += tsize_per_class

                train_letter = letter_set[vsize_per_class + tsize_per_class:end_l, :, :]
                train_dataset[start_t:end_t, :, :] = train_letter
                train_labels[start_t:end_t] = label
                start_t += tsize_per_class
                end_t += tsize_per_class
        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise

    return valid_dataset, valid_labels, train_dataset, train_labels, test_dataset, test_labels


train_size = 45000
valid_size = 10000
test_size = 10000

valid_dataset, valid_labels, train_dataset, train_labels, test_dataset, test_labels = make_datasets_all_classes(
    train_datasets, train_size, valid_size)

print(' Training Dataset:', train_dataset.shape, train_labels.shape)
print('Validation Dataset:', valid_dataset.shape, valid_labels.shape)
print('Testing Dataset:', test_dataset.shape, test_labels.shape)

# Function to dump all the datasets so that it can be reused
pickle_file = 'Imagely.pickle'

try:
  f = open(pickle_file, 'wb')
  save = {
    'train_dataset': train_dataset,
    'train_labels': train_labels,
    'valid_dataset': valid_dataset,
    'valid_labels': valid_labels,
    'test_dataset': test_dataset,
    'test_labels': test_labels,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise

statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)


def extract_overlap(dataset_1, dataset_2):
  overlap = {}
  for i, img_1 in enumerate(dataset_1):
    for j, img_2 in enumerate(dataset_2):
      if np.array_equal(img_1, img_2):
        if not i in overlap.keys():
          overlap[i] = []
        overlap[i].append(j)
  return overlap

overlap_test_train = extract_overlap(test_dataset , train_dataset)

regr = LogisticRegression(multi_class="multinomial", solver="lbfgs")
X_test = test_dataset.reshape(test_dataset.shape[0],test_dataset.shape[1]*test_dataset.shape[2])
y_test = test_labels

sample_size = 50
print (50)
X_train = train_dataset[:sample_size].reshape(sample_size, train_dataset.shape[1]*train_dataset.shape[2])
y_train = train_labels[:sample_size]
regr.fit(X_train, y_train)
regr.score(X_test, y_test)

sample_size = 100
print (100)
X_train = train_dataset[:sample_size].reshape(sample_size, train_dataset.shape[1]*train_dataset.shape[2])
y_train = train_labels[:sample_size]
regr.fit(X_train, y_train)
regr.score(X_test, y_test)


sample_size = 1000
print (1000)
X_train = train_dataset[:sample_size].reshape(sample_size, train_dataset.shape[1]*train_dataset.shape[2])
y_train = train_labels[:sample_size]
regr.fit(X_train, y_train)
regr.score(X_test, y_test)

sample_size = 5000
print (5000)
X_train = train_dataset[:sample_size].reshape(sample_size, train_dataset.shape[1]*train_dataset.shape[2])
y_train = train_labels[:sample_size]
regr.fit(X_train, y_train)
regr.score(X_test, y_test)

regr2 = LogisticRegression(solver='sag', multi_class="multinomial")
sample_size = len(train_dataset)
print (sample_size)
X_train = train_dataset[:sample_size].reshape(sample_size, train_dataset.shape[1]*train_dataset.shape[2])
y_train = train_labels[:sample_size]
regr2.fit(X_train, y_train)
print (regr2.score(X_test, y_test))
