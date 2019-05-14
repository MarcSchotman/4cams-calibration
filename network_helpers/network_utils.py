from __future__ import print_function, division
import os,time,cv2, sys, math
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time, datetime
import os, random
from scipy.misc import imread
import ast
from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report, \
    accuracy_score, f1_score

from network_helpers import helpers


def shuffle_rgb(img):

    index = [0,1,2]

    r = index[random.randint(0,len(index)-1)]
    index_r = index.index(r)
    del index[index_r]

    g = index[random.randint(0,len(index)-1)]
    index_g = index.index(g)    
    del index[index_g]

    b = index[0]

    shuffled_img = np.zeros(img.shape)
    shuffled_img[:,:,0] = img[:,:,r]
    shuffled_img[:,:,1] = img[:,:,g]
    shuffled_img[:,:,2] = img[:,:,b]

    return shuffled_img

def rad2Deg(rad):
    return rad * (180 / math.pi)


def deg2Rad(deg):
    return deg * (math.pi / 180)
def warpMatrix(sw, sh, theta, phi, gamma, scale, fovy):
    st = math.sin(deg2Rad(theta))
    ct = math.cos(deg2Rad(theta))
    sp = math.sin(deg2Rad(phi))
    cp = math.cos(deg2Rad(phi))
    sg = math.sin(deg2Rad(gamma))
    cg = math.cos(deg2Rad(gamma))

    halfFovy = fovy * 0.5
    d = math.hypot(sw, sh)
    sideLength = scale * d / math.cos(deg2Rad(halfFovy))
    if halfFovy == 0:
        h =0
    else:
        h = d / (2.0 * math.sin(deg2Rad(halfFovy)))
    n = h - (d / 2.0)
    f = h + (d / 2.0)

    Rtheta = np.identity(4)
    Rphi = np.identity(4)
    Rgamma = np.identity(4)

    T = np.identity(4)
    P = np.zeros((4, 4))

    Rtheta[0, 0] = Rtheta[1, 1] = ct
    Rtheta[0, 1] = -st
    Rtheta[1, 0] = st

    Rphi[1, 1] = Rphi[2, 2] = cp
    Rphi[1, 2] = -sp
    Rphi[2, 1] = sp

    Rgamma[0, 0] = cg
    Rgamma[2, 2] = cg
    Rgamma[0, 2] = sg
    Rgamma[2, 0] = sg

    T[2, 3] = -h

    if halfFovy ==0:
        P[0,0]=0
    else:
        P[0, 0] = P[1, 1] = 1.0 / math.tan(deg2Rad(halfFovy))
    P[2, 2] = -(f + n) / (f - n)
    P[2, 3] = -(2.0 * f * n) / (f - n)
    P[3, 2] = -1.0

    F = np.matmul(Rtheta, Rgamma)
    F = np.matmul(Rphi, F)
    F = np.matmul(T, F)
    F = np.matmul(P, F)

    ptsIn = np.zeros(12)
    ptsOut = np.zeros(12)
    halfW = sw / 2
    halfH = sh / 2

    ptsIn[0] = -halfW
    ptsIn[1] = halfH
    ptsIn[3] = halfW
    ptsIn[4] = halfH
    ptsIn[6] = halfW
    ptsIn[7] = -halfH
    ptsIn[9] = -halfW
    ptsIn[10] = -halfH
    ptsIn[2] = ptsIn[5] = ptsIn[8] = ptsIn[11] = 0

    ptsInMat = np.array([[ptsIn[0], ptsIn[1], ptsIn[2]], [ptsIn[3], ptsIn[4], ptsIn[5]], [ptsIn[6], ptsIn[7], ptsIn[8]],
                         [ptsIn[9], ptsIn[10], ptsIn[11]]], dtype=np.float32)
    ptsOutMat = np.array(
        [[ptsOut[0], ptsOut[1], ptsOut[2]], [ptsOut[3], ptsOut[4], ptsOut[5]], [ptsOut[6], ptsOut[7], ptsOut[8]],
         [ptsOut[9], ptsOut[10], ptsOut[11]]], dtype=np.float32)
    ptsInMat = np.array([ptsInMat])
    ptsOutMat = cv2.perspectiveTransform(ptsInMat, F)

    ptsInPt2f = np.array([[0, 0], [0, 0], [0, 0], [0, 0]], dtype=np.float32)
    ptsOutPt2f = np.array([[0, 0], [0, 0], [0, 0], [0, 0]], dtype=np.float32)

    i = 0

    while i < 4:
        ptsInPt2f[i][0] = ptsIn[i * 3 + 0] + halfW
        ptsInPt2f[i][1] = ptsIn[i * 3 + 1] + halfH
        ptsOutPt2f[i][0] = (ptsOutMat[0][i][0] + 1) * sideLength * 0.5
        ptsOutPt2f[i][1] = (ptsOutMat[0][i][1] + 1) * sideLength * 0.5
        i = i + 1

    M = cv2.getPerspectiveTransform(ptsInPt2f, ptsOutPt2f)
    return M


def warpImage(src, theta, phi, gamma, scale, fovy, interpolation):
    halfFovy = fovy * 0.5
    d = math.hypot(src.shape[1], src.shape[0])
    sideLength = scale * d / math.cos(deg2Rad(halfFovy))
    sideLength = np.int32(sideLength)
    (h,w,ch) = src.shape
    M = warpMatrix(src.shape[1], src.shape[0], theta, phi, gamma, scale, fovy)
    dst = cv2.warpPerspective(src, M, (sideLength, sideLength), flags=interpolation)
    

    x1 = 0; y1 = 0
    x2 = w; y2 = h

    #corners
    c1 = [x1,y1,1]; c2 = [x1,y2,1];
    c3 = [x2,y1,1]; c4 = [x2,y2,1];

    #warped of corners (x,y)
    Wc1 = (np.dot(M[0,:],c1)/np.dot(M[2,:],c1), np.dot(M[1,:],c1)/np.dot(M[2,:],c1))
    Wc2 = (np.dot(M[0,:],c2)/np.dot(M[2,:],c2), np.dot(M[1,:],c2)/np.dot(M[2,:],c2))
    Wc3 = (np.dot(M[0,:],c3)/np.dot(M[2,:],c3), np.dot(M[1,:],c3)/np.dot(M[2,:],c3))
    Wc4 = (np.dot(M[0,:],c4)/np.dot(M[2,:],c4), np.dot(M[1,:],c4)/np.dot(M[2,:],c4))

    #get the max crop size
    Lminy = min(Wc1[1],Wc2[1])
    Lmaxy = max(Wc1[1],Wc2[1])
    Lminx = min(Wc1[0],Wc2[0])
    Lmaxx = max(Wc1[0],Wc2[0])

    Rminy = min(Wc3[1],Wc4[1])
    Rmaxy = max(Wc3[1],Wc4[1])
    Rminx = min(Wc3[0],Wc4[0])
    Rmaxx = max(Wc3[0],Wc4[0])

    minY = int(max(Lminy,Rminy))
    maxY = int(min(Lmaxy,Rmaxy))

    minX = int(Lmaxx)
    maxX = int(Rminx)

    #crop
    if len(dst.shape) == 3:
        dstcrop = dst[minY:maxY,minX:maxX,:]
    else:
        dtscrop = dst[minY:maxY,minX:maxX]    
    
    return dstcrop

def add_gaussian_noise(img,var):

    row, col, ch = img.shape
    # Gaussian distribution parameters
    mean = 0
    sigma = var ** 0.5
    

    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = img + gauss

    return noisy



def resize_image(image, width = None, height = None, inter = cv2.INTER_NEAREST):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)
        
    # otherwise, the height is None
    elif height is None:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))
    else:
        dim  = (width,height)

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def overlay_label(img,label,alpha):
    return cv2.addWeighted(img,1,label,alpha,0)
    
def prepare_data(dataset_dir):
    train_input_names=[]
    train_output_names=[]
    val_input_names=[]
    val_output_names=[]
    test_input_names=[]
    test_output_names=[]
    for file in os.listdir(dataset_dir + "/train"):
        cwd = os.getcwd()
        train_input_names.append(cwd + "/" + dataset_dir + "/train/" + file)
    for file in os.listdir(dataset_dir + "/train_labels"):
        cwd = os.getcwd()
        train_output_names.append(cwd + "/" + dataset_dir + "/train_labels/" + file)
    for file in os.listdir(dataset_dir + "/val"):
        cwd = os.getcwd()
        val_input_names.append(cwd + "/" + dataset_dir + "/val/" + file)
    for file in os.listdir(dataset_dir + "/val_labels"):
        cwd = os.getcwd()
        val_output_names.append(cwd + "/" + dataset_dir + "/val_labels/" + file)
    for file in os.listdir(dataset_dir + "/test"):
        cwd = os.getcwd()
        test_input_names.append(cwd + "/" + dataset_dir + "/test/" + file)
    for file in os.listdir(dataset_dir + "/test_labels"):
        cwd = os.getcwd()
        test_output_names.append(cwd + "/" + dataset_dir + "/test_labels/" + file)
    train_input_names.sort(),train_output_names.sort(), val_input_names.sort(), val_output_names.sort(), test_input_names.sort(), test_output_names.sort()
    return train_input_names,train_output_names, val_input_names, val_output_names, test_input_names, test_output_names

def load_image(path):
    return cv2.cvtColor(cv2.imread(path,1), cv2.COLOR_BGR2RGB)

# Takes an absolute file path and returns the name of the file without th extension
def filepath_to_name(full_name):
    file_name = os.path.basename(full_name)
    file_name = os.path.splitext(file_name)[0]
    return file_name

# Print with time. To console or file
def LOG(X, f=None):
    time_stamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    if not f:
        print(time_stamp + " " + X)
    else:
        f.write(time_stamp + " " + X)


# Count total number of parameters in the model
def count_params():
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print("This model has %d trainable parameters"% (total_parameters))

# Subtracts the mean images from ImageNet
def mean_image_subtraction(inputs, means=[123.68, 116.78, 103.94]):
    inputs=tf.to_float(inputs)
    num_channels = inputs.get_shape().as_list()[-1]
    if len(means) != num_channels:
      raise ValueError('len(means) must match the number of channels')
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=inputs)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=3, values=channels)

def _lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    gts = tf.reduce_sum(gt_sorted)
    intersection = gts - tf.cumsum(gt_sorted)
    union = gts + tf.cumsum(1. - gt_sorted)
    jaccard = 1. - intersection / union
    jaccard = tf.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)
    return jaccard

def _flatten_probas(probas, labels, ignore=None, order='BHWC'):
    """
    Flattens predictions in the batch
    """
    if order == 'BCHW':
        probas = tf.transpose(probas, (0, 2, 3, 1), name="BCHW_to_BHWC")
        order = 'BHWC'
    if order != 'BHWC':
        raise NotImplementedError('Order {} unknown'.format(order))
    C = probas.shape[3]
    probas = tf.reshape(probas, (-1, C))
    labels = tf.reshape(labels, (-1,))
    if ignore is None:
        return probas, labels
    valid = tf.not_equal(labels, ignore)
    vprobas = tf.boolean_mask(probas, valid, name='valid_probas')
    vlabels = tf.boolean_mask(labels, valid, name='valid_labels')
    return vprobas, vlabels

def _lovasz_softmax_flat(probas, labels, only_present=True):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      only_present: average only on classes present in ground truth
    """
    C = probas.shape[1]
    losses = []
    present = []
    for c in range(C):
        fg = tf.cast(tf.equal(labels, c), probas.dtype) # foreground for class c
        if only_present:
            present.append(tf.reduce_sum(fg) > 0)
        errors = tf.abs(fg - probas[:, c])
        errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], name="descending_sort_{}".format(c))
        fg_sorted = tf.gather(fg, perm)
        grad = _lovasz_grad(fg_sorted)
        losses.append(
            tf.tensordot(errors_sorted, tf.stop_gradient(grad), 1, name="loss_class_{}".format(c))
                      )
    losses_tensor = tf.stack(losses)
    if only_present:
        present = tf.stack(present)
        losses_tensor = tf.boolean_mask(losses_tensor, present)
    return losses_tensor

def lovasz_softmax(probas, labels, only_present=True, per_image=False, ignore=None, order='BHWC'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, H, W, C] or [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      only_present: average only on classes present in ground truth
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
      order: use BHWC or BCHW
    """
    probas = tf.nn.softmax(probas, 3)
    labels = helpers.reverse_one_hot(labels)

    if per_image:
        def treat_image(prob, lab):
            prob, lab = tf.expand_dims(prob, 0), tf.expand_dims(lab, 0)
            prob, lab = _flatten_probas(prob, lab, ignore, order)
            return _lovasz_softmax_flat(prob, lab, only_present=only_present)
        losses = tf.map_fn(treat_image, (probas, labels), dtype=tf.float32)
    else:
        losses = _lovasz_softmax_flat(*_flatten_probas(probas, labels, ignore, order), only_present=only_present)
    return losses


# Randomly crop the image to a specific size. For data augmentation
def random_crop(image, label, crop_height, crop_width):
    if (image.shape[0] != label.shape[0]) or (image.shape[1] != label.shape[1]):
        raise Exception('Image and label must have the same dimensions!')
        
    if (crop_width <= image.shape[1]) and (crop_height <= image.shape[0]):
        x = random.randint(0, image.shape[1]-crop_width)
        y = random.randint(0, image.shape[0]-crop_height)
        
        if len(label.shape) == 3:
            return image[y:y+crop_height, x:x+crop_width, :], label[y:y+crop_height, x:x+crop_width, :]
        else:
            return image[y:y+crop_height, x:x+crop_width, :], label[y:y+crop_height, x:x+crop_width]
    else:

        # raise Exception('Crop shape (%d, %d) exceeds image dimensions (%d, %d)!' % (crop_height, crop_width, image.shape[0], image.shape[1]))
        print('WATCH IT: Crop shape (%d, %d) exceeds image dimensions (%d, %d)!' % (crop_height, crop_width, image.shape[0], image.shape[1]))
        if (crop_width >= image.shape[1]) and (crop_height >= image.shape[0]):
            return image, label

        elif crop_width >= image.shape[1]:        
            y = random.randint(0, image.shape[0]-crop_height)
            if len(label.shape) == 3:
                return image[y:y+crop_height,:,:], label[y:y+crop_height,:,:]
            else:
                return image[y:y+crop_height,:], label[y:y+crop_height,:]
        
        elif crop_height >= image.shape[0]:

            x = random.randint(0, image.shape[1]-crop_width)
            if len(label.shape) == 3:
                return image[:,x:x+crop_width,:], label[x:x+crop_width,:]
            else:
                return image[:,x:x+crop_width], label[:,x:x+crop_width]
        else:
            raise Exception("Something wrong in image_cropper...")





# Compute the average segmentation accuracy across all classes
def compute_global_accuracy(pred, label):
    total = len(label)
    count = 0.0
    for i in range(total):
        if pred[i] == label[i]:
            count = count + 1.0
    return float(count) / float(total)

# Compute the class-specific segmentation accuracy
def compute_class_accuracies(pred, label, num_classes):
    total = []
    for val in range(num_classes):
        total.append((label == val).sum())

    count = [0.0] * num_classes
    for i in range(len(label)):
        if pred[i] == label[i]:
            count[int(pred[i])] = count[int(pred[i])] + 1.0

    # If there are no pixels from a certain class in the GT, 
    # it returns NAN because of divide by zero
    # Replace the nans with a 1.0.
    accuracies = []
    for i in range(len(total)):
        if total[i] == 0:
            accuracies.append(1.0)
        else:
            accuracies.append(count[i] / total[i])

    return accuracies


def compute_mean_iou(pred, label):

    unique_labels = np.unique(label)
    num_unique_labels = len(unique_labels);

    I = np.zeros(num_unique_labels)
    U = np.zeros(num_unique_labels)

    for index, val in enumerate(unique_labels):
        pred_i = pred == val
        label_i = label == val

        I[index] = float(np.sum(np.logical_and(label_i, pred_i)))
        U[index] = float(np.sum(np.logical_or(label_i, pred_i)))


    mean_iou = np.mean(I / U)
    return mean_iou


def evaluate_segmentation(pred, label, num_classes, score_averaging="weighted"):
    flat_pred = pred.flatten()
    flat_label = label.flatten()

    global_accuracy = compute_global_accuracy(flat_pred, flat_label)
    class_accuracies = compute_class_accuracies(flat_pred, flat_label, num_classes)

    prec = precision_score(flat_pred, flat_label, average=score_averaging)
    rec = recall_score(flat_pred, flat_label, average=score_averaging)
    f1 = f1_score(flat_pred, flat_label, average=score_averaging)

    iou = compute_mean_iou(flat_pred, flat_label)

    return global_accuracy, class_accuracies, prec, rec, f1, iou

    
def compute_class_weights(labels_dir, label_values):
    '''
    Arguments:
        labels_dir(list): Directory where the image segmentation labels are
        num_classes(int): the number of classes of pixels in all images

    Returns:
        class_weights(list): a list of class weights where each index represents each class label and the element is the class weight for that label.

    '''
    image_files = [os.path.join(labels_dir, file) for file in os.listdir(labels_dir) if file.endswith('.png')]

    num_classes = len(label_values)

    class_pixels = np.zeros(num_classes) 

    total_pixels = 0.0

    for n in range(len(image_files)):
        image = imread(image_files[n])                

        for index, colour in enumerate(label_values):
            class_map = np.all(np.equal(image, colour), axis = -1)            
            class_map = class_map.astype(np.float32)
            class_pixels[index] += np.sum(class_map)

            
        print("\rProcessing image: " + str(n) + " / " + str(len(image_files)), end="")
        sys.stdout.flush()


    total_pixels = float(np.sum(class_pixels))
    class_weights = np.zeros(num_classes)
    #if a class is not present we give it weight zero
    for index in range(0,num_classes):
        if class_pixels[index] == 0:
            class_weights[index] = 0
        else:
           class_weights[index] = total_pixels / class_pixels[index]

    class_weights = class_weights / np.sum(class_weights)

    return class_weights

# Compute the memory usage, for debugging
def memory():
    import os
    import psutil
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0]/2.**30  # Memory use in GB
    print('Memory usage in GBs:', memoryUse)

