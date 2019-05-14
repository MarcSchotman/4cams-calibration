import numpy as np 
import pyrealsense2 as rs
import re, csv, os
from plyfile import PlyData
import cv2 as cv
import time
import pptk
import requests

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


def download_model_weights():
    print("Downloading model weights from Marc Schotman's Google Drive! This might take a minute.")
    file_id = '1-Ngc8lqEdw4P4ZTEWKHW_dGw_jvajR09'
    destination = 'Data/NNModel/latest_model.ckpt.data-00000-of-00001'

    print("\tDownloading latest_model.ckpt.data-00000-of-00001...")
    download_file_from_google_drive(file_id, destination)


    destination = 'Data/NNModel/latest_model.ckpt.index'
    file_id = "188TpeiPmXV2_S9296Ac1A_6QvY6IZw8E"
    print("\tDownloading latest_model.ckpt.index...")
    download_file_from_google_drive(file_id, destination)


    file_id = "1Z9YaZ2P95r2LSBywQeaIXesk-2FQ0TYf"
    destination = 'Data/NNModel/latest_model.ckpt.meta'
    print("\tDownloading latest_model.ckpt.meta...")
    download_file_from_google_drive(file_id, destination)

def points_side2side(input_ply):

    (r,c) = input_ply.shape

    if r ==3:
        points = np.ones((4,c))
        points[0:3, :] = input_ply
    elif c==3:
        points = np.ones((4,r))
        points[0:3,:] = np.transpose(input_ply)
    else:
        points=input_ply



    number_squares = 6
    size_squares= 0.042
    thickness_plate = 0.005

    #ASSUMES SIZE AND NUMBER OF SQUARES
    #DURING NORMAL CALIBRATION
    R = np.array([[-1,0,0],
                [0,1,0],
                [0,0,-1]])
    
    translation_volans_asus = np.array([number_squares*size_squares,0,thickness_plate])

    H = np.zeros((4,4))
    H[0:3,0:3] = R
    H[0:3,3] = translation_volans_asus
    H[3,3] = 1 

    points = np.matmul(H,points)
    points = points[0:3,:]
    return np.transpose(points)


def make_color_array(len_plot_points, color):

    color_array = np.zeros((len_plot_points,3))
    R,G,B = color
    color_array[:,0] = R; color_array[:,1] = G; color_array[:,2] = B
    return color_array


def timer():
   now = time.localtime(time.time())
   return '%sm:%ss'%(now[4], now[5])


def make_list(extrinsics):
    list_extr = re.split(',', str(extrinsics), 7) # string converted into list
    del list_extr[6]
    # supress the string parts of the list
    list_extr=[i.split(': ', 1)[1] for i in list_extr]            
    return list_extr


def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv.line(img, corner, tuple(imgpts[0].ravel()), (0,255,0), 5)
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0,0,255), 5)
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (255,0,0), 5)
    return img

def get_color_intr_depth2color_extr(path_bag):
    # read the file .bag # https://github.com/IntelRealSense/librealsense/issues/314
    no_frames_cnt = 0
    while True:
        pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device_from_file(path_bag)
        pipe_profile = pipeline.start(cfg)
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not depth_frame.get_data() or not color_frame.get_data():
            no_frames_cnt+=1

            if no_frames_cnt < 50:
                continue
            else:
                raise Exception("ERROR in `get_color_intr_depth2color_extr()`: Could not find BOTH a color and depth frame....")
        
        # Get the Intrinsics & Extrinsics
        color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
        depth_to_color_extrin = depth_frame.profile.get_extrinsics_to(color_frame.profile)
    
        trans_depth2color = np.zeros((4,4))
        trans_depth2color[0:3,0:3] = np.reshape(depth_to_color_extrin.rotation, (3,3))
        trans_depth2color[0:3,3] = depth_to_color_extrin.translation
        trans_depth2color[3,3] = 1

        pipeline.stop()
        break

    color_intrin = make_list(color_intrin)

    # intrinisic matrices
    color_intrin = make_intrinsic_matrix(color_intrin)

    return color_intrin, trans_depth2color


def get_values_from_csv(path):
   return np.loadtxt(path, delimiter =';', usecols=[0,1,2,3,4,5])


# build the intrinsic matrix associated to val_intr, which is : [width, height, ppx, ppy, fx, fy]
def make_intrinsic_matrix(val_intr):
    intr_arr= np.zeros((3,3))
    intr_arr[0,0] = val_intr[4]
    intr_arr[1,1] = val_intr[5]
    intr_arr[0,2] = val_intr[2]
    intr_arr[1,2] = val_intr[3]
    intr_arr[2,2] = 1
    return intr_arr

def remove_outliers(points, labels = 1, max_std = 3):
    (r,c) = points.shape

    if r == 3:
        X = points[0,:]
        Y = points[1,:]
        Z = points[2,:]
    elif c == 3:
        X = points[:,0]
        Y = points[:,1]
        Z = points[:,2]

    x_mean = np.mean(X)
    x_std = np.std(X)

    y_mean = np.mean(Y)
    y_std = np.std(Y)

    z_mean = np.mean(Z)
    z_std = np.std(Z)

    #Outliers defined as being further then 3 times std from the mean
    x_outliers1 =  np.array([X > x_mean + max_std*x_std], dtype = np.bool)
    x_outliers2 =  np.array([X < x_mean - max_std*x_std], dtype = np.bool)

    y_outliers1 =  np.array([Y > y_mean + max_std*y_std], dtype = np.bool)
    y_outliers2 =  np.array([Y < y_mean - max_std*y_std], dtype = np.bool)

    z_outliers1 =  np.array([Z > z_mean + max_std*z_std], dtype = np.bool)
    z_outliers2 =  np.array([Z < z_mean - max_std*z_std], dtype = np.bool)

    #gets indices where any of the booleans x_outliers1 ... z_outliers2 are True
    indices_to_delete = np.where(np.logical_or.reduce((x_outliers1, x_outliers2, y_outliers1, y_outliers2, z_outliers1, z_outliers2)))

    if c == 3:
        points_out = np.delete(points,indices_to_delete, axis = 0)
    elif r == 3:
        points_out = np.delete(points,indices_to_delete, axis = 1)

    if not isinstance(labels, int):
        labels_out = np.delete(labels, indices_to_delete, axis = 0)

    return points_out, labels_out

def load_PLY(path_ply):
    if not path_ply.endswith(".ply") or not path_ply.endswith(".npy"):
        if os.path.isfile(path_ply+".npy"):
            path_ply = path_ply + ".npy"
        elif os.path.isfile(path_ply+".ply"):
            path_ply = path_ply + ".ply"
        else:
            raise Exception("ERROR in `load_PLY()`: PLY file not found.. PLY_path=%s"%path_ply)

    # path_ply = path_ply + ".ply"
  
    if path_ply.endswith(".npy"):
        points = np.load(path_ply)

        if points.shape[1] == 3:
            points = points.T
        X = points[0,:]
        Y = points[1,:]
        Z = points[2,:]

    elif path_ply.endswith(".ply"):
        plydata = PlyData.read(path_ply)

        X = plydata['vertex']['x'] # X is a memmap : a memory-map to an array stored in a binary file on disk.
        Y = plydata['vertex']['y']
        Z = plydata['vertex']['z']

        X_temp = np.memmap.tolist(X) 
        Y_temp = np.memmap.tolist(Y)
        Z_temp = np.memmap.tolist(Z)

        #Now remove outliers ( + 3*std) and save as npy
        save_name = path_ply[:-4] + '.npy'

        points = remove_outliers(np.array([X_temp, Y_temp, Z_temp]))
        np.save(save_name, points)

        X = points[0,:]
        Y = points[1,:]
        Z = points[2,:]

    else:
        raise Exception("ERROR in `load_PLY()`: ply_path given is not found using extensions '.ply' and '.npy'... Given: %s"%path_ply)

    return X, Y, Z


def load_NPY(path_npy):
    points = np.load(path_npy)

    if points.shape[1] == 3:
        points = points.T
    X = points[0,:]
    Y = points[1,:]
    Z = points[2,:]

    return X, Y, Z

# return the mean of the values contained in a list
def mean_list(list_el):
    return np.sum(list_el)/len(list_el)

# concatenate lists of X, Y, Z with a list of 1
def concatenate_XYZ1(X_list, Y_list, Z_list):
    conc_XYZ = np.ones((4, len(X_list)))
    conc_XYZ[0, :] = X_list
    conc_XYZ[1, :] = Y_list
    conc_XYZ[2, :] = Z_list
    return conc_XYZ


# extraction of the 3 first lines in a matrix
def extr_XYZ(proj):
    return proj[0, :], proj[1, :], proj[2, :]


def make_plot(plot_points,colors, point_size = 0.0005):
    #label colors should be raning from 0 to 1. 
    v = pptk.viewer(plot_points)
    v.attributes(colors)
    v.set(point_size=point_size)


def remove_mean(plot_points):
    new_points = np.zeros(np.shape(plot_points))
    new_points[:,0] = plot_points[:,0] - sum(plot_points[:,0])/len(plot_points[:,0])
    new_points[:,1] = plot_points[:,1] - sum(plot_points[:,1])/len(plot_points[:,1])
    new_points[:,2] = plot_points[:,2] - sum(plot_points[:,2])/len(plot_points[:,2])
    return new_points


def crop_sides_x(plot_points, percentage, crop_direction = None, labels = None):
    #needs a Nx3 matrix
    min_x = min(plot_points[:,0])
    max_x = max(plot_points[:,0])

    L = max_x - min_x

    #cut percentage/2 from both sides 
    crop_distance = (L*percentage/100) / 2

    if crop_direction == None:
        indices_too_low_x = np.ma.where(plot_points[:,0]<(min_x+crop_distance))
        indices_too_high_x = np.ma.where(plot_points[:,0]>(max_x-crop_distance))
        indices_to_delete = tuple(indices_too_low_x[0]) + tuple(indices_too_high_x[0])
    
    elif crop_direction == 'Left' or crop_direction == 'left':
        indices_too_low_x = np.ma.where(plot_points[:,0]<(min_x+crop_distance*2))
        indices_to_delete = tuple(indices_too_low_x[0])
    
    elif crop_direction == "Right" or crop_direction =='right':
        indices_too_high_x = np.ma.where(plot_points[:,0]>(max_x-crop_distance*2))
        indices_to_delete = tuple(indices_too_high_x[0])
    else:
        raise Exception("ERROR in crop_sides_x(): crop_direction should be either 'left' or 'right' ")

    plot_points = np.delete(plot_points, indices_to_delete, axis = 0)

    if labels is not None:
        labels = np.delete(labels, indices_to_delete, axis = 0)

    return plot_points, labels


def crop_sides(plot_points, percentage, XYorZ,  crop_direction = 'both', labels = None):
    #needs a Nx3 matrix
    if XYorZ == 'x' or XYorZ == 'X':
        col = 0
    elif XYorZ == 'y' or XYorZ == 'Y':
        col = 1
    elif XYorZ == 'z' or XYorZ == 'Z':
        col = 2
    else:
        raise Exception("ERROR in crop_sides_XYorZ(): XYorZ must be X, Y or Z ")

    min_XYorZ = min(plot_points[:,col])
    max_XYorZ = max(plot_points[:,col])

    L = max_XYorZ - min_XYorZ

    #cut percentage/2 from both sides 
    crop_distance = (L*percentage/100) / 2

    if crop_direction =='both':
        indices_too_low_XYorZ = np.ma.where(plot_points[:,col]<(min_XYorZ+crop_distance))
        indices_too_high_XYorZ = np.ma.where(plot_points[:,col]>(max_XYorZ-crop_distance))
        indices_to_delete = tuple(indices_too_low_XYorZ[0]) + tuple(indices_too_high_XYorZ[0])
    
    elif crop_direction == 'Left' or crop_direction == 'left':
        indices_too_low_XYorZ = np.ma.where(plot_points[:,col]<(min_XYorZ+crop_distance*2))
        indices_to_delete = tuple(indices_too_low_XYorZ[0])
    
    elif crop_direction == "Right" or crop_direction =='right':
        indices_too_high_XYorZ = np.ma.where(plot_points[:,col]>(max_XYorZ-crop_distance*2))
        indices_to_delete = tuple(indices_too_high_XYorZ[0])
    else:
        raise Exception("ERROR in crop_sides_XYorZ(): crop_direction should be either 'left' or 'right' ")

    plot_points = np.delete(plot_points, indices_to_delete, axis = 0)

    if labels is not None:
        labels = np.delete(labels, indices_to_delete, axis = 0)

    return plot_points, labels

def print_some(points):
    (r,c) = points.shape

    if r == 3:
        X = points[0,:]
        Y = points[1,:]
        Z = points[2,:]
    elif c == 3:
        X = points[:,0]
        Y = points[:,1]
        Z = points[:,2]

    x_max = np.max(X)
    x_min = np.min(X)

    y_max = np.max(Y)
    y_min = np.min(Y)

    z_max = np.max(Z)
    z_min = np.min(Z)

    print("X: ",x_min, x_max)
    print("Y: ",y_min, y_max)
    print("Z: ",z_min, z_max)

def crop_default(points, cam):
    ##################################
    #############DEFAULTS#############
    ##################################

    if cam =="left" or cam == "Left":
        #############LEFT CAM#############
        min_x = -0.4; max_x = 0.8
        min_y = -0.4; max_y = 0.6
        min_z = -0.6; max_z = 0.2
    if cam =="right" or cam == "Right":
        #############LEFT CAM#############
        min_x = -0.6; max_x = 0.4
        min_y = -0.4; max_y = 0.6
        min_z = -0.6; max_z = 0.2

    (r,c) = points.shape

    if r == 3:
        X = points[0,:]
        Y = points[1,:]
        Z = points[2,:]
    elif c == 3:
        X = points[:,0]
        Y = points[:,1]
        Z = points[:,2]


    #Outliers defined as being further then 3 times std from the mean
    x_outliers1 =  np.array([X > max_x], dtype = np.bool)
    x_outliers2 =  np.array([X < min_x], dtype = np.bool)

    y_outliers1 =  np.array([Y > max_y], dtype = np.bool)
    y_outliers2 =  np.array([Y < min_y], dtype = np.bool)

    z_outliers1 =  np.array([Z > max_z], dtype = np.bool)
    z_outliers2 =  np.array([Z < min_z], dtype = np.bool)

    #gets indices where any of the booleans x_outliers1 ... z_outliers2 are True
    indices_to_delete = np.where(np.logical_or.reduce((x_outliers1, x_outliers2, y_outliers1, y_outliers2, z_outliers1, z_outliers2)))

    if c == 3:
        points_out = np.delete(points,indices_to_delete, axis = 0)
    elif r == 3:
        points_out = np.delete(points,indices_to_delete, axis = 1)

    return points_out


def get_label_info(csv_path):
    """
    Retrieve the class names and label values for the selected dataset.
    Must be in CSV format!

    # Arguments
        csv_path: The file path of the class dictionairy
        
    # Returns
        Two lists: one for the class names and the other for the label values
    """
    if not os.path.isfile(csv_path):
        raise Exception("ERROR in get_label_info: class_dict was not found...\nGiven:%s"%(csv_path))
    filename, file_extension = os.path.splitext(csv_path) 
    if not file_extension == ".csv":
        return ValueError("ERROR in get_label_info: File is not a CSV!")

    csv_path = os.path.abspath(csv_path)

    class_names = []
    label_values = []
    with open(csv_path, 'r') as csvfile:
        file_reader = csv.reader(csvfile, delimiter=',')
        header = next(file_reader)
        for row in file_reader:
            class_names.append(row[0])
            label_values.append([int(row[1]), int(row[2]), int(row[3])])
        # print(class_dict)
    return np.array(label_values)


def get_average_label(labels, label_values, weight_void = 1):
    labels_to_use = labels.astype(int)
    labels_to_use = labels_to_use.tolist()
    average_label = None
    count = 0
    for label in label_values :
        label = label.tolist()
        current_count = labels_to_use.count(label)

        #counts void labels as less important if a smaller weight is given
        
        if (label == [0,0,0]):
            
            current_count = 0 #current_count * weight_void

        if current_count > count:
            average_label = label
            count = current_count

    if count == 0:
        average_label = [0,0,0]

    return average_label


def convert_label_to_one_hot(label, class_dict):
    class_dict = os.path.abspath(class_dict)
    label_values = get_label_info(class_dict)

    # return label_out
    semantic_map = []
    for colour in label_values:
        # colour_map = np.full((label.shape[0], label.shape[1], label.shape[2]), colour, dtype=int)
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis = -1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)
    # print("Time 2 = ", time.time() - st)

    return semantic_map.astype(float)


def add_labels2points(points, labels):
    return np.concatenate((points,labels), axis=1)