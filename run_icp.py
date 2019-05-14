from time import clock
from ICP.icp import *
from utils import *
import numpy as np
import os, random


def rotation_matrix(axis, theta):
    axis = axis/np.sqrt(np.dot(axis, axis))
    a = np.cos(theta/2.)
    b, c, d = -axis*np.sin(theta/2.)

    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                  [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                  [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])


def run_icp(points_left, points_right, max_iters = 20, use_labels= False, one_hot_labels_left= None, one_hot_labels_right= None, weight = 0.1, tolerance = 0.0001):
    
    """
    points left will be fit ON points right i.e. points left are the source and with destination points_right
    """

    #Maps points_left on to points_right
    if use_labels:
        
        #For some very dark illuster python reason just saying:
        #`labels_left[labels_left>0] = weight` does not work.......
        #This workaround works
        weighted_labels_left = np.array(one_hot_labels_left> 0)
        weighted_labels_left = weighted_labels_left.astype(int) * weight

        weighted_labels_right = np.array(one_hot_labels_right> 0)
        weighted_labels_right = weighted_labels_right.astype(int) * weight

        data_left = add_labels2points(points_left, weighted_labels_left)
        data_right = add_labels2points(points_right, weighted_labels_right)
    else:

        data_left = points_left
        data_right = points_right

    t1 = clock()
    H, distances, i, error = icp(data_left, data_right, use_labels=use_labels, max_iterations= max_iters, tolerance = tolerance)
    t2 = clock()
    time = round(t2 - t1,4)
    print("ICP resutls:\n\tTime: \t{} seconds\n\tError: \t{}".format(time, error))

    for row in H:
        if (row == H[0,:]).all():
            print("\tH:\t{}".format(row))
        else:
            print("\t\t{}".format(row))

    points_left = concatenate_XYZ1(points_left[:,0], points_left[:,1], points_left[:,2])
    points_left = np.matmul(H, points_left)
    points_left = points_left[0:3,:]
    points_left = np.transpose(points_left)

    return points_left, points_right, error

def PCA(points):
    """
    INPUT:
        - points as Nx3

        Find the dims principal components
        (ordered from big to small)
        returns unit vectors w as dimsx3 unit vectors
    """
    
    w = []
    max_points = 1000

    # points,_ = remove_outliers(points, 2.5)
    if len(points) > max_points:
        take_random_points = random.sample(range(0,len(points)), max_points)
        points = points[take_random_points,:]

    #Remove mean
    points = points - np.mean(points,axis = 0)
    
    covariance = np.cov(points.T)

    values, vectors = np.linalg.eig(covariance)

    sorted_indices = np.argsort(-values)

    values_sorted = values[sorted_indices]
    vectors_sorted = vectors[:,sorted_indices]


    return vectors_sorted

def draw_vector_lines(vectors, start_point):
    vector_lines = []
    interval = 0.025
    sizes = np.arange(interval,.3,interval)


    for vector in vectors:
        for size in sizes:
            point =  -vector * np.array(size)
                           
            point[0] +=start_point[0]
            point[1] +=start_point[1]
            point[2] +=start_point[2]

            vector_lines.append(point)

    return np.array(vector_lines)

if __name__ == "__main__":
    ##  INPUTS  ##
    max_iters = 10
    use_labels = False
    weights = 0.1 #list of to be used weights
    CROP_SIDES = True
    #SAVE RESULTS?
    SAVE = False

    save_name_icp_points_4cams = os.getcwd() + "/Data/RESULTS/icp/icp_both_sides.npy"      
    save_name_icp_points_2cams_side1 = os.getcwd() + "/Data/RESULTS/icp/side1_icp.npy"
    save_name_icp_points_2cams_side2 = os.getcwd() + "/Data/RESULTS/icp/side2_icp.npy"
    
    # with 4 cams : 
    side1_right = np.load('Data/RESULTS/side1_points_right.npy')
    side1_left = np.load('Data/RESULTS/side1_points_left.npy')
    side2_right = np.load('Data/RESULTS/side2_points_right.npy')
    side2_left = np.load('Data/RESULTS/side2_points_left.npy')

    class_dict ='./Data/NNModel/class_dict.csv'
    points_left_side1 = side1_left[0]
    labels_left_side1 = side1_left[1]

    points_right_side1 = side1_right[0]
    labels_right_side1= side1_right[1]

    points_left_side2 = side2_left[0]
    labels_left_side2 = side2_left[1]

    points_right_side2 = side2_right[0]
    labels_right_side2= side2_right[1]

    #BEFORE ICP
    labels_side1 = np.concatenate((np.array(labels_left_side1), labels_right_side1), axis = 0) 
    labels_side2 = np.concatenate((np.array(labels_left_side2), labels_right_side2), axis = 0)
    labels_both = np.concatenate((labels_left_side1, labels_right_side1, np.concatenate((labels_left_side2, labels_right_side2),axis = 0)), axis = 0)

    if use_labels:

        one_hot_labels_left_side2 = convert_label_to_one_hot(labels_left_side2, class_dict)
        one_hot_labels_right_side2 = convert_label_to_one_hot(labels_right_side2, class_dict)

        one_hot_labels_left_side1 = convert_label_to_one_hot(labels_left_side1, class_dict)
        one_hot_labels_right_side1 = convert_label_to_one_hot(labels_right_side1, class_dict)

        labels_list = [ one_hot_labels_left_side1, one_hot_labels_right_side1, one_hot_labels_left_side2, one_hot_labels_right_side2,
                        np.concatenate((one_hot_labels_left_side1, one_hot_labels_right_side1), axis = 0), np.concatenate((one_hot_labels_left_side2, one_hot_labels_right_side2), axis = 0)]

        print("Running test %s with weight %s!\n"%(weights.index(weight),weight))

    point_list = [points_left_side1, points_right_side1, points_left_side2, points_right_side2]
        
    for i in range(3):
        points_left = point_list[i*2]
        points_right = point_list[i*2+1]

        if i ==2:
            points_right = points_side2side(points_right)
        
        if i ==0:
            camera = "side1"
        elif i ==1:
            camera = "side2"
        else:
            camera = "Both"
        print("Running ICP with %s cameras!\n \tLeft cam: %s points\n\tRight cam: %s points"%(camera,len(points_left), len(points_right)))
        t0 = clock()
        if use_labels:
            points_left, points_right, error = run_icp(points_left, points_right, use_labels = use_labels, max_iters = max_iters, one_hot_labels_left = labels_list[i*2], one_hot_labels_right =  labels_list[i*2+1], weight = weight)
        else:
            points_left, points_right, error = run_icp(points_left, points_right, use_labels = use_labels, max_iters = max_iters)
        t1 = clock()
        concatenated_points = np.concatenate((points_left, points_right), axis = 0)
        point_list.append(concatenated_points)
        time = round(t1 - t0, 3)

        print("ICP resutls:\n\tTime: \t\t%s seconds\n\tError: \t\t%s"%(time, error))

    make_plot(point_list[4], labels_side1/255)
    make_plot(point_list[5], labels_side2/255)

    make_plot(np.concatenate((point_list[4], points_side2side(point_list[5])), axis = 0), labels_both/255)
    make_plot(point_list[6], labels_both/255)
            

    if SAVE:

        np.save(save_name_icp_points_4cams, [point_list[6], labels_both])
        np.save(save_name_icp_points_2cams_side1, [point_list[5], labels_side1])
        np.save(save_name_icp_points_2cams_side2, [point_list[4], labels_side2])
