import pyrealsense2 as rs
import cv2 as cv
import numpy as np
import os, csv, sys
from plyfile import PlyData
from run_icp import *
from utils import *


def get_PLY(ply_path, calibration_folder):

    print("\tGetting calibration files from: \n\t %s"%(calibration_folder))
    print("\tGetting PLY file from:\n\t%s"%(ply_path))
    
    ######################################################################
    ################### LOAD CALIBRATION FILES INPUTS ####################
    ######################################################################

    #trans_color2world, trans_world2color, trans_depth2color, camera_matrix, dist_coeff
    trans_color2world   = np.load(os.path.join(calibration_folder,'trans_color2world.npy'))
    trans_world2color   = np.load(os.path.join(calibration_folder,'trans_world2color.npy'))
    trans_depth2color   = np.load(os.path.join(calibration_folder,'trans_depth2color.npy'))
    camera_matrix       = np.load(os.path.join(calibration_folder,'camera_matrix.npy'))
    dist_coeff          = np.load(os.path.join(calibration_folder,'dist_coeff.npy'))

        
    ######################################################################
    ########################### MANIPULATE PLY ###########################
    ######################################################################
    
    X_color, Y_color, Z_color = load_NPY(ply_path)
    
    conc_XYZ1 = concatenate_XYZ1(X_color, Y_color, Z_color)
    # from depth to color, 
    conc_XYZ1 = np.matmul(trans_depth2color, conc_XYZ1)
    # from color to real world
    proj_w = np.matmul(trans_color2world, conc_XYZ1)

    (X_w, Y_w, Z_w) = (proj_w[0, :], proj_w[1, :], proj_w[2, :])

    points = np.transpose((np.array([X_w, Y_w, Z_w])))
    print("len of points =", len(points))
    print("\tLoaded PLY...")

    return points


if __name__ == '__main__':     

    #######################################################
    ######################## INPUTS #######################
    #######################################################
    
#              // SIDE1 \\  // SIDE 2 \\
    cam_list=['cam1','cam2','cam3','cam4']
    ply_names = ['cam1_ply.npy','cam2_ply.npy', 'cam3_ply.npy', 'cam4_ply.npy']
    img_names = ['cam1_ply_image.jpg','cam2_ply_image.jpg', 'cam3_ply_image.jpg', 'cam4_ply_image.jpg']
    data_folder = "Data"
    crop_sides_PLY = True
    ##ICP##
    RUN_ICP = True
    use_labels = False
    max_iters = 10
    #SAVE RESULTS?
    SAVE_PLEASE = True
    save_dir = os.path.join(os.getcwd(), 'Data','RESULTS')

    root=os.getcwd()

    #######################################################
    ######## SET PATH TO PLY + CALIBRATION FILES ##########
    #######################################################

    #SIDE 1
    ##RIGHT CAMERA
    side1_right_calibration  = os.path.join(root,data_folder, cam_list[0], 'intrinsics')

    side1_right_PLY = os.path.join(root, data_folder, cam_list[0], ply_names[0])
    side1_right_IMG = os.path.join(root, data_folder, cam_list[0], img_names[0])

    ##LEFT CAMERA
    side1_left_calibration  = os.path.join(root,data_folder, cam_list[1], 'intrinsics')

    side1_left_PLY = os.path.join(root, data_folder, cam_list[1], ply_names[1])
    side1_left_IMG = os.path.join(root, data_folder, cam_list[1], img_names[1])

    #SIDE 2
    ##LEFT CAMERA
    side2_right_calibration  = os.path.join(root,data_folder, cam_list[2], 'intrinsics')

    side2_right_PLY = os.path.join(root, data_folder, cam_list[2], ply_names[2])
    side2_right_IMG = os.path.join(root, data_folder, cam_list[2], img_names[2])

    ##RIGHT CAMERA
    side2_left_calibration = os.path.join(root,data_folder, cam_list[3], 'intrinsics')

    side2_left_PLY = os.path.join(root, data_folder, cam_list[3], ply_names[3])
    side2_left_IMG = os.path.join(root, data_folder, cam_list[3], img_names[3])
    
    ######################################################
    ####################### CHECKS #######################
    ######################################################

    for calibration_folder in [side1_right_calibration, side1_left_calibration, side2_right_calibration, side2_left_calibration]:
        if not os.path.isdir(calibration_folder):
            raise Exception("ERROR: calibration folder not found...\nCalibration folder:%s"%(calibration_folder))
    for PLY_path in [side1_right_PLY, side1_left_PLY, side2_right_PLY, side2_left_PLY]:
        if not os.path.isfile(PLY_path):
            raise Exception("ERROR: PLY file not found...\nPLY path:%s"%(PLY_path))
    for PLY_image in [ side1_right_IMG, side1_left_IMG, side2_right_IMG, side2_left_IMG]:
        if not os.path.isfile(PLY_image):
            raise Exception("ERROR: PLY image not found...\nPLY image:%s"%(PLY_image))



    #GET PLY POINTS
    print("Processing PLY files...")
    plot_points_right_side1 = get_PLY(side1_right_PLY, side1_right_calibration)
    plot_points_left_side1 = get_PLY(side1_left_PLY, side1_left_calibration)

    plot_points_right_side2 = get_PLY(side2_right_PLY, side2_right_calibration)
    plot_points_left_side2 = get_PLY(side2_left_PLY, side1_left_calibration)

    #Here some specific default crop settings are used
    if crop_sides_PLY:
        plot_points_right_side1 = crop_default(plot_points_right_side1, 'right')
        plot_points_left_side1 = crop_default(plot_points_left_side1, 'left')
        plot_points_right_side2 = crop_default(plot_points_right_side2, 'right')
        plot_points_left_side2 = crop_default(plot_points_left_side2, 'left')


    ######################################################################
    ############################## RUN ICP ###############################
    ######################################################################
    if RUN_ICP:
        #Rnu icp on each side seperately:
        print("Running ICP with side1 cameras!\n \tLeft cam: %s points\n\tRight cam: %s points"%(len(plot_points_right_side1), len(plot_points_left_side1)))
        plot_points_right_side1, plot_points_left_side1, error = run_icp(plot_points_right_side1, plot_points_left_side1, use_labels = use_labels, max_iters = max_iters)

        print("Running ICP with side2 cameras!\n \tLeft cam: %s points\n\tRight cam: %s points"%(len(plot_points_right_side2), len(plot_points_left_side2)))
        plot_points_right_side2, plot_points_left_side2, error = run_icp(plot_points_right_side2, plot_points_left_side2, use_labels = use_labels, max_iters = max_iters)
        #combine:
        plot_points_side1 = np.concatenate((plot_points_right_side1, plot_points_left_side1), axis =0)

        #combine:
        plot_points_side2 = np.concatenate((plot_points_right_side2, plot_points_left_side2), axis =0)

        side2_homogeneous_plot_points = concatenate_XYZ1(plot_points_side2[:,0], plot_points_side2[:,1], plot_points_side2[:,2])
        plot_points_side2_in_frame_side1 = points_side2side(side2_homogeneous_plot_points)

        #Run icp when  combining both sides:
        print("Running ICP with boh sides cameras!\n \tLeft cam: %s points\n\tRight cam: %s points"%(len(plot_points_side1), len(plot_points_side2)))
        
        plot_points_side1, plot_points_side2_in_frame_side1, error = run_icp(plot_points_side1, plot_points_side2_in_frame_side1, use_labels = use_labels, max_iters = max_iters)

        all_plot_points = np.concatenate((plot_points_side1, plot_points_side2_in_frame_side1), axis =0)
        all_plot_points = remove_mean(all_plot_points)
    else:
        #combine:
        plot_points_side1 = np.concatenate((plot_points_right_side1, plot_points_left_side1), axis =0)
        
        #combine:
        plot_points_side2 = np.concatenate((plot_points_right_side2, plot_points_left_side2), axis =0)

        side2_homogeneous_plot_points = concatenate_XYZ1(plot_points_side2[:,0], plot_points_side2[:,1], plot_points_side2[:,2])
        plot_points_side2_in_frame_side1 = points_side2side(side2_homogeneous_plot_points)

        all_plot_points = np.concatenate((plot_points_side1, plot_points_side2_in_frame_side1), axis =0)
        all_plot_points = remove_mean(all_plot_points)

    ######################################################################
    ############################ MAKE PLOTS ##############################
    ######################################################################
    
    #SIDE1
    labels_right_side1 = make_color_array(len(plot_points_right_side1), (.45,.1,0))
    labels_left_side1 = make_color_array(len(plot_points_left_side1), (.9,.2,0))

    side1_labels_highlighted= np.concatenate((labels_left_side1, labels_right_side1), axis =0)

    #SIDE2
    labels_right_side2 = make_color_array(len(plot_points_right_side2), (0,.45,.1))
    labels_left_side2 = make_color_array(len(plot_points_left_side2), (0,.9,.2))

    side2_labels_highlighted = np.concatenate((labels_left_side2, labels_right_side2), axis =0)

    

    make_plot(remove_mean(plot_points_side1), side1_labels_highlighted)
    make_plot(remove_mean(plot_points_side2), side2_labels_highlighted)

    labels_side1 = make_color_array(len(plot_points_side1),(.9,.2,0))
    labels_side2 = make_color_array(len(plot_points_side2),(0,.9,0.2))
    all_labels = np.concatenate((labels_side1, labels_side2), axis =0)
    make_plot(all_plot_points, all_labels)


    ##########################################
    ######### SAVE POINTS + LABELS ###########
    ##########################################

    if SAVE_PLEASE:

        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
            
        save_name_side1_left     = os.path.join(save_dir,'side1_points_left.npy')
        save_name_side1_right    = os.path.join(save_dir,'side1_points_right.npy')

        save_name_side2_left   = os.path.join(save_dir,'side2_points_left.npy')
        save_name_side2_right  = os.path.join(save_dir,'side2_points_right.npy')

        save_name_side1      = os.path.join(save_dir,'side1_points.npy')
        save_name_side2    = os.path.join(save_dir,'side2_points.npy')
        save_name_4cams     = os.path.join(save_dir,'4cams.npy')

        side1_left       = [plot_points_left_side1, labels_left_side1*255]
        side1_right      = [plot_points_right_side1, labels_right_side1*255]

        side2_left     = [plot_points_left_side2, labels_left_side2*255]
        side2_right    = [plot_points_right_side2, labels_right_side2*255]

        side1    = [plot_points_side1, labels_side1*255]
        side2  = [plot_points_side2, labels_side2*255]
        cams    = [all_plot_points, all_labels*255]

        np.save(save_name_side1_left, side1_left)
        np.save(save_name_side1_right, side1_right)

        np.save(save_name_side2_left, side2_left)
        np.save(save_name_side2_right, side2_right)

        np.save(save_name_side1, side1)
        np.save(save_name_side2, side2)
        np.save(save_name_4cams, cams)
