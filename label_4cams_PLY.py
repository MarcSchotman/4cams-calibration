from utils import *
from network_helpers.predict import *
from visualize_4cams_PLY import get_PLY
from run_icp import *


def label_PLY(points, prediction, calibration_folder):
    
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
    ###################### LINK PLY AND LABELS ###########################
    ######################################################################
    (h,w,_) =np.shape(prediction)

    rvec, _ = cv.Rodrigues(trans_world2color[0:3,0:3])
    tvec = np.array(trans_world2color[0:3,3])

    labels =  project_prediction2PLY(points, rvec, tvec, camera_matrix, dist_coeff, prediction, w, h)

    labels = np.array(labels, np.float) 
    print("\tLinked PLY and Neural Network prediction...")

    return labels


if __name__ == '__main__':
    
    #              // SIDE1 \\  // SIDE 2 \\
    cam_list=['cam1','cam2','cam3','cam4']
    ply_names = ['cam1_ply.npy','cam2_ply.npy', 'cam3_ply.npy', 'cam4_ply.npy']
    img_names = ['cam1_ply_image.jpg','cam2_ply_image.jpg', 'cam3_ply_image.jpg', 'cam4_ply_image.jpg']
    data_folder = "Data"
    crop_sides_PLY = True
    ##ICP##
    RUN_ICP = True
    max_iters = 20
    use_labels = True

    #SAVE RESULTS?
    SAVE_PLEASE = True
    save_dir = os.path.join(os.getcwd(), 'Data','RESULTS')

    root=os.getcwd()
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

    ######################################################################
    ####################### RUN NEURAL NETWORK ###########################
    ######################################################################
    #If weights not present: Download
    if not os.path.isfile(os.path.join(root,"Data", "NNModel","latest_model.ckpt.data-00000-of-00001")):
        download_model_weights()

    print("Running Neural Network on given images...")
    checkpoint_model = os.path.join(os.getcwd(),"Data","NNModel", "latest_model.ckpt")
    network_name = "DeepLabV3_plus"
    class_dict = os.path.join(os.getcwd(),"Data","NNModel", "class_dict.csv")

    image_path = [side1_right_IMG, side1_left_IMG, side2_right_IMG, side2_left_IMG]

    predictions = get_prediction(image_path, checkpoint_model, network_name, class_dict)

    prediction_side1_right = predictions[0]
    prediction_side1_left = predictions[1]   
    prediction_side2_right = predictions[2]   
    prediction_side2_left = predictions[3]   

    cv2.imwrite("%s_pred.jpg"%(cam_list[0]),prediction_side1_right)
    cv2.imwrite("%s_pred.jpg"%(cam_list[1]),prediction_side1_left)
    cv2.imwrite("%s_pred.jpg"%(cam_list[2]),prediction_side2_right)
    cv2.imwrite("%s_pred.jpg"%(cam_list[3]),prediction_side2_left)


    #GET PLY POINTS
    print("Processing PLY files...")
    plot_points_right_side1 = get_PLY(side1_right_PLY, side1_right_calibration)
    plot_points_left_side1 = get_PLY(side1_left_PLY, side1_left_calibration)

    plot_points_right_side2 = get_PLY(side2_right_PLY, side2_right_calibration)
    plot_points_left_side2 = get_PLY(side2_left_PLY, side2_left_calibration)

     #Here some specific default crop settings are used
    if crop_sides_PLY:
        plot_points_right_side1 = crop_default(plot_points_right_side1, 'right')
        plot_points_left_side1 = crop_default(plot_points_left_side1, 'left')
        plot_points_right_side2 = crop_default(plot_points_right_side2, 'right')
        plot_points_left_side2 = crop_default(plot_points_left_side2, 'left')


    print("projecting predicted labels on PLY...")
    labels_right_side1 = label_PLY(plot_points_right_side1, prediction_side1_right, side1_right_calibration)
    labels_left_side1 = label_PLY(plot_points_left_side1, prediction_side1_left, side1_left_calibration)
    labels_right_side2 = label_PLY(plot_points_right_side2, prediction_side2_right, side2_right_calibration)
    labels_left_side2 = label_PLY(plot_points_left_side2, prediction_side2_left, side2_left_calibration)


    ######################################################################
    ############################## RUN ICP ###############################
    ######################################################################
    if RUN_ICP:
        #Makes the labels one hot encoded and then runs the icp algorithm on each side seperately and then using the results of the two sides together
        class_dict ='./Data/NNModel/class_dict.csv'
        
        one_hot_labels_right_side1 = convert_label_to_one_hot(labels_right_side1, class_dict)
        one_hot_labels_left_side1 = convert_label_to_one_hot(labels_left_side1, class_dict)

        one_hot_labels_right_side2 = convert_label_to_one_hot(labels_right_side2, class_dict)
        one_hot_labels_left_side2 = convert_label_to_one_hot(labels_left_side2, class_dict)

        one_hot_labels_side1 = np.concatenate((one_hot_labels_right_side1, one_hot_labels_left_side1))
        one_hot_labels_side2 = np.concatenate((one_hot_labels_right_side2, one_hot_labels_left_side2))

        print("Running ICP with side1 cameras!\n \tLeft cam: %s points\n\tRight cam: %s points"%(len(plot_points_right_side1), len(plot_points_left_side1)))
        plot_points_right_side1, plot_points_left_side1, error = run_icp(plot_points_right_side1, plot_points_left_side1, 
                                                    use_labels = use_labels,
                                                    max_iters = max_iters, 
                                                    one_hot_labels_left = one_hot_labels_right_side1, 
                                                    one_hot_labels_right = one_hot_labels_left_side1)
        print("Running ICP with side2 cameras!\n \tLeft cam: %s points\n\tRight cam: %s points"%(len(plot_points_right_side2), len(plot_points_left_side2)))
        plot_points_right_side2, plot_points_left_side2, error = run_icp(plot_points_right_side2, plot_points_left_side2, 
                                                    use_labels = use_labels,
                                                    max_iters = max_iters, 
                                                    one_hot_labels_left = one_hot_labels_right_side2, 
                                                    one_hot_labels_right = one_hot_labels_left_side2)

        plot_points_side1 = np.concatenate((plot_points_right_side1, plot_points_left_side1), axis =0)
        plot_points_side2 = np.concatenate((plot_points_right_side2, plot_points_left_side2), axis =0) 
        plot_points_side2_with_1 = concatenate_XYZ1(plot_points_side2[:,0], plot_points_side2[:,1], plot_points_side2[:,2])
        plot_points_side2_in_frame_side1 = points_side2side(plot_points_side2_with_1)    

        print("Running ICP with boh sides cameras!\n \tLeft cam: %s points\n\tRight cam: %s points"%(len(plot_points_side1), len(plot_points_side2)))
        plot_points_side1, plot_points_side2_in_frame_side1, error = run_icp(plot_points_side1, plot_points_side2_in_frame_side1, 
                                                    use_labels = use_labels,
                                                    max_iters = max_iters, 
                                                    one_hot_labels_left = one_hot_labels_side1, 
                                                    one_hot_labels_right = one_hot_labels_side2)

        all_plot_points = np.concatenate((plot_points_side1, plot_points_side2_in_frame_side1), axis =0)
        all_plot_points = remove_mean(all_plot_points)

    else:
        #combine without ICP
        plot_points_side1 = np.concatenate((plot_points_right_side1, plot_points_left_side1), axis =0)
        plot_points_side2 = np.concatenate((plot_points_right_side2, plot_points_left_side2), axis =0) 
        plot_points_side2_with_1 = concatenate_XYZ1(plot_points_side2[:,0], plot_points_side2[:,1], plot_points_side2[:,2])
        plot_points_side2_in_frame_side1 = points_side2side(plot_points_side2_with_1)        

        all_plot_points = np.concatenate((plot_points_side1, plot_points_side2_in_frame_side1), axis =0)
        all_plot_points = remove_mean(all_plot_points)

    ######################################################################
    ############################ MAKE PLOTS ##############################
    ######################################################################
    #Concatenates all the points and labels to make the plots
    #BRIGHT LABELS ARE: RIGHT CAMERA OR SIDE1:

    labels_side1_highlighted = np.concatenate((labels_right_side1*2, labels_left_side1), axis =0)
    labels_side2_highlighted= np.concatenate((labels_right_side2*2, labels_left_side2), axis =0)            

    #label_side1 before is highlighting the left camera, undoing this and high;ighting the whole of side1
    labels_side1 =np.concatenate((labels_right_side1, labels_left_side1), axis =0)
    labels_side2 =np.concatenate((labels_right_side2, labels_left_side2), axis =0)
    all_labels = np.concatenate((labels_side1*2, labels_side2), axis =0)


    make_plot(plot_points_side1, labels_side1_highlighted/255)
    make_plot(plot_points_side2, labels_side2_highlighted/255)
    make_plot(all_plot_points, all_labels/255)

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

        side1_left       = [plot_points_left_side1, labels_left_side1]
        side1_right      = [plot_points_right_side1, labels_right_side1]

        side2_left     = [plot_points_left_side2, labels_left_side2]
        side2_right    = [plot_points_right_side2, labels_right_side2]

        side1    = [plot_points_side1, labels_side1]
        side2  = [plot_points_side2, labels_side2]
        cams    = [all_plot_points, all_labels]

        np.save(save_name_side1_left, side1_left)
        np.save(save_name_side1_right, side1_right)

        np.save(save_name_side2_left, side2_left)
        np.save(save_name_side2_right, side2_right)

        np.save(save_name_side1, side1)
        np.save(save_name_side2, side2)
        np.save(save_name_4cams, cams)

