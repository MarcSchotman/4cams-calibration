3d-Calibration-Stand-Alone
==================

This repo is a simplified version of the already existing 3d calibraion repo. It is a showcase for 4 things:

1) Calibrating cameras using the chessboard image with the camera intrinsics
2) Combining the pointcloud of the different cameras using the calibration files.
3) Using ICP to enhance the acquired combinination of pointclouds
4) Projecting the labels of a Neural Network on the pointcloud


**Set up**

The set up used here is 4 depth+RGB-cameras, 2 on each side of the object (a vine plant).
The Neural Network here is trained on a reasonable dataset containing labels void, trunc, branch and bud. 


In the `Data` folder you can find 5 folders, `cam1` upto `cam4` and `NNModel`.
The cam folders containg the following:
- A calibration image, an image of the chessboard. Keep in mind that the calibration images have to be of the same chessboard at EXACTLY the same place. 
- ```camera_matrix.npy``` = The [camera matrix](https://en.wikipedia.org/wiki/Camera_matrix)
- ```dist_coeff.npy``` = The [distortion coefficients](https://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration/camera_calibration.html)
- ```trans_depth2color.npy``` = The homogeneous transfer matrix between the RGB-camera and the depth-camera

The `NNModel` folder contains the following:
- class_dict.csv  = A dictionary telling the neural network which rgb value to use for which label.
- checkpoint = Checkpoint infor used by tensorflow

The model weights will be automtically downloaded from my Google Drive. This code can be checked in ```utils.py -> download_model_weights()```

**Run example**

1) Run ```python calibrate.py``` to calculate the homogeneous transfer matrices from the cameras to the world frame (which is located on the chessboard). The following transfer matrices are saved in Data/camX/intrinsics:

- `trans_world2color.npy`
- `trans_color2world.npy`

2) Run ```python visualize_4cams_PLY.py``` or ```python label_4cams_PLY.py``` to visualize the pointclouds with or without predicted labels from the network, respectively. The setup of both scripts are similar. At the top of ```"__main__"``` one can change some parameters indicating wheter to save the output/use_icp/paths_to_pointclouds.

