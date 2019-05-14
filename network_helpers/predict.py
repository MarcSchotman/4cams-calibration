import os,time,cv2, sys, math
import tensorflow as tf
import numpy as np

from network_helpers import network_utils, helpers
from network_helpers import model_builder



def project_prediction2PLY(points, rvec, tvec, intr_mtx, dist_coeff, img_label, image_w, image_h):

    #Can also return the visualization of rpojected points on the 2d camera plane
    #Uncomment the `return projection`

    rvec = np.array(rvec, np.float)
    tvec = np.array(tvec, np.float)
    intr_mtx = np.array(intr_mtx, np.float)
    dist_coeff = np.array(dist_coeff, np.float)
    
    imagepoints, _ = cv2.projectPoints(points, rvec, tvec, intr_mtx, dist_coeff)

    labels = []

    projection = np.zeros((image_h,image_w))


    for i in range(len(imagepoints)):

        x = int(round(imagepoints[i][0][0]))
        y = int(round(imagepoints[i][0][1]))

        if x >= image_w or y >= image_h:
            label = [0,0,0]

        else:
            projection[y,x] = 255

            label = np.flip(img_label[y,x,:])

        
        labels.append(label)

    return labels #, projection

def get_prediction(image, model_checkpoint, network_name, class_dict):

    class_names_list, label_values = helpers.get_label_info(os.path.abspath(class_dict))

    num_classes = len(label_values)

    #Rest 
    tf.reset_default_graph()

    # Initializing network
    config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    sess=tf.Session(config=config)

    net_input = tf.placeholder(tf.float32,shape=[None,None,None,3])
    net_output = tf.placeholder(tf.float32,shape=[None,None,None,num_classes]) 



    network, _ = model_builder.build_model(network_name, net_input=net_input,
                                            num_classes=num_classes,
                                            is_training=False)

    sess.run(tf.global_variables_initializer())

    print('Loading model checkpoint weights')
    saver=tf.train.Saver(max_to_keep=1000)
    saver.restore(sess,model_checkpoint)

    images_out = []
    images = []

    #checks what type of input is given
    #Handles:
    #   - [path1, path2]
    #   - [relative_path1, relative_path2]
    #   - path1
    #   - relative_path1
    #   - diretory with images

    if isinstance(image,list):
        for img in image:
            if os.path.isfile(os.path.abspath(img)) and (img.endswith("jpg") or img.endswith(".png")):
                images.append(os.path.abspath(img))
                continue
            else:
                raise Exception("ERROR: Provided images in list are either 1) not correct paths or 2) dont have *.jpg/*.png format...\n Given: %s"%(image))
    elif os.path.isfile(os.path.abspath(image)):
        #if single image is given:
        images = [os.path.abspath(image)]
        
    elif os.path.isdir(os.path.abspath(image)):
        #if directory is given with images
        
        files = os.listdir(os.path.abspath(image))
        for file in files:
            if file.endswith("jpg") or file.endswith("png"):
                images.append(os.path.join(os.path.abspath(image),file))
    else:
        print("ERROR: --image should be either an image path or directory with images")
        print("received --image ", image)
        sys.exit()

    #Do prediction
    for image in images:

        loaded_image = network_utils.load_image(image)
        # resized_image =cv2.resize(loaded_image, (args.crop_width, args.crop_height))
        input_image = np.expand_dims(np.float32(loaded_image),axis=0)/255.0

        st = time.time()
        output_image = sess.run(network,feed_dict={net_input:input_image})

        run_time = time.time()-st

        output_image = np.array(output_image[0,:,:,:])
        output_image = helpers.reverse_one_hot(output_image)

        out_vis_image = helpers.colour_code_segmentation(output_image, label_values)
        file_name = network_utils.filepath_to_name(image)
        # cv2.imwrite("%s_pred.png"%(file_name),cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))

        # print("Wrote image " + "%s_pred.png"%(file_name))

        images_out.append(cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))

    sess.close()


    return images_out


