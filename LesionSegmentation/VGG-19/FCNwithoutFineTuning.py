'''
Code modifided from Shaw's work at https://github.com/anonymoussss/FCN_pretrained
'''
from __future__ import print_function
import tensorflow as tf
import numpy as np

from scipy import ndimage
from scipy.spatial.distance import euclidean
import TensorflowUtils as utils
import read_Data_list as scene_parsing
import BatchDatsetReader as dataset
from six.moves import xrange
import sys

#import BatchTestReader as testset

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "2", "batch size for training")
tf.flags.DEFINE_integer("validation_batch_size","1","batch size for training")
tf.flags.DEFINE_string("logs_dir", "logs/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "Data/", "path to dataset")

tf.flags.DEFINE_string("test_dir", "testData/", "path to dataset")

tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "Model/", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")

#the parameters of froc
nbr_of_thresholds = 5
range_threshold = [0.125, 0.25, 0.5, 0.75, 0.875]
allowedDistance = 5

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

#MAX_ITERATION = int(5e4 + 1)
MAX_ITERATION = 1
NUM_OF_CLASSESS = 96
IMAGE_SIZE = 224


def vgg_net(weights, image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    net = {}
    current = image
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
            bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
            current = utils.conv2d_basic(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
            if FLAGS.debug:
                utils.add_activation_summary(current)
        elif kind == 'pool':
            current = utils.avg_pool_2x2(current)
        net[name] = current

    return net


def inference(image, keep_prob):
    """
    Semantic segmentation network definition
    :param image: input image. Should have values in range 0-255
    :param keep_prob:
    :return:
    """
    print("setting up vgg initialized conv layers ...")
    model_data = utils.get_model_data(FLAGS.model_dir, MODEL_URL)

    mean = model_data['normalization'][0][0][0] #224*224*3
    mean_pixel = np.mean(mean, axis=(0, 1)) #return a mean pixel (3,)

    weights = np.squeeze(model_data['layers'])

    processed_image = utils.process_image(image, mean_pixel) #centerlize the images

    with tf.variable_scope("inference"):
        image_net = vgg_net(weights, processed_image)
        conv_final_layer = image_net["conv5_4"] #change the original conv5_3

        VGG_stopped = tf.stop_gradient(conv_final_layer) #stop the BP

        pool5 = utils.max_pool_2x2(VGG_stopped) 

        W6 = utils.weight_variable([7, 7, 512, 4096], name="W6")
        b6 = utils.bias_variable([4096], name="b6")
        conv6 = utils.conv2d_basic(pool5, W6, b6)
        relu6 = tf.nn.relu(conv6, name="relu6")
        if FLAGS.debug:
            utils.add_activation_summary(relu6)
        relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)

        W7 = utils.weight_variable([1, 1, 4096, 4096], name="W7")
        b7 = utils.bias_variable([4096], name="b7")
        conv7 = utils.conv2d_basic(relu_dropout6, W7, b7)
        relu7 = tf.nn.relu(conv7, name="relu7")
        if FLAGS.debug:
            utils.add_activation_summary(relu7)
        relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)

        W8 = utils.weight_variable([1, 1, 4096, NUM_OF_CLASSESS], name="W8")
        b8 = utils.bias_variable([NUM_OF_CLASSESS], name="b8")
        conv8 = utils.conv2d_basic(relu_dropout7, W8, b8)
        # annotation_pred1 = tf.argmax(conv8, dimension=3, name="prediction1")

        # now to upscale to actual image size
        deconv_shape1 = image_net["pool4"].get_shape()
        W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, NUM_OF_CLASSESS], name="W_t1")
        b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
        conv_t1 = utils.conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(image_net["pool4"]))
        fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")

        deconv_shape2 = image_net["pool3"].get_shape()
        W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
        b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
        conv_t2 = utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(image_net["pool3"]))
        fuse_2 = tf.add(conv_t2, image_net["pool3"], name="fuse_2")

        shape = tf.shape(image)
        deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], NUM_OF_CLASSESS])
        W_t3 = utils.weight_variable([16, 16, NUM_OF_CLASSESS, deconv_shape2[3].value], name="W_t3")
        b_t3 = utils.bias_variable([NUM_OF_CLASSESS], name="b_t3")
        conv_t3 = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)

        annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")

    return tf.expand_dims(annotation_pred, dim=3), conv_t3


def train(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    if FLAGS.debug:
        # print(len(var_list))
        for grad, var in grads:
            utils.add_gradient_summary(grad, var)
    return optimizer.apply_gradients(grads)

#FROC score
def computeConfMatElements(thresholded_proba_map, ground_truth, allowedDistance):
    
    if allowedDistance == 0 and type(ground_truth) == np.ndarray:
        P = np.count_nonzero(ground_truth)
        TP = np.count_nonzero(thresholded_proba_map*ground_truth)
        FP = np.count_nonzero(thresholded_proba_map - (thresholded_proba_map*ground_truth))    
    else:
    
        #reformat ground truth to a list  
        if type(ground_truth) == np.ndarray:
            #convert ground truth binary map to list of coordinates
            labels, num_features = ndimage.label(ground_truth)
            list_gt = ndimage.measurements.center_of_mass(ground_truth, labels, range(1,num_features+1))   
        elif type(ground_truth) == list:        
            list_gt = ground_truth        
        else:
            raise ValueError('ground_truth should be either of type list or ndarray and is of type ' + str(type(ground_truth)))
        
        #reformat thresholded_proba_map to a list
        labels, num_features = ndimage.label(thresholded_proba_map)
        list_proba_map = ndimage.measurements.center_of_mass(thresholded_proba_map, labels, range(1,num_features+1)) 
         
        #compute P, TP and FP  
        FP = 0
        TP = 0
        P = len(list_gt) #P=TP+FN
        #compute FP
        for point_pm in list_proba_map:
            found = False
            for point_gt in list_gt:                           
                if euclidean(point_pm,point_gt) < allowedDistance:
                    found = True
                    break
            if found == False:
                FP += 1
        #compute TP
        for point_gt in list_gt:
            for point_pm in list_proba_map:                           
                if euclidean(point_pm,point_gt) < allowedDistance:
                    TP += 1
                    break
                                 
    return P,TP,FP
    
def computeFROC(proba_map, ground_truth, allowedDistance, threshold_list):
    #INPUTS
    #proba_map : numpy array of dimension [number of image, xdim, ydim,...], values preferably in [0,1]
    #ground_truth: numpy array of dimension [number of image, xdim, ydim,...], values in {0,1}; or list of coordinates
    #allowedDistance: Integer. euclidian distance distance in pixels to consider a detection as valid (anisotropy not considered in the implementation)  
    #threshhold_list: list. list of thresholds 
    #OUTPUTS
    #sensitivity_list_treshold: list of average sensitivy over the set of images for increasing thresholds
    #FPavg_list_treshold: list of average FP over the set of images for increasing thresholds
    
    #verify that proba_map and ground_truth have the same shape
    if proba_map.shape != ground_truth.shape:
        raise ValueError('Error. Proba map and ground truth have different shapes.')
        
    #rescale ground truth and proba map between 0 and 1
    proba_map = proba_map.astype(np.float32)
    proba_map = (proba_map - np.min(proba_map)) / (np.max(proba_map) - np.min(proba_map))
    if type(ground_truth) == np.ndarray:
        ground_truth = ground_truth.astype(np.float32)    
        ground_truth = (ground_truth - np.min(ground_truth)) / (np.max(ground_truth) - np.min(ground_truth))
    
    sensitivity_list_treshold = []
    FPavg_list_treshold = []
    #loop over thresholds
    for threshold in threshold_list:
        sensitivity_list_proba_map = []
        FP_list_proba_map = []
        #loop over proba map
        for i in range(len(proba_map)):

            #threshold the proba map
            thresholded_proba_map = np.zeros(np.shape(proba_map[i]))
            thresholded_proba_map[proba_map[i] >= threshold] = 1
            #print(np.shape(thresholded_proba_map)) #(400,640)
            
            #save proba maps
            #imageio.imwrite('thresholded_proba_map_'+str(threshold)+'.png', thresholded_proba_map)                   
                   
            #compute P, TP, and FP for this threshold and this proba map
            P,TP,FP = computeConfMatElements(thresholded_proba_map, ground_truth[i], allowedDistance)       
            
            #append results to list
            FP_list_proba_map.append(FP)
            
            #check that ground truth contains at least one positive
            if (type(ground_truth) == np.ndarray and np.count_nonzero(ground_truth) > 0) or (type(ground_truth) == list and len(ground_truth) > 0):
                sensitivity_list_proba_map.append(TP*1./P)
            
        
        #average sensitivity and FP over the proba map, for a given threshold
        sensitivity_list_treshold.append(np.mean(sensitivity_list_proba_map))
        FPavg_list_treshold.append(np.mean(FP_list_proba_map))    
        
    return sensitivity_list_treshold, FPavg_list_treshold

def get_FROC_avg_score(sensitivity_list_treshold,nbr_of_thresholds):
    '''
    arg:
        sensitivity_list_treshold:the list of sensitivity_list_tresholds by compute
        nbr_of_threshold:the number of thresholds
    '''
    print('sensitivity_list_treshold:',sensitivity_list_treshold)
    froc_total_score = 0.0
    for item in sensitivity_list_treshold:
        froc_total_score += item
    return froc_total_score*1.0/nbr_of_thresholds
    
def main(argv=None):
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name="input_image")
    annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name="annotation")

    pred_annotation, logits = inference(image, keep_probability)
    
    tf.summary.image("input_image", image, max_outputs=2)
    tf.summary.image("ground_truth", tf.cast(annotation, tf.uint8), max_outputs=2)
    tf.summary.image("pred_annotation", tf.cast(pred_annotation, tf.uint8), max_outputs=2)
    loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                          labels=tf.squeeze(annotation, squeeze_dims=[3]),
                                                                        name="entropy")))
    #if set the classes num=2,sparse_softmax_cross_entropy_with_logits will be wrong!
    tf.summary.scalar("training_entropy_loss", loss)

    trainable_var = tf.trainable_variables()
    if FLAGS.debug:
        for var in trainable_var:
            utils.add_to_regularization_and_summary(var)
    train_op = train(loss, trainable_var)

    print("Setting up summary op...")
    summary_op = tf.summary.merge_all()


    
    # For Training
    print("Setting up image reader...")
    train_records, valid_records = scene_parsing.read_dataset(FLAGS.data_dir) #get read lists
    print(len(train_records)) #44
    print(len(valid_records)) #10
	
    '''
    # For Testing
    print("Setting up testing image reader...")
    train_records, valid_records = scene_parsing.read_dataset(FLAGS.test_dir) #get read lists
    print(len(train_records)) #44
    print(len(valid_records)) #10
    '''


    print("Setting up dataset reader")
    image_options = {'resize': True, 'resize_size': IMAGE_SIZE} #resize all your images

    #if train mode,get datas batch by bactch
    if FLAGS.mode == 'train':
        train_dataset_reader = dataset.BatchDatset(train_records, image_options)
    
    validation_dataset_reader = dataset.BatchDatset(valid_records, image_options)

    sess = tf.Session()

    print("Setting up Saver...")
    saver = tf.train.Saver(max_to_keep=2)
    summary_writer = tf.summary.FileWriter(FLAGS.logs_dir, sess.graph)

    sess.run(tf.global_variables_initializer())
    
    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)#if model has been trained,restore it
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)   
        print("Model restored...")

    if FLAGS.mode == "train":
        for itr in xrange(MAX_ITERATION):
            train_images, train_annotations = train_dataset_reader.next_batch(FLAGS.batch_size)
            #print(train_images.shape)
            #print(train_annotations.shape)
            #print(itr)
            feed_dict = {image: train_images, annotation: train_annotations, keep_probability: 0.85}
            sess.run(train_op, feed_dict=feed_dict)

            if itr % 15 == 0:
                valid_images, valid_annotations = validation_dataset_reader.next_batch(FLAGS.validation_batch_size)
                
                train_loss, summary_str = sess.run([loss, summary_op], feed_dict=feed_dict)
                valid_loss = sess.run(loss, feed_dict={image: valid_images, annotation: valid_annotations,
                                                       keep_probability: 1.0})
                pre_train_image = sess.run(pred_annotation, feed_dict={image: train_images, keep_probability: 1.0})
                pre_valid_image = sess.run(pred_annotation, feed_dict={image: valid_images, keep_probability: 1.0})
                
                sensitivity_list_t, FPavg_list_t = computeFROC(pre_train_image,train_annotations, allowedDistance, range_threshold)
                froc_score_t = get_FROC_avg_score(sensitivity_list_t,nbr_of_thresholds)
                #f1_score = metrics.f1_score(valid_annotations_flat, pre_valid_image_flat) 
                sensitivity_list, FPavg_list = computeFROC(pre_valid_image,valid_annotations, allowedDistance, range_threshold)
                froc_score = get_FROC_avg_score(sensitivity_list,nbr_of_thresholds)
                
                #SN_score_tb = tf.Summary(value = [tf.Summary.Value(tag="f1_score", simple_value=f1_score)])
                froc_score_t_tb = tf.Summary(value = [tf.Summary.Value(tag="froc_score_training", simple_value=froc_score_t)])
                froc_score_tb = tf.Summary(value = [tf.Summary.Value(tag="froc_score_validation", simple_value=froc_score)])
                validation_loss = tf.Summary(value = [tf.Summary.Value(tag="validation_loss", simple_value=valid_loss)])
                print('froc_score_traing:',froc_score_t)
                print('froc_score:',froc_score)
                
                #summary_writer.add_summary(SN_score_tb,itr)
                summary_writer.add_summary(summary_str, itr)
                summary_writer.add_summary(froc_score_t_tb,itr)
                summary_writer.add_summary(froc_score_tb,itr)
                summary_writer.add_summary(validation_loss, itr)
                summary_writer.flush()
                
                print("Step: %d, learning_rate:%g" % (itr, FLAGS.learning_rate))
                print("Step: %d, Train_loss:%g" % (itr, train_loss))
                print("Step: %d, Validation_loss:%g" % (itr, valid_loss))
                sys.stdout.flush()
            
            #if itr % 5000 == 0:
            #if itr % 2000 == 0:
            saver.save(sess, FLAGS.logs_dir + "model.ckpt", itr)
            sys.stdout.flush()
            '''
            if itr == 30000:
                FLAGS.learning_rate = 1e-6
            if itr == 40000:
                FLAGS.learning_rate = 1e-7
            '''
            
    elif FLAGS.mode == "visualize":
        valid_images, valid_annotations = validation_dataset_reader.get_random_batch(FLAGS.validation_batch_size)
        pred = sess.run(pred_annotation, feed_dict={image: valid_images, annotation: valid_annotations,
                                                    keep_probability: 1.0})
        valid_annotations = np.squeeze(valid_annotations, axis=3)
        pred = np.squeeze(pred, axis=3)

        for itr in range(FLAGS.validation_batch_size):
            utils.save_image(valid_images[itr].astype(np.uint8), FLAGS.logs_dir+'images/', name="inp_" + str(1+itr))
            utils.save_image(valid_annotations[itr].astype(np.uint8), FLAGS.logs_dir+'images/', name="gt_" + str(1+itr))
            utils.save_image(pred[itr].astype(np.uint8), FLAGS.logs_dir+'images/', name="pred_" + str(1+itr))
            print("Saved image: %d" % itr)
            sys.stdout.flush()
    
    else: # FLAGS.mode == "test":
        test_images, test_annotations = validation_dataset_reader.get_random_batch(FLAGS.validation_batch_size)
        pred = sess.run(pred_annotation, feed_dict={image: test_images, annotation: test_annotations,
                                                    keep_probability: 1.0})
        test_annotations = np.squeeze(test_annotations, axis=3)
        pred = np.squeeze(pred, axis=3)

        for itr in range(FLAGS.validation_batch_size):
            utils.save_image(test_images[itr].astype(np.uint8), FLAGS.logs_dir+'images/', name="inp_" + str(1+itr))
            utils.save_image(test_annotations[itr].astype(np.uint8), FLAGS.logs_dir+'images/', name="gt_" + str(1+itr))
            utils.save_image(pred[itr].astype(np.uint8), FLAGS.logs_dir+'images/', name="pred_" + str(1+itr))
            print("Saved image: %d" % itr)
            sys.stdout.flush()
	


if __name__ == "__main__":
    tf.app.run()
