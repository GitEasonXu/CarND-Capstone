from styx_msgs.msg import TrafficLight
import os
import tensorflow as tf
import numpy as np
import rospy
import cv2
class TLClassifier(object):
    def __init__(self, case):
        #TODO load classifier
        self.case = case
        self.current_light = TrafficLight.UNKNOWN
        self.category_index = {1: {'id': 1, 'name': 'Green'}, 2: {'id': 2, 'name': 'Red'},
                               3: {'id': 3, 'name': 'Yellow'}, 4: {'id': 4, 'name': 'off'}}
        self.min_score_thresh = 0.5

        cwd = os.path.dirname(os.path.realpath(__file__))
        if self.case == 'sim': 
            model_fname = cwd + '/model/sim_frozen_inference_graph.pb'
        else:
            model_fname = cwd + '/model/real_frozen_inference_graph.pb'
        
        rospy.logwarn("model_path={}".format(model_fname))
        # load frozen tensorflow model
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_fname, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        # create tensorflow session for detection
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.detection_graph, config=config)
        # end
        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')



    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        (im_width, im_height, _) = image_rgb.shape
        image_np = np.expand_dims(image_rgb, axis=0)
        #print "Image shape: {}".format(image_np.shape)

        with self.detection_graph.as_default():
            (boxes, scores, classes, num) = self.sess.run([self.detection_boxes, self.detection_scores,self.detection_classes, self.num_detections],
                                                          feed_dict={self.image_tensor: image_np})

        '''
        print "boxes: {}".format(np.squeeze(boxes))
        print "scores: {}".format(np.squeeze(scores))
        print "classes: {}".format(np.squeeze(classes))
        print "num: {}".format(np.squeeze(num))
        '''
        boxes_sque = np.squeeze(boxes)
        scores_sque = np.squeeze(scores)
        classes_sque = np.squeeze(classes)
        num_sque = np.squeeze(num)
        
        count_red = 0
        count_green = 0
        for i in range(num_sque):
            if scores_sque[i] > self.min_score_thresh:
                class_name = self.category_index[classes_sque[i]]['name']

                # Traffic light thing
                if class_name == 'Red':
                    count_red += 1
                elif class_name == 'Green':
                    count_green += 1
             
        if count_green > count_red:
            #print "Light: green"
            self.current_light = TrafficLight.GREEN
        else:
            #print "Light: red"
            self.current_light = TrafficLight.RED

        return self.current_light
