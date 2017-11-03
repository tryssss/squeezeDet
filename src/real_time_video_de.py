"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from imutils.video import VideoStream
from imutils.video import FPS

import imutils
import time
import cv2

from demo import image_demo

# initialize the video stream, allow the camera sensor to warmup
print("Starting video stream...")
vs = VideoStream(src=1).start()
time.sleep(2.0)
fps = FPS().start()

# loop over the frames from the threaded video stream and resize it 
while True:
    # Read and resize
    frame = vs.read()
    #frame = imutils.resize(frame, width=, )
    (h, w) = frame.shape[:2]

    # deal with this frame


    # SHOW
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # update the FPS counter
    fps.update()

# stop the timer and display FPS information
fps.stop()
print("elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps))

cv2.destroyAllWindows()
vs.stop()
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import time
import sys
import os
import glob

import numpy as np
import tensorflow as tf

from config import *
from train import _draw_box
from nets import *

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'mode', 'video', """'image' or 'video'.""")
tf.app.flags.DEFINE_string(
    'checkpoint', './data/model_checkpoints/squeezeDet/model.ckpt-87000',
    """Path to the model parameter file.""")
tf.app.flags.DEFINE_string(
    'input_path', './data/sample.mp4',
    """Input image or video to be detected. Can process glob input such as """
    """./data/00000*.png.""")
tf.app.flags.DEFINE_string(
    'out_dir', './data/out/', """Directory to dump output image or video.""")
tf.app.flags.DEFINE_string(
    'demo_net', 'squeezeDet', """Neural net architecture.""")


def video_demo():
  """Detect videos."""

  #cap = cv2.VideoCapture(FLAGS.input_path)
  cap = cv2.VideoCapture()#IP address
  #cap = cv2.VideoCapture(0)


  assert FLAGS.demo_net == 'squeezeDet' or FLAGS.demo_net == 'squeezeDet+', \
      'Selected nueral net architecture not supported: {}'.format(FLAGS.demo_net)

  with tf.Graph().as_default():
    # Load model
    if FLAGS.demo_net == 'squeezeDet':
      mc = kitti_squeezeDet_config()
      mc.BATCH_SIZE = 1
      # model parameters will be restored from checkpoint
      mc.LOAD_PRETRAINED_MODEL = False
      model = SqueezeDet(mc, FLAGS.gpu)
    elif FLAGS.demo_net == 'squeezeDet+':
      mc = kitti_squeezeDetPlus_config()
      mc.BATCH_SIZE = 1
      mc.LOAD_PRETRAINED_MODEL = False
      model = SqueezeDetPlus(mc, FLAGS.gpu)

    saver = tf.train.Saver(model.model_params)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
      saver.restore(sess, FLAGS.checkpoint)

      times = {}
      count = 0
      while cap.isOpened():
        t_start = time.time()
        count += 1
        out_im_name = os.path.join(FLAGS.out_dir, str(count).zfill(6)+'.jpg')

        # Load images from video and crop
        ret, frame = cap.read()
        cv2.imshow("input", frame)
        if ret==True:
          # crop frames
          frame = cv2.resize(frame, (mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT))
          im_input = frame.astype(np.float32) - mc.BGR_MEANS
        else:
          break

        t_reshape = time.time()
        times['reshape']= t_reshape - t_start

        # Detect
        det_boxes, det_probs, det_class = sess.run(
            [model.det_boxes, model.det_probs, model.det_class],
            feed_dict={model.image_input:[im_input]})

        t_detect = time.time()
        times['detect']= t_detect - t_reshape
        
        # Filter
        final_boxes, final_probs, final_class = model.filter_prediction(
            det_boxes[0], det_probs[0], det_class[0])

        keep_idx    = [idx for idx in range(len(final_probs)) \
                          if final_probs[idx] > mc.PLOT_PROB_THRESH]
        final_boxes = [final_boxes[idx] for idx in keep_idx]
        final_probs = [final_probs[idx] for idx in keep_idx]
        final_class = [final_class[idx] for idx in keep_idx]

        t_filter = time.time()
        times['filter']= t_filter - t_detect

        # Draw boxes

        # TODO(bichen): move this color dict to configuration file, blue, yellow, purple
        cls2clr = {
            'car': (255, 191, 0),
            'cyclist': (0, 191, 255),
            'pedestrian':(255, 0, 191)
        }
        _draw_box(
            frame, final_boxes,
            [mc.CLASS_NAMES[idx]+': (%.2f)'% prob \
                for idx, prob in zip(final_class, final_probs)],
            cdict=cls2clr
        )

        t_draw = time.time()
        times['draw']= t_draw - t_filter

        #cv2.imwrite(out_im_name, frame)
        cv2.imshow("output", frame)
        # out.write(frame)

        times['total']= time.time() - t_start

        # time_str = ''
        # for t in times:
        #   time_str += '{} time: {:.4f} '.format(t[0], t[1])
        # time_str += '\n'
        time_str = 'Total time: {:.4f}, detection time: {:.4f}, filter time: '\
                   '{:.4f}'. \
            format(times['total'], times['detect'], times['filter'])

        print (time_str)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
  # Release everything if job is finished
  cap.release()
  # out.release()
  cv2.destroyAllWindows()

def main(argv=None):
  if not tf.gfile.Exists(FLAGS.out_dir):
    tf.gfile.MakeDirs(FLAGS.out_dir)
  if FLAGS.mode == 'video':
    video_demo()

if __name__ == '__main__':
    tf.app.run()
