# -*- coding: utf-8 -*-
"""
DeepDream

This technique is founded by Google engineer Alexander Mordvintsev.
The neural network find and increase patterns in images, thus creating a dream-like psychedelic appearance.

# Requirements
- https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip

# References
- [DeepDream Code](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/tutorials/deepdream)
- [Inceptionism](https://research.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html)
- [DeepDream - a code example for visualizing Neural Networks](https://ai.googleblog.com/2015/07/deepdream-code-example-for-visualizing.html)

@author: Christopher Masch
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import numpy as np
import PIL.Image
import tensorflow as tf

FLAGS = None

class DeepDream:
    
    def __init__(self):
        self.sess = tf.Session()
        self.load_model()
        self.resize = self.tffunc(np.float32, np.int32)(self.resize)
        
    def load_model(self):
        """Load the pretrained inception model"""
        print('Loading model:', FLAGS.model)
        try:
            with tf.gfile.FastGFile(FLAGS.model, 'rb') as f:
                self.graph_def = tf.GraphDef()
                self.graph_def.ParseFromString(f.read())
        except:
            print('Model not found, please download and exctract: %s' % 
                  'https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip')
        
        self.input_image = tf.placeholder(np.float32, name='input')
        imagenet_mean = 117.0
        input_processed = tf.expand_dims(self.input_image - imagenet_mean, 0)
        tf.import_graph_def(self.graph_def, {'input': input_processed})
        
        # List of Tensor's that are the output of convolutions
        graph = tf.get_default_graph()
        layers = [op.name for op in graph.get_operations() if op.type=='Conv2D' and 'import/' in op.name]
        feature_nums = [int(graph.get_tensor_by_name(name+':0').get_shape()[-1]) for name in layers]
        print('Number of layers:', len(layers))
        print('Total number of feature channels:', sum(feature_nums))
        if(FLAGS.list_layers):
            print('\n'.join(layers))
        
    def generate_visualizations(self):
        """Generating images"""
        
        # Gray image with some noise
        img_noise = np.random.uniform(size=(224,224,3)) + 100.0

        layer_channel = self.get_tensor(FLAGS.layer)[:,:,:, FLAGS.channel]
        print('##############\nUsing layer: %s\nChannel %i: %s' % (FLAGS.layer, FLAGS.channel, layer_channel))        
        
        # Inspecting one Channel
        print('\n*** Starting - Generating image of channel %i ***' % FLAGS.channel)
        image = self.multiscale(layer_channel, img_noise, FLAGS.nb_iterations, FLAGS.nb_octave)
        self.save_jpeg('multiscaler_viz_%s_%s.jpg'%(FLAGS.nb_iterations,FLAGS.nb_octave), image)
        
        
        # Combining two channels
        random_channel = np.random.randint(self.get_tensor(FLAGS.layer).shape[3])
        combined_channels = layer_channel + self.get_tensor(FLAGS.layer)[:,:,:,random_channel]
        print('\n*** Starting - Generating image of channels %i and %i ***' % (FLAGS.channel,random_channel))
        image = self.multiscale(combined_channels, img_noise, FLAGS.nb_iterations, FLAGS.nb_octave)
        self.save_jpeg('multiscaler_random_viz.jpg', image)
        
        
        # Deepdream
        if(FLAGS.input_img):
            print('\n*** Starting deepdream with %s ***' % FLAGS.input_img)
            img = PIL.Image.open(FLAGS.input_img)
            img = np.float32(img)
            # You can also use mixed5b etc. but mixed4c seems to be the best for generating interesting artifacts
            image = self.deepdream(tf.square(self.get_tensor('mixed4c')), img, FLAGS.nb_iterations, FLAGS.nb_octave) 
            self.save_jpeg('deepdream_viz_%s_%s.jpg' % (FLAGS.nb_iterations,FLAGS.nb_octave), self.normalize_image(image/255))
        
    
    def get_tensor(self, layer):
        """Getting output tensor for a given `layer`"""
        graph = tf.get_default_graph()
        return graph.get_tensor_by_name("import/%s:0"%layer)

    def normalize_image(self, image):
        """Stretch the range and prepare the image for saving as a JPEG"""
        image = np.clip(image, 0, 1)
        image = np.uint8(image * 255)
        return image

    def save_jpeg(self, jpeg_file, image):
        """Saving results as an image"""
        pil_image = PIL.Image.fromarray(image)
        pil_image.save(jpeg_file)
        print('\nImage saved: ', jpeg_file)
        
    def tffunc(self, *argtypes):
        """Helper that transforms TF-graph generating function into a regular one.
        See `resize` function."""
        placeholders = list(map(tf.placeholder, argtypes))
        def wrap(f):
            out = f(*placeholders)
            def wrapper(*args, **kw):
                return out.eval(dict(zip(placeholders, args)), session=self.sess)
            return wrapper
        return wrap

    def resize(self, img, size):
        """Resize an image"""
        img = tf.expand_dims(img, 0)
        return tf.image.resize_bilinear(img, size)[0,:,:,:]

    def calc_grad_tiled(self, img, t_grad, t_score, tile_size=512):
        """Compute the value of tensor t_grad over the image in a tiled way.
        Random shifts are applied to the image to blur tile boundaries over 
        multiple iterations."""
        sz = tile_size
        h, w = img.shape[:2]
        sx, sy = np.random.randint(sz, size=2)
        img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
        grad = np.zeros_like(img)
        for y in range(0, max(h-sz//2, sz),sz):
            for x in range(0, max(w-sz//2, sz),sz):
                sub = img_shift[y:y+sz,x:x+sz]
                g, s = self.sess.run([t_grad, t_score], {self.input_image:sub})
                grad[y:y+sz,x:x+sz] = g
        return np.roll(np.roll(grad, -sx, 1), -sy, 0), s
    
    def multiscale(self, t_obj, img0, nb_iter, nb_octave, step=1.0, octave_scale=1.4):
        """
        Generating an image of learned features of the model
        
        Arguments:
            t_obj        : Tensor object
            img          : Input image which will be affected by DeepDream
            nb_iter      : A higher value will result in more artifacts overlay the input image
            nb_octave    : Displaying higher / lower features. Higher value will generate more indicating figures.
            
        Returns:
            img          : New image created
        """
        t_score = tf.reduce_mean(t_obj)
        t_grad = tf.gradients(t_score, self.input_image)[0]
        img = img0.copy()
        self.printStatus(0, nb_octave*nb_iter)
        current_step = 0
        for octave in range(nb_octave):
            if octave>0:
                hw = np.float32(img.shape[:2])*octave_scale
                img = self.resize(img, np.int32(hw))
            for i in range(nb_iter):
                g, s = self.calc_grad_tiled(img, t_grad, t_score)
                g /= g.std() + 1e-8
                img += g * step
                current_step += 1
                self.printStatus(current_step, nb_octave*nb_iter)
        stddev = 0.1
        img = (img - img.mean()) / max(img.std(), 1e-4) * stddev + 0.5
        img = self.normalize_image(img)
        return img
    
    def deepdream(self, t_obj, img, nb_iter, nb_octave, step=1.5, octave_scale=1.4):
        """
        Creating a new image using an input image
        
        Arguments:
            t_obj        : Tensor object
            img          : Input image which will be affected by DeepDream
            nb_iter      : A higher value will result in more artifacts overlay the input image
            nb_octave    : Displaying higher / lower features. Higher value will generate more indicating figures.
            
        Returns:
            img          : New image created
        """
        t_score = tf.reduce_mean(t_obj)
        t_grad = tf.gradients(t_score, self.input_image)[0]

        octaves = []
        for i in range(nb_octave-1):
            hw = img.shape[:2]
            lo = self.resize(img, np.int32(np.float32(hw)/octave_scale))
            hi = img-self.resize(lo, hw)
            img = lo
            octaves.append(hi)

        self.printStatus(0, nb_octave * nb_iter)
        current_step = 0
        
        for octave in range(nb_octave):
            if octave>0:
                hi = octaves[-octave]
                img = self.resize(img, hi.shape[:2])+hi
            for i in range(nb_iter):
                g, s = self.calc_grad_tiled(img, t_grad, t_score)
                img += g * (step / (np.abs(g).mean()+1e-7))
                current_step += 1
                self.printStatus(current_step, nb_octave*nb_iter)
        return img
    
    def printStatus(self, current_step, total_steps, value=''):
        """
        Prints the current status of processing
        
        Arguments:
            current_step : actual step
            total_steps  : total steps
            value        : additional value to display at each step
        """
        percent = ("{:.1f}").format(100 *(current_step/float(total_steps)))
        print('\r%s%% %s %s'%(percent, 'Complete', value), end='\r')

def main(_):
    model = DeepDream()
    model.generate_visualizations()
    
def parse_args(parser):
    parser.add_argument('--model', default='./tensorflow_inception_graph.pb', type=str, help='Path to model (default: tensorflow_inception_graph.pb)')
    parser.add_argument('--list_layers', default=False, type=bool, help='List all layers of the loaded model (default: False)')
    parser.add_argument('--input_img', default='input.jpg', type=str, help='Image using for deepdream (default: input.jpg)')
    parser.add_argument('--channel', default=139, type=int, help='Feature channel to visualize (default: 139)')
    parser.add_argument('--nb_iterations', default=40, type=int, help='Contrast of artifacts (default:40)')
    parser.add_argument('--nb_octave', default=6, type=int, help='Controlls the size of artifacts (default:6)')
    parser.add_argument('--layer', default='mixed4d_3x3_bottleneck_pre_relu', type=str, help='Layer outputs.')
    return parser
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = parse_args(parser)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)