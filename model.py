
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import tensorflow as tf
from tensorflow.contrib.quantize.python import quant_ops
from google.protobuf import text_format
import numpy as np
import imageio
 


class model:
    def read_pb(self):
        graph_def = None
        with tf.io.gfile.GFile(self._filepath,'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        return graph_def


    def read_pbtxt(self):
        graph_def = None
        with tf.io.gfile.GFile(self._filepath, 'r') as f:
            graph_def = tf.compat.v1.GraphDef()
            file_content = f.read()
            text_format.Merge(file_content, graph_def)
        return graph_def


    def __init__(self, filepath):
        self._filepath = filepath
        self._folder,tempfilename = os.path.split(filepath)
        self._filename,self._extension = os.path.splitext(tempfilename)
       
        if self._extension == ".pb":
            self._graph_def = self.read_pb()
        elif self._extension == ".pbtxt":
            self._graph_def = self.read_pbtxt()
        else:
            print("input file is not supported", filepath)

        assert(self._graph_def != None)


    def import_graph_def(self):
        tf.reset_default_graph()
        tf.import_graph_def(self._graph_def, name='')


    def write_pbtxt(self, is_quant=False):
        if is_quant:
            filename = self._filename + "_quant.pbtxt"
            tf.io.write_graph(self._quant_graph_def, './', filename, as_text=True)
        else:
            filename = self._filename + ".pbtxt"
            tf.io.write_graph(self._graph_def, './', filename, as_text=True)
        print("write pbtxt to %s" %filename)
    

    def write_pb(self):
        print("write pb to %s.pb" %self._filename)
        tf.io.write_graph(self._graph_def, './', self._filename+".pb", as_text=False)


    def write_summary(self):
        graph = tf.compat.v1.get_default_graph()
        print("write summary to log\nTo visilize model: python3 -m tensorboard.main --logdir=./log")
        summaryWriter = tf.compat.v1.summary.FileWriter('log/', graph)


    def inference(self, input_tensor_name_list, input_data_list, trace_tensor_name_list):
        #tf.import_graph_def(self._graph_def, name='')
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            
            trace_tensor_list = []
            for tensor_name in trace_tensor_name_list:
                trace_tensor_list.append(sess.graph.get_tensor_by_name(tensor_name))

            input_tensor_list = []
            for tensor_name in input_tensor_name_list:
                input_tensor_list.append(sess.graph.get_tensor_by_name(tensor_name))
            
            feed_dict = {}
            for input_tensor,input_data in zip(input_tensor_list, input_data_list):
                feed_dict[input_tensor] = input_data
            
            outputs = sess.run(trace_tensor_list,
                              feed_dict=feed_dict)

            if len(outputs) != len(trace_tensor_list):
                print("inference error")
                assert(0)

            return outputs        


    def get_conv2d_info(self):
        conv2d_node_dict = {}
        with tf.Session() as sess:
            for node in sess.graph_def.node:
                if node.op == "Conv2D":

                    if len(node.input) != 2:
                        raise Exception("unexpected input number")
                    weight_found = False
                    weight_idx = -1
                    for idx,input in enumerate(node.input):
                        if "weights/read" in input:
                            weight_found = True
                            weight_idx = idx
                    if weight_found:
                        conv2d_node_dict[node.name] = [node.input[weight_idx], node.input[1-weight_idx]]
                    else:
                        raise Exception("weight not found")
        

        return conv2d_node_dict


    def add_fake_quant(self, name):
       node = self._quant_graph_def.node.add()
       node.op = "FakeQuant" 
       node.name = name
       return node



    def quantize(self):
        self._quant_graph_def = tf.GraphDef()
        conv2d_node_dict = self.get_conv2d_info()
        with tf.Session() as sess:
            for node in sess.graph_def.node:
               if node.name in conv2d_node_dict.keys():
                   self.add_fake_quant(conv2d_node_dict[node.name][1]+"/quant_input")
                   self.add_fake_quant(conv2d_node_dict[node.name][0]+"/quant_weight")
                   new_node = self._quant_graph_def.node.add()
                   new_node.CopyFrom(node)

               else:
                   new_node = self._quant_graph_def.node.add()
                   new_node.CopyFrom(node)
                    
        print(conv2d_node_dict)



if __name__ == '__main__':
    input_model = sys.argv[1]
    #convert_pb_to_pbtxt(input_model)
    m = model(input_model)
    m.import_graph_def()
    #m.write_pbtxt()
    #m.write_summary()

    # insert add after postprobs
    #const_value = tf.constant([1.], name="fake_const")
    #target_tensor = m.get_tensor_by_name("postprobs:0")
    #tf.add(target_tensor, const_value, name="FakeAdd")
    #m._graph_def = tf.compat.v1.get_default_graph().as_graph_def()
     

    # test cifar
    trace_tensor_list = ["CifarNet/Predictions/Reshape_1:0"]
    input_tensor_list = ["Placeholder:0"]
    #im = imageio.imread("../../../data/tmp/images/test_49_6.png")
    im = imageio.imread(sys.argv[2])
    im = np.expand_dims(np.array(im), axis=0)
    input_data_list = [im]
    outputs = m.inference(input_tensor_list, input_data_list, trace_tensor_list)
    for tensor,data in zip(trace_tensor_list,outputs):
        print("%s\n%r" %(tensor, data))


    m.quantize()
    m.write_pbtxt(is_quant=True)

    
    


