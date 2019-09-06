
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import tensorflow as tf
from google.protobuf import text_format
 


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


    def write_pbtxt(self):
        print("write pbtxt to %s.pbtxt" %self._filename)
        tf.io.write_graph(self._graph_def, './', self._filename+".pbtxt", as_text=True)
    

    def write_pb(self):
        print("write pb to %s.pb" %self._filename)
        tf.io.write_graph(self._graph_def, './', self._filename+".pb", as_text=False)


    def write_summary(self):
        graph = tf.compat.v1.get_default_graph()
        tf.import_graph_def(self._graph_def, name='graph')
        print("write summary to log\nTo visilize model: python3 -m tensorboard.main --logdir=./log")
        summaryWriter = tf.compat.v1.summary.FileWriter('log/', graph)


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





if __name__ == '__main__':
    input_model = sys.argv[1]
    #convert_pb_to_pbtxt(input_model)
    m = model(input_model)
    m.write_pbtxt()
    m.write_summary()




