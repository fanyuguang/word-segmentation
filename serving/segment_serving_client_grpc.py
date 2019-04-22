#!/usr/bin/python3
# -*- coding: utf-8 -*-

import grpc
import time
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc


tf.app.flags.DEFINE_string('sentence', '车的颜色非常纯正', 'Sentence')
tf.app.flags.DEFINE_string('server', 'localhost:8500', 'PredictionService host:port')
FLAGS = tf.app.flags.FLAGS

class ServingClient(object):

    def __init__(self):
        channel = grpc.insecure_channel(FLAGS.server)
        self.predict_stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
        self.model_name = 'segment'
        self.signature_name = 'predict_segment'
        self.timeout = 3 


    def predict(self, sentences):
        """ 
        predict sentences by call tensorflow serving.
        :param sentences:
        :return:
        """
        # Send request
        # See prediction_service.proto for gRPC request/response details.
        request = predict_pb2.PredictRequest()
        request.model_spec.name = self.model_name
        request.model_spec.signature_name = self.signature_name
        request.inputs['input_sentences'].CopyFrom(tf.contrib.util.make_tensor_proto(sentences, shape=[len(sentences)]))
        result = self.predict_stub.Predict(request, 3.0)  # 10 secs timeout

        predict_labels = result.outputs['predict_labels'].string_val
        predict_labels = [label.decode('utf-8').split() for label in predict_labels]
        predict_scores = result.outputs['predict_scores'].string_val
        predict_scores = [[float(item) for item in score.decode('utf-8').split()] for score in predict_scores]
        return predict_labels, predict_scores


def main(_):
    serving_client = ServingClient()

    sentences = [' '.join([char for char in FLAGS.sentence])]
    for i in range(10)[3:]:
        for j in range(5):
            try:
                start = time.time()
                predict_labels, predict_scores = serving_client.predict(sentences[: i % 2 + 1])
                end = time.time()
                print(predict_labels)
                print(predict_scores)
                print('Time cost: %d ms' % ((end - start) * 1000))
            except:
                print('Predict failed')
        print('---------------------------------------------------------')
        print('sleep %d s' % ((i + 1) * 60))
        time.sleep((i + 1) * 60)


if __name__ == '__main__':
    tf.app.run()
