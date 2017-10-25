import cv2
from label_image import *

import time

VideoPath = "./drive_thru.avi"

file_name = "tensorflow/examples/label_image/data/grace_hopper.jpg"
model_file = "/home/ahmer/Tecknow/TensorFlow-Retraining/output_tecknow_graph.pb"
label_file = "/home/ahmer/Tecknow/TensorFlow-Retraining/output_tecknow_labels.txt"
input_height = 224
input_width = 224
input_mean = 128
input_std = 128
input_layer = "input"
output_layer = "final_result"


graph = load_graph(model_file)
cap = cv2.VideoCapture(VideoPath)
while not cap.isOpened():
    cap = cv2.VideoCapture(VideoPath)
    cv2.waitKey(1000)
    print "Wait for the header"


while True:
    start_time = time.time()
    flag, frame = cap.read()
    if flag:
        encoded_frame = cv2.imencode('.jpg',frame)[1].tostring()   
        t = read_tensor_from_jpeg_encoded_data(encoded_frame,
                                  input_height=input_height,
                                  input_width=input_width,
                                  input_mean=input_mean,
                                  input_std=input_std)

        input_name = "import/" + input_layer
        output_name = "import/" + output_layer
        input_operation = graph.get_operation_by_name(input_name);
        output_operation = graph.get_operation_by_name(output_name);

        with tf.Session(graph=graph) as sess:
            results = sess.run(output_operation.outputs[0],
                              {input_operation.outputs[0]: t})
            results = np.squeeze(results)

            top_k = results.argsort()[-1:][::-1]
            labels = load_labels(label_file)
            for i in top_k:
                print(labels[i], results[i])
                cv2.putText(frame,'Label: ' + labels[i] + ' ' + str(results[i]) ,(10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,0,0),2)

        print("--- %s seconds ---" % (time.time() - start_time))
        # The frame is ready and already captured
        cv2.imshow('video', frame)
        cv2.waitKey(0)
    else:
        continue


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
