from flask import Flask, render_template, request, redirect, url_for
from pytube import YouTube
import os
import os
import cv2
import pafy
import math
import random
import numpy as np
import datetime as dt
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip


from moviepy.editor import *

from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
from keras.utils import to_categorical
IMAGE_HEIGHT , IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 20
import tensorflow as tf
from tensorflow.keras.models import load_model
CLASSES_LIST = ["BaseballPitch", "BenchPress", "BreastStroke", "CleanAndJerk", "HorseRace", "JavelinThrow", "Lunges", "PoleVault",  "PullUps", "Rowing", "SkateBoarding", "Basketball", "SoccerJuggling", "TaiChi", "TennisSwing", "TrampolineJumping"]
app = Flask(__name__)
LRCN_model=load_model("LRCN_model___Date_Time_2024_04_18__07_04_04___Loss_0.31659677624702454___Accuracy_0.9118198752403259.h5")
import tensorflow as tf
import numpy as np

def print_convolution_matrix(model_path, layer_name):
    model = tf.keras.models.load_model(model_path)

    conv_layer = None
    for layer in model.layers:
        if layer.name == layer_name:
            conv_layer = layer
            break

    if conv_layer is None:
        print(f"Convolutional layer '{layer_name}' not found in the model.")
        return

    weights, biases = conv_layer.get_weights()

    print(f"Convolution Matrix for Layer '{layer_name}':")
    num_filters = weights.shape[3]
    filter_size = weights.shape[0]

    for i in range(num_filters):
        print(f"Filter {i + 1}:")
        filter_matrix = weights[:, :, :, i]
        for row in filter_matrix:
            print(' '.join(f'{x:.2f}' for x in row))
        print()

model_path = 'LRCN_model___Date_Time_2024_04_18__07_04_04___Loss_0.31659677624702454___Accuracy_0.9118198752403259.h5'
conv_layer_name = 'conv2d_1'
print_convolution_matrix(model_path, conv_layer_name)


@app.route('/')
def index():
    return render_template('indexHAR.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        youtube_link = request.form['youtubeLink']
        return redirect(url_for('result', youtube_link=youtube_link))

@app.route('/result')
def result():
    predicted_class_name=' '
    youtube_link = request.args.get('youtube_link', '')
    def download_youtube_videos(youtube_video_url, output_directory):
       
        yt = YouTube(youtube_video_url)

        title = yt.title

        stream = yt.streams.get_highest_resolution()

        output_file_path = f'{output_directory}/{title}.mp4'

        stream.download(output_directory)

        return title
    test_videos_directory = 'static'
    os.makedirs(test_videos_directory, exist_ok = True)

    video_title = download_youtube_videos(youtube_link, test_videos_directory)

    input_video_file_path = f'{test_videos_directory}/{video_title}.mp4'
    def predict_on_video(video_file_path, output_file_path, SEQUENCE_LENGTH):
        

        video_reader = cv2.VideoCapture(video_file_path)

        original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

        video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), 
                                    video_reader.get(cv2.CAP_PROP_FPS), (original_video_width, original_video_height))

        frames_queue = deque(maxlen = SEQUENCE_LENGTH)

        predicted_class_name = ''

        while video_reader.isOpened():

                ok, frame = video_reader.read() 
            
                if not ok:
                 break

                resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
            
                normalized_frame = resized_frame / 255

                frames_queue.append(normalized_frame)

                if len(frames_queue) == SEQUENCE_LENGTH:

                        predicted_labels_probabilities = LRCN_model.predict(np.expand_dims(frames_queue, axis = 0))[0]

                        predicted_label = np.argmax(predicted_labels_probabilities)

                        predicted_class_name = CLASSES_LIST[predicted_label]

                cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                video_writer.write(frame)
            
        video_reader.release()
        video_writer.release()
    def predict_single_action(video_file_path, SEQUENCE_LENGTH):


        video_reader = cv2.VideoCapture(video_file_path)

        original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frames_list = []
        
        predicted_class_name = ''

        video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

        skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH),1)

        for frame_counter in range(SEQUENCE_LENGTH):

                video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)

                success, frame = video_reader.read() 

                if not success:
                    break

                resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
            
                normalized_frame = resized_frame / 255
            
                frames_list.append(normalized_frame)

                LRCN_model = load_model('LRCN_model___Date_Time_2024_04_18__07_04_04___Loss_0.31659677624702454___Accuracy_0.9118198752403259.h5')
        predicted_labels_probabilities = LRCN_model.predict(np.expand_dims(frames_list, axis = 0))[0]

        predicted_label = np.argmax(predicted_labels_probabilities)

        predicted_class_name = CLASSES_LIST[predicted_label]
        
        print(f'Action Predicted: {predicted_class_name}\nConfidence: {predicted_labels_probabilities[predicted_label]}')
            
        video_reader.release()
        return predicted_class_name,predicted_labels_probabilities[predicted_label]
        
    video_title = download_youtube_videos(youtube_link, test_videos_directory)

    input_video_file_path = f'{test_videos_directory}/{video_title}.mp4'

    predicted_class_name,confidence=predict_single_action(input_video_file_path, SEQUENCE_LENGTH)

    print(f"predicted VALUE:{predicted_class_name}")   

    output_video_file_path = f'{test_videos_directory}/{video_title}-Output-SeqLen{SEQUENCE_LENGTH}.mp4'


    return render_template('result.html', youtube_link=youtube_link,confidence=confidence, input_video_path=input_video_file_path,predicted_class_name=predicted_class_name)    

if __name__ == '__main__':
    app.run()
