#importing all the nessecary libraries
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
from PIL import Image
import os
import matplotlib.pyplot as plt
import librosa
import librosa.display as lplt
from sklearn.preprocessing import normalize
import time
import pyaudio
import wave
import pickle
from util import extract_feature
import form as f
import warnings
import streamlit as st
import numpy as np
import pandas as pd
import sqlite3
warnings.filterwarnings('ignore')

st.set_option('deprecation.showPyplotGlobalUse', False)

#creating the layout of the side menu
with st.sidebar:
    choose = option_menu("Speech Emotion Recognition", ["About", "Audio clip upload and visualization", "Emotion Prediction", "Feedback"],
                         icons=['house', 'camera-video', 'arrow-clockwise','person lines fill'],
                         menu_icon="app-indicator", default_index=0,
                         styles={
                                    "container": {"padding": "5!important"},
                                    "icon": {"color": "#B5CF22", "font-size": "25px"}, 
                                    "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
                                    "nav-link-selected": {"background-color": "#B5CFF6"}})



#Starting of the about page
if choose == 'About':
    st.write("""
            ## Speech Emotion Recognition using Machine Learning                        
            """)
    st.markdown("Have you ever called a phone service? How was your experience? It is usually very frustrating when you have a robot asking you a bunch of questions. Now imagine you are upset, you decide to call back the company, and still get a robot on the other end of the line. That is an example in which you could try to recognize speech emotion with machine learning and improve customer services. Adding emotions to machines has been recognized as a critical factor in making machines appear and act in a human-like manner.")
    st.write("## What is Speech emotion recognition?")
    st.markdown("Speech Emotion Recognition (SER) is the task of recognizing the emotion from speech irrespective of the semantic contents. However, emotions are subjective and even for humans it is hard to notate them in natural speech communication regardless of the meaning. The ability to automatically conduct it is a very difficult task and still an ongoing subject of research.")
    img = Image.open("D:\sem 6\Major Project\GUI code\Overview-structure-of-the-Speech-emotion-recognition.png")
    st.image(img)
    st.markdown("Speech emotion recognition belongs to a larger group in computing referred to as Affective Computing. It revolves around systems in computing that interpret human affects and then analyze or simulate them. Affective computing ")
    st.markdown("Human affect is a field in psychology that refers to human emotions, feelings and mood. It has three major principals i.e. Valence which refers to then good-bad spectrum of a perceived event or situation, Arousal which is the activation of the Sympathetic nervous system that can be measured objectively and finally Motivational intensity which is the process which compels or motivates an individual to take some action on any particular event.")
    st.markdown("In this project we will be using machine learning and Natural language processing to analyze human speech and classify it. There are two main type of classifications for human speech, continuous and what we will be using for this project, Categorical. We will create neural networks that analyze audio of human speech and then categorize it into different emotions i.e. angry, happy, sad etc. This will allow us to understand human speech and is a step towards giving computers emotional intelligence, which make computers capable of understanding human emotions and personalize their behavior accordingly. There are various machine learning algorithms employed in creating this project such as discriminant classifiers (LDC), k-nearest neighbor (k-NN), Gaussian mixture model (GMM), support vector machines (SVM), artificial neural networks (ANN), decision tree algorithms and hidden Markov models (HMMs) depending on our need and specific classifier is used for different needs to achieve the highest accuracy.")
    


#Second page Audio clip upload and visualization
elif choose == 'Audio clip upload and visualization':
    st.write("""
        ### Upload the audio file
        """)
    
    #creating a variable to store the uploaded audio file
    uploaded_audio = st.file_uploader("Choose audio", type=["mp3", "wav"])

    if uploaded_audio is not None:
        aud = uploaded_audio.name

        with open(os.path.join("tempDir",aud),"wb") as f:                       #saving the file in directory
            f.write(uploaded_audio.getbuffer())
        st.success("Audio saved successfully")

        audio_file = open(os.path.join("tempDir",aud), mode = 'rb').read()      #accessing the saved audio file from directory
        st.audio(audio_file)                                                    #showing the playable interface for the audio

        audio_path = os.path.abspath(os.path.join("tempDir",aud))               #saving the path from the directory of the file

        #feature extraction process
        st.write("### Feature Extraction")

        if st.checkbox("Show Waveplot", value =True):
            plt.figure(figsize=(8,5))
            plt.title("Waveplot")
            ax = plt.axes()
            ax.set_facecolor("#B6C7D0")
            data, sampling_rate = librosa.load(audio_path)
            librosa.display.waveshow(data, sr=sampling_rate)
            wavePlot = plt.show()
            st.pyplot(wavePlot)

        if st.checkbox("Show mel-frequency spectrogram", value =True):
            spec = librosa.feature.melspectrogram(y=data, sr=sampling_rate, n_mels=128,

                                    fmax=8000)
            fig, ax = plt.subplots()

            spec_db = librosa.power_to_db(spec, ref=np.max)

            img = librosa.display.specshow(spec_db, x_axis='time',

                         y_axis='mel', sr=sampling_rate,

                         fmax=8000, ax=ax)

            fig.colorbar(img, ax=ax, format='%+2.0f dB')

            ax.set(title='Mel-frequency spectrogram')
            st.pyplot(plt.show())

        if st.checkbox("Show Spectrogram", value = True):
            x = librosa.stft(data)
            xdb = librosa.amplitude_to_db(abs(x))
            plt.figure(figsize=(11,4))
            plt.title("Spectrogram")
            librosa.display.specshow(xdb, sr=sampling_rate, x_axis='time', y_axis='hz')
            plt.colorbar()
            specPlot = plt.show()
            st.pyplot(specPlot)

        if st.checkbox("Show zero-crossing rate", value = True):
            start = 1000
            end = 1200
            plt.figure(figsize = (14, 5))
            plt.plot(data, color = "#9400D3")
            plt.grid()
            plt.title("Zero-Crossing Rate")
            st.pyplot(plt.show())
        
        if st.checkbox("Show Chroma Frequencies", value = True):
            chroma = librosa.feature.chroma_stft(data, sr = sampling_rate)
            plt.figure(figsize = (16, 6))
            lplt.specshow(chroma , sr = sampling_rate, x_axis = 'time', y_axis = 'chroma', cmap = 'gist_rainbow')
            plt.colorbar()
            plt.title("Chroma Features")
            st.pyplot(plt.show())

        if st.checkbox("Show Spectral Rolloff", value = True):
            spectral_rolloff = librosa.feature.spectral_rolloff(data+0.01, sr = sampling_rate)[0]
            plt.figure(figsize = (16, 6))
            librosa.display.waveshow(data, sr = sampling_rate, alpha = 0.4, color = "#9400D3")
            plt.title('Spectral Rolloff')
            st.pyplot(plt.show())


#Third page for emotion prediction
elif choose == 'Emotion Prediction':
    st.write("### Emotion Prediction")

    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 48000
    CHUNK = 512
    RECORD_SECONDS = 5
    WAVE_OUTPUT_FILENAME = "recordedFile.wav"
    device_index = 2
    audio = pyaudio.PyAudio()

    #listing the input device available in user device
    st.markdown("Below is the list of input device available in your device:")

    st.write("----------------------record device list---------------------")
    info = audio.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    for i in range(0, numdevices):
        if (audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            st.write("Input Device id ", i, " - ", audio.get_device_info_by_host_api_device_index(0, i).get('name'))

    st.write("---------------------------end of list---------------------------")

    #getting the input from user for the preferred input device
    index = st.number_input("Enter the input device index: ",0,10,0,1)
    st.write("Recording via index "+str(index))

    #starting the recording process
    if st.button("Start Recording"):
        stream = audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,input_device_index = index,
                frames_per_buffer=CHUNK)
        st.write("Recording started")

        #recording the audio frame by frame
        Recordframes = []
        with st.spinner('Recording in progress...'):
            for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK, exception_on_overflow = False)
                Recordframes.append(data)
        st.success("Audio Recorded successfully!")

        stream.stop_stream()
        stream.close()
        audio.terminate()
    
        #saving the recorded audio in the desired file and showing to the user
        waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(Recordframes))
        waveFile.close()
        st.audio("recordedFile.wav")


        #predicting the emotion of the recorded audio using the machine learning model we created
        if __name__ == "__main__":
            model = pickle.load(open("result/mlp_classifier.model", "rb"))
            filename = "recordedFile.wav"
            features = extract_feature(filename, mfcc=True, chroma=True, mel=True).reshape(1, -1)
            result = model.predict(features)[0]
            st.write("### Emotion detected:", result) 



#last page for collecting the user feedback
elif choose == 'Feedback':
    conn = sqlite3.connect('student_feedback.db')           #connecting the database to the system
    c = conn.cursor()
    

    #fuction to creating table in the database
    def create_table():
        c.execute('CREATE TABLE IF NOT EXISTS feedback(date_submitted DATE, Q1 TEXT, Q2 TEXT, Q3 INTEGER, Q4 TEXT, Q5 TEXT, Q6 TEXT, Q7 TEXT)')

    #function to insert the user feedback into the table
    def add_feedback(date_submitted, Q1, Q2, Q3, Q4, Q5, Q6, Q7):
        c.execute('INSERT INTO feedback (date_submitted,Q1, Q2, Q3, Q4, Q5, Q6, Q7) VALUES (?,?,?,?,?,?,?,?)',(date_submitted,Q1, Q2, Q3, Q4, Q5, Q6, Q7))
        conn.commit()

    def main():
        
        #defining the questions and there type
        st.title("Feedback")

        d = st.date_input("Today's date",None, None, None, None)
    
        question_1 = st.selectbox('What emotion you tried for?',('','Angry', 'Happy', 'Sad','Fearful','Neutral','Peaceful','Surprised','Disgust'))
        st.write('You selected:', question_1)
    
        question_2 = st.selectbox('What emotion you got in result?',('','Angry', 'Happy', 'Sad','Fearful','Neutral','Peaceful','Surprised','Disgust'))
        st.write('You selected:', question_2)
    
        question_3 = st.slider('Overall, how satisfied are you with the results? (5 being very happy and 1 being very dissapointed)', 1,5,1)
        st.write('You selected:', question_3)

        question_4 = st.selectbox('Was the application fun and interactive?',('','Yes', 'No'))
        st.write('You selected:', question_4)

        question_5 = st.selectbox('Was the content interesting and engaging?',('','Yes', 'No'))
        st.write('You selected:', question_5)

        question_6 = st.selectbox('Is there something to be changed?',('','Yes', 'No'))
        st.write('You selected:', question_6)

        question_7 = st.text_input('What we can do to make it more interactive?', max_chars=50)

        if st.button("Submit feedback"):
            create_table()
            add_feedback(d, question_1, question_2, question_3, question_4, question_5, question_6, question_7)
            st.success("Feedback submitted")

    if __name__ == '__main__':
        main()