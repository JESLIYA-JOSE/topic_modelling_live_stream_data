import m3u8
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm_notebook as tqdm
import subprocess
import moviepy.editor as mp 
import speech_recognition as sr
import pandas as pd
import os
from topic_modelling_lda import topic_modelling
# Initializing the topic modelling part. Training and tokenization is done here. 1 is passed if the training is required. For using old models, pass 0 to the function
lda,vectorizer_tf,topics=topic_modelling(1)
sess = requests.Session()
#publicly avilable video of ipl data is taken
r = sess.get("https://www.iplt20.com/video/144829/final-csk-vs-srh-fbb-stylish-player-of-the-match-lungi-ngidi")

soup = BeautifulSoup(r.content, 'html5lib')
video_id = soup.find('video', attrs={'id': 'playlistPlayer'})['data-video-id']
account_id = soup.find('video', attrs={'id': 'playlistPlayer'})['data-account']

url = "https://secure.brightcove.com/services/mobile/streaming/index/master.m3u8"

params = {
    'videoId': video_id,
    'pubId': account_id,
    'secure': True
}

r = sess.get(url, params=params)
m3u8_master = m3u8.loads(r.text)
m3u8_playlist_uris = [playlist['uri'] for playlist in m3u8_master.data['playlists']]
m3u8_master.data

playlist_uri = m3u8_playlist_uris[0]

r = sess.get(playlist_uri)
playlist = m3u8.loads(r.text)
m3u8_segment_uris = [segment['uri'] for segment in playlist.data['segments']]
with open("video.ts", 'wb') as f:
    for segment_uri in tqdm(m3u8_segment_uris):
        r = sess.get(segment_uri)
        f.write(r.content)
for count,segment_uri in enumerate(tqdm(m3u8_segment_uris)):
    with open("video.ts", 'wb') as f:
        r = sess.get(segment_uri)
        f.write(r.content)
  
        # Insert Local Video File Path  
        clip = mp.VideoFileClip(r"video.ts") 
  
        # Insert Local Audio File Path 
        aud_path="vid"+str(count)+".wav"
        clip.audio.write_audiofile(aud_path) 
        filename = aud_path
        r = sr.Recognizer()
        # open the file
        with sr.AudioFile(filename) as source:
            # with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source)
            audio_data = r.record(source) 
            # recognize speech using google
            try:
                print("Decoded voice is \n" + r.recognize_google(audio_data))
                text_from_vid= r.recognize_google(audio_data)
                data = [[0,text_from_vid]]
                df = pd.DataFrame (data, columns = ['no','text'])
                # print(df.text)
                from topic_modelling_lda import test_topic_model
                print(test_topic_model(df,lda,vectorizer_tf,topics))

            except Exception as e:
                print("Error :  " + str(e))
#removing created wav files(change the file path)
filepath="/home/..../topic_modelling"
filelist = [ f for f in os.listdir(filepath) if f.endswith(".wav") ]
for f in filelist:
    os.remove(os.path.join(filepath, f))
  
