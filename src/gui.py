import speech_recognition as sr
from curse_detector import CurseDetector
import wave
import pyaudio
import numpy as np
import mysql.connector
from datetime import datetime
import tkinter as tk

def save_to_db(uid, origin_text, filter_text, score):
    try:
        connection = mysql.connector.connect(user='js',
                                             password='js',
                                             host='localhost',
                                             database='js')
        cursor = connection.cursor()

        query = '''INSERT INTO PVMM (uid, origin_text, filter_text, score, time_stamp)
                   VALUES (%s, %s, %s, %s, %s)'''
        timestamp = datetime.now()
        cursor.execute(query, (uid, origin_text, filter_text, float(score), timestamp))
        connection.commit()

    except mysql.connector.Error as error:
        print(f"Error connecting to MySQL server: {error}")
        return False

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

    return True

def callback(recognizer, audio):
    try:
        text_var.set("start")
        text = recognizer.recognize_google(audio, language='ko-KR')
        current_text = text_var.get()
        if current_text:
            text_var.set(current_text + "\n" + str(curse.ensemble(text))+"\n"+curse.masking(text))
        else:
            text_var.set(" ")
    except Exception as e:
        result_label.config(text="음성을 인식할 수 없습니다.")
        
def get_threshold(stream, chunk, RATE,seconds=2):
    audio_energy_values = []

    for _ in range(int(seconds / chunk * RATE)):
        data = stream.read(chunk)
        audio_energy = np.frombuffer(data, dtype=np.int16).max()
        audio_energy_values.append(audio_energy)

    average = np.mean(audio_energy_values)
    std_dev = np.std(audio_energy_values)

    return int(average + std_dev+100)

def recognize_speech(curse, output_file, text_var, stop_when_silence=1):
    text_var.set('start')
    app.update()
    recognizer = sr.Recognizer()
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 1024
    
    
    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    SILENCE_THRESHOLD = get_threshold(stream, CHUNK,RATE)  # 소리 강도 임계 값 (잠잠한 환경에 적합)
    current_text = text_var.get()
    text_var.set(current_text + "\n" +str(SILENCE_THRESHOLD) )
    app.update()
    #print(f"Calculated silence threshold: {SILENCE_THRESHOLD}")
    current_text = text_var.get()
    text_var.set(current_text + "\n" +"당신의 목소리를 녹음 중입니다..." )
    app.update()
    frames = []
    silence_count = 0

    while True:
        data = stream.read(CHUNK)
        frames.append(data)
        audio_energy = np.frombuffer(data, dtype=np.int16).max()  # audio_energy 변환 수정

        # 소리가 임계값 이하일 경우 카운트 증가, 아니면 카운트 초기화
        if audio_energy < SILENCE_THRESHOLD:
            silence_count += 1
        else:
            silence_count = 0

        # 지정된 시간 동안 소리가 없을 때 녹음 종료
        if silence_count >= RATE/CHUNK * stop_when_silence:
            break

        app.update()  # Tkinter 윈도우 갱신 추가

    #print("녹음이 끝났습니다.")
    current_text = text_var.get()
    text_var.set(current_text + "\n" +"녹음이 끝났습니다." )
    app.update()

    stream.stop_stream()
    stream.close()
    audio.terminate()

    # 녹음한 음성 파일 저장
    with wave.open(output_file, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        
    with sr.AudioFile(output_file) as source:
        audio_data = recognizer.record(source)
    try:
        #print("음성을 인식 중입니다...")
        current_text = text_var.get()
        text_var.set(current_text + "\n" +"음성을 인식 중입니다..." )
        app.update()
        text = recognizer.recognize_google(audio_data, language='ko-KR')
        #print("Here's the text from the audio:")
        current_text = text_var.get()
        text_var.set(current_text + "\n" +"Here's the text from the audio:"+"\n" )
        app.update()
        ensemble = curse.ensemble(text)
        masking = curse.masking(text)
        current_text = text_var.get()
        text_var.set(current_text + "\n" +str(ensemble)+"\n"+str(masking)+"\n")
        app.update()
        
        #print(ensemble)
        #print(masking)
        save_to_db("1", text,masking,ensemble)
        app.after(1500, lambda: recognize_speech(curse, output_file, text_var, stop_when_silence))
    except Exception:
        current_text = text_var.get()
        text_var.set(current_text + "\n" +"다시 녹음해주세요")
        app.update()
        #print("다시 녹음해주세요")
        app.after(1500, lambda: recognize_speech(curse, output_file, text_var, stop_when_silence))



weights_paths = ['C:/hw/Curse-detection-v2/src/models/weights6.h5']

curse = CurseDetector(weights_paths)
print(curse.masking("loding complete"))

app = tk.Tk()
app.title("Continuous Speech Recognition and Tkinter")
output_file = "recorded_audio.wav"
start_btn = tk.Button(app, text="음성 인식 시작", command=lambda: recognize_speech(curse, output_file, text_var))
start_btn.pack(padx=10, pady=10)

text_var = tk.StringVar()
result_label = tk.Label(app, textvariable=text_var)
result_label.pack(padx=10, pady=10)

app.mainloop()