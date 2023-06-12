import speech_recognition as sr
from curse_detector import CurseDetector
import os
import wave
import pyaudio
import numpy as np
import mysql.connector
from datetime import datetime
import tkinter as tk
from tkinter import ttk, Label, Button, Text, Scrollbar, messagebox

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

def get_threshold(stream, chunk, RATE,seconds=2):
    audio_energy_values = []

    for _ in range(int(seconds / chunk * RATE)):
        data = stream.read(chunk)
        audio_energy = np.frombuffer(data, dtype=np.int16).max()
        audio_energy_values.append(audio_energy)

    average = np.mean(audio_energy_values)
    std_dev = np.std(audio_energy_values)

    return int(average + std_dev+100)

def record_audio(txt, output_file, stop_when_silence=5):
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 1024
    
    
    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    SILENCE_THRESHOLD = get_threshold(stream, CHUNK,RATE)  # 소리 강도 임계 값 (잠잠한 환경에 적합)
    txt.insert(tk.END, f"Calculated silence threshold: {SILENCE_THRESHOLD}\n")
    txt.insert(tk.END, "당신의 목소리를 녹음 중입니다...\n")
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

    txt.insert(tk.END, "녹음이 끝났습니다.\n")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    # 녹음한 음성 파일 저장
    with wave.open(output_file, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        

def recognize_speech(txt, filename, curse):
    txt.insert(tk.END, "음성을 인식 중입니다...\n")
    txt.see(tk.END)
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(filename) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data, language="ko-KR")
            ensemble = curse.ensemble(text)
            masking = curse.masking(text)
            save_to_db("1", text, masking, ensemble)
            txt.insert(tk.END, f'원래 텍스트: {text}\n')
            txt.insert(tk.END, f'앙상블 결과: {ensemble}\n')
            txt.insert(tk.END, f'마스킹 결과: {masking}\n\n')
            txt.see(tk.END)

    except Exception:
        txt.insert(tk.END, "다시 녹음해주세요\n\n")
        txt.see(tk.END)
    



def recognize_speech_wrapper(txt, filename, curse):
    record_audio(txt, filename)
    recognizer = sr.Recognizer() 
    try:
        with sr.AudioFile(filename) as source:
            audio_data = recognizer.record(source)
            print("음성을 인식 중입니다...")
            text = recognizer.recognize_google(audio_data, language="ko-KR")
            ensemble = curse.ensemble(text)
            masking = curse.masking(text)
            save_to_db("1", text, masking, ensemble)
            txt.insert(tk.END, f'원래 텍스트: {text}\n')
            txt.insert(tk.END, f'앙상블 결과: {ensemble}\n')
            txt.insert(tk.END, f'마스킹 결과: {masking}\n\n')

    except Exception:
        txt.insert(tk.END, "다시 녹음해주세요\n\n")

def main_window(recognizer_output_file, curse_detector):
    window = tk.Tk()
    window.title("음성 인식기")

    txt = Text(window, wrap="word", height=10, width=40)
    txt.grid(column=0, row=0, padx=5, pady=5)

    # 스크롤 바 생성
    scroll_bar = tk.Scrollbar(window, command=txt.yview)
    scroll_bar.grid(row=0, column=1, padx=0, pady=5, sticky='ns')
    txt['yscrollcommand'] = scroll_bar.set

    btn = Button(window, text="음성 인식 시작", command=lambda: recognize_speech(txt, recognizer_output_file, curse_detector))
    btn.grid(column=0, row=1)

    window.mainloop()

if __name__ == "__main__":
    recognizer_output_file = "recorded_audio.wav"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    weights_paths = ['C:/hw/Curse-detection-v2/src/models/weights6.h5']

    curse_detector = CurseDetector(weights_paths)
    i = 0

    while (True):
        if (i == 0):
            print(curse_detector.masking("loding complete"))

        i = 1
        output_file = "recorded_audio.wav"

        main_window(recognizer_output_file, curse_detector)