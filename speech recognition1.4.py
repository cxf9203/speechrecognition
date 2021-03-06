import time
import os
import threading
import sys
import nls  # ALI SDK
import pyttsx3  # python -m pip install pyttsx3==2.71  注意版本问题，过高会报错。。
import pyaudio   #3.7版本whell安装
import wave
from tqdm import tqdm  # pip install tqdm
from playsound import playsound
from pygame import mixer    #pip install pygame
#setup
CHUNK = 1024
cmd = ""
URL = "wss://nls-gateway.cn-shang
AKKEY = "XPimfUwDi08jI2fWFG1
APPKEY = "m9qkQFFc0yn

TEXT=''
def SpeakText(command):
    engine = pyttsx3.init()
    engine.say(command)
    engine.runAndWait()

class Test_to_sounds(threading.Thread):
    def __init__(self, threadID, file_name, text):
        threading.Thread.__init__(self)
        self.id = threadID
        self.test_file = file_name
        self.text = text
    def test_on_metainfo(self, message, *args):
        print("on_metainfo message=>{}".format(message))

    def test_on_error(self, message, *args):
        print("on_error args=>{}".format(args))

    def test_on_close(self, *args):
        print("on_close: args=>{}".format(args))
        try:
            self.f.close()
        except Exception as e:
            print("close file failed since:", e)

    def test_on_data(self, data, *args):
        try:
            self.f.write(data)
        except Exception as e:
            print("write data failed:", e)
    def test_on_completed(self, message, *args):
        print("on_completed:args=>{} message=>{}".format(args, message))
    def run(self):
        threadLock.acquire()
        print("thread tts:{} start..".format(self.id))
        self.f = open(self.test_file, "wb+")
        #api 接口结构
        tts = nls.NlsSpeechSynthesizer(
            url=URL,
            akid=AKID,
            aksecret=AKKEY,
            appkey=APPKEY,
            on_metainfo=self.test_on_metainfo,
            on_data=self.test_on_data,
            on_completed=self.test_on_completed,
            on_error=self.test_on_error,
            on_close=self.test_on_close,
            callback_args=[self.id]
        )
        print("{}: API session start".format(self.id))
        r = tts.start(self.text, voice="xiaobei", aformat="mp3")
        print("{}: API tts done with result:{}".format(self.id, r))
        time.sleep(5)

        threadLock.release()

class Sounds_to_text(threading.Thread):  # 语音识别类
    def __init__(self, threadID, filename):
        threading.Thread.__init__(self)
        self.__id = threadID
        self.__test_file = filename
    def test_on_sentence_begin(self, message, *args):
        print("test_on_sentence_begin:{}".format(message))
    def test_on_sentence_end(self, message, *args):
        global cmd
        print("test_on_sentence_end:{}".format(message))
        dict_1 = eval(message)  ###str convert to dict
        print(dict_1)
        print("test_on_sentence_end:{}".format(dict_1["payload"]))
        if ('result' in dict_1["payload"].keys()):
            print("result received")
            cmd = dict_1["payload"]["result"]
            print("current text is ", cmd)
        else:
            pass
    def test_on_start(self, message, *args):
        print("test_on_start:{}".format(message))
    def test_on_error(self, message, *args):
        print("on_error args=>{}".format(args))
    def test_on_close(self, *args):
        print("on_close: args=>{}".format(args))
    def test_on_result_chg(self, message, *args):
        print("test_on_chg:{}".format(message))
    def test_on_completed(self, message, *args):
        print("on_completed:args=>{} message=>{}".format(args, message))
    def run(self):
        # Get lock to synchronize threads
        threadLock.acquire()
        with open(self.__test_file, "rb") as f:
            self.__data = f.read()
        print("sounds_to texts thread:{} start..".format(self.__id))
        sr = nls.NlsSpeechTranscriber(
            url=URL,
            akid=AKID,
            aksecret=AKKEY,
            appkey=APPKEY,
            on_sentence_begin=self.test_on_sentence_begin,
            on_sentence_end=self.test_on_sentence_end,
            on_start=self.test_on_start,
            on_result_changed=self.test_on_result_chg,
            on_completed=self.test_on_completed,
            on_error=self.test_on_error,
            on_close=self.test_on_close,
            callback_args=[self.__id]
        )
        print("{}: API sounds to text session start".format(self.__id))
        r = sr.start(aformat="pcm",
                     enable_intermediate_result=False,
                     enable_punctutation_prediction=True,
                     enable_inverse_text_normalization=True)
        self.__slices = zip(*(iter(self.__data),) * 640)
        for i in self.__slices:
            sr.send_audio(bytes(i))
            time.sleep(0.01)
        sr.ctrl(ex={"test": "tttt"})
        time.sleep(1)
        r = sr.stop()
        print("{}: sr stopped:{}".format(self.__id, r))
        time.sleep(5)
        # Free lock to release next thread
        threadLock.release()
class audioplay(threading.Thread):
    def __init__(self, audio):
        threading.Thread.__init__(self)
        self.audio = audio

    def run(self):
        threadLock.acquire()
        playsound(self.audio)
        threadLock.release()

# 获取麦克风音频并保存
def record_audio(wave_out_path, record_second):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    #FORMAT = pyaudio.paInt8 #8
    CHANNELS = 1  #单声道 双声道请设置2 该语音识别请务必使用单声道 采样率为16000/8000
    RATE = 16000
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    wf = wave.open(wave_out_path, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    print("* recording")
    for i in tqdm(range(0, int(RATE / CHUNK * record_second))):
        data = stream.read(CHUNK)
        wf.writeframes(data)
    print("* done recording")
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf.close()


# 新建一个线程用来获取麦克风输入后得到的临时文档
class audiothread(threading.Thread):
    def __init__(self, threadID, name):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name

    def run(self):
        try:
            print("listen ....")
            record_audio("cmdaudio.wav", record_second=4)
        except:
            pass


# 新建一个实例线程用来获取麦克风输入后得到的临时文档
def torecordeaudio():
    thread3recordvideo = audiothread(3, "thread3audioget")
    thread3recordvideo.start()
    thread3recordvideo.join()

nls.enableTrace(False)
threadLock = threading.Lock()
threads = []
# Add threads to thread list


if __name__ == '__main__':
    #初始化
    #SpeakText('hello brother I am Alex，how is it going')
   # SpeakText("我是离线时的语音助手，我叫 爱勒克斯")
    thread_tts = Test_to_sounds("thread1", "output.mp3", "我是陈信奉的语音助手小贝，祝老婆元旦快乐！新的一年没有加班")
    thread_tts.start()
    thread_tts.join()
    thread_audioplay = audioplay("output.mp3")  # 将源码中command = ' '.join(command).encode('utf-16')变为command = ' '.join(command)即可
    thread_audioplay.start()
    thread_audioplay.join()
    os.remove("output.mp3")

    time.sleep(1)
    while 1:
        threads = []
        # triggrt to speak
        SpeakText("please speaking")
        # record audio for command
        torecordeaudio()
        # transfer command to text and get cmd text
        thread_stt = Sounds_to_text("thread4", "cmdaudio.wav")
        threads.append(thread_stt)
        thread_stt.start()  #get text cmd
        thread_stt.join()
        print(cmd)
        thread_tts = Test_to_sounds("thread1", "output.mp3", cmd)
        # Add threads to thread list
        threads.append(thread_tts)
        # threads.append(thread_audioplay)
        thread_tts.start()
        thread_tts.join()
        print("Exiting tts Thread_tts")
        for t in threads:
            t.join()
        print("Exiting  Thread")
        # Delete file
        thread_audioplay = audioplay("output.mp3")  #将源码中command = ' '.join(command).encode('utf-16')变为command = ' '.join(command)即可
        thread_audioplay.start()
        thread_audioplay.join()
        os.remove("output.mp3")




