import cv2
import numpy as np
import librosa
def solution(audio_path):
    ############################
    ############################

    ############################
    ############################
    ## comment the line below before submitting else your code wont be executed##
    # pass
    class_name = 'cardboard'

    spec_db=extract_mel_spectrogram(audio_path=audio_path)
    spec_db=normalize_spectrum_to_color(spec_db=spec_db)
    a=2*calculate_white_threshold(spec_db)+calculate_intensity_wise(spec_db)+calculate_intensity_wise1(spec_db)
    # print(a)
        
    if a>=0:
        class_name='metal'
    return class_name
def normalize_spectrum_to_color(spec_db):
    if spec_db.min() != spec_db.max():
        spec_db=255*((spec_db-spec_db.min())/(spec_db.max()-spec_db.min()))
    else:
        spec_db=np.zeroes_like(spec_db)
    return spec_db
def extract_mel_spectrogram(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    n_fft = 2048
    hop_length = 512

    spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,fmin=20, fmax=22000)
    # print(spec)
    spec_db = librosa.power_to_db(spec, ref=np.max)
    # print(spec_db.shape)
    return spec_db
    # spec_db=255*((spec_db-spec_db.min())/(spec_db.max()-spec_db.min()))
    # plt.imshow(spec_db,cmap="gray")
    # plt.show()
    # spec_db=-spec_db
    # print(spec_db.shape)
    # print(np.sum(spec_db>200))
    # print(spec_db)
    # return spec_db
def calculate_white_threshold(spec_db):
    a=np.sum(spec_db>200)
    # print(a)
    if a>1500:
        return 1  #metal
    else:
        return -1      #cardboard
def calculate_intensity_wise(spec_db):
    calculate_intensity_wise1(spec_db=spec_db)
    avg_intensity_x = np.median(spec_db, axis=0)
    b=np.max(avg_intensity_x)
    # print(b)
    if b>165:
        return 1
    else:
        return -1
def calculate_intensity_wise1(spec_db):
    avg_intensity_x = np.mean(spec_db, axis=0)
    b=np.max(avg_intensity_x)
    # print(b)
    if b>155:
        return 1
    else:
        return -1
# for i in range(4,9):
#     print(solution("../test_train/test/Cardboard"+str(i)+".mp3"))