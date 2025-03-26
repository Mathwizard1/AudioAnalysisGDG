import os
import multiprocessing

import numpy as np
import csv

import librosa

import gc

TIME_INTERVAL = 60  # in seconds
IGNORE_MIN = 1      # in minutes

N_MFCC = 20

OUTPUT_FOLDER = "processdata"
INPUT_FOLDER = "data"

FILE_PROCESS_LIMIT = 5
PROCESS_LIMIT = 50 
MEMORY_LIMIT = 150 * 1e+6   # In x MB

MIN_AUDIO_LENGTH = 256  # If end parts are too small, skip them

def feature_extract(y, sr, n_mfcc = 20):
    # Tempo information
    tempo = librosa.feature.tempo(y=y, sr=sr).round()

    # Separate harmonic and percussive components, Tonnetz
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)

    # Mathematical features
    features_list = {}

    features_list['tempo'] = [tempo.min(), tempo.mean(), tempo.max(), tempo.var()]
    features_list['y_harmoic'] = [y_harmonic.min(), y_harmonic.mean(), y_harmonic.max(), y_harmonic.var()]
    features_list['y_percussive'] = [y_percussive.min(), y_percussive.mean(), y_percussive.max(), y_percussive.var()]
    features_list['tonnetz'] = [tonnetz.min(), tonnetz.mean(), tonnetz.max(), tonnetz.var()]

    # Other Sound features
    cstft=librosa.feature.chroma_stft(y=y, sr=sr)
    features_list['cstft'] = [cstft.min(), cstft.mean(), cstft.max(), cstft.var()]

    srms=librosa.feature.rms(y=y)
    features_list['srms'] = [srms.min(), srms.mean(), srms.max(), srms.var()]

    specband=librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features_list['specband'] = [specband.min(), specband.mean(), specband.max(), specband.var()]

    speccent=librosa.feature.spectral_centroid(y=y, sr=sr)
    features_list['speccent'] = [speccent.min(), speccent.mean(), speccent.max(), speccent.var()]

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features_list['rolloff'] = [rolloff.min(), rolloff.mean(), rolloff.max(), rolloff.var()]

    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)
    features_list['zero_crossing_rate'] = [zero_crossing_rate.min(), zero_crossing_rate.mean(), zero_crossing_rate.max(), zero_crossing_rate.var()]

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc= n_mfcc)
    for i in range(n_mfcc):
        features_list[f'mfcc_{i}'] = [mfcc[i].min(), mfcc[i].mean(), mfcc[i].max(), mfcc[i].var()]

    # key extends
    flattened_list = []
    for key in sorted(features_list.keys()):
        flattened_list.extend(features_list[key])
    return flattened_list

def writer_process(queue, file_path):
    try:
        with open(file_path, 'a', newline='\n') as csvfile:
            writer = csv.writer(csvfile)
            while True:
                item = queue.get()
                if item is None:  # Sentinel value to stop
                    break
                writer.writerow(item)
    except Exception as e:
        print(f"Writer process error: {e}")

def data_processor(y, sr, genre, queue):
    try:
        X = feature_extract(y, sr, N_MFCC)
        X.append(genre)
        queue.put(X)

        del X, y
        gc.collect()
    except Exception as e:
        print(f"data Processor error: {e}")

def segment_mp3(input_file, input_folder = INPUT_FOLDER, output_folder= OUTPUT_FOLDER):
    output_file = input_file.split('.')[0] + '.csv'
    genre = input_folder.split('\\')[-1]

    queue = multiprocessing.Queue()
    writer = multiprocessing.Process(target=writer_process, args=(queue, output_folder + "\\" + output_file))
    writer.start()

    try:
        # print(input_folder + "\\" + input_file)
        y, sr = librosa.load(input_folder + "\\" + input_file)  # Load with original sampling rate
    except Exception as e:
        print(f"Error loading audio: {e}")
        return

    interval_samples = sr * TIME_INTERVAL
    num_intervals = int(np.floor(len(y) / interval_samples))

    processes = []
    process_count = 0
    for i in range(IGNORE_MIN, num_intervals + 1):
        start_sample = i * interval_samples
        end_sample = (i + 1) * interval_samples
        interval_audio = None

        # Save the remaining part
        if(i < num_intervals):
            interval_audio = y[start_sample:end_sample]
        else:
            interval_audio = y[start_sample:]

        # Save the output
        # output_file = output_folder + "\\" + input_file.split('.')[0] + f"_{i+1}.mp3"
        # sf.write(output_file, interval_audio, sr)

        if(process_count == PROCESS_LIMIT):
            for p in processes:
                p.join()
            processes.clear()
            process_count = 0

        if(len(interval_audio) >= MIN_AUDIO_LENGTH):
            p = multiprocessing.Process(target=data_processor, args=(interval_audio, sr, genre, queue), 
                                        daemon= True)
            process_count += 1

            processes.append(p)        
            p.start()

        del interval_audio
        gc.collect()

    for p in processes:
        p.join()

    queue.put(None)  # Sentinel to signal writer to stop
    writer.join()

    del y
    gc.collect()
    print(f"Audio file '{input_file}' split and saved to '{output_folder}' successfully.")

if __name__ == "__main__":
    genres= os.listdir(INPUT_FOLDER)
    
    for genre in genres:
        for _,_,files in os.walk(INPUT_FOLDER + "\\" + genre):
            os.makedirs(OUTPUT_FOLDER + "\\" + genre, exist_ok= True)

            processes = []
            file_size_limits = 0
            file_process_limits = 0

            for file in files:
                # print(file)

                # add the current memory Usage
                file_size_limits += os.path.getsize(INPUT_FOLDER + "\\" + genre + "\\" + file)

                # Limit memory
                if(file_size_limits >= MEMORY_LIMIT or file_process_limits > FILE_PROCESS_LIMIT):
                    print(file_size_limits / 1e+6)
                    print(file_process_limits)

                    for p in processes:
                        p.join()
                        gc.collect()

                    processes.clear()
                    file_process_limits = 0
                    file_size_limits = os.path.getsize(INPUT_FOLDER + "\\" + genre + "\\" + file)

                p = multiprocessing.Process(
                    target= segment_mp3,
                    args= (file, INPUT_FOLDER + "\\" + genre, OUTPUT_FOLDER + "\\" + genre)
                )
                file_process_limits += 1

                processes.append(p)
                p.start()

            for p in processes:
                p.join()
                gc.collect()

        print("Genre:", genre)
    print("Segments done!")