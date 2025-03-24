import os
import numpy as np
import multiprocessing

import librosa
import soundfile as sf

TIME_INTERVAL = 60  # in seconds
IGNORE_MIN = 5      # in minutes

OUTPUT_FOLDER = "processdata"
INPUT_FOLDER = "data"

def segment_mp3(input_file, input_folder = INPUT_FOLDER, output_folder= OUTPUT_FOLDER):
    try:
        # print(input_folder + "\\" + input_file)
        y, sr = librosa.load(input_folder + "\\" + input_file)  # Load with original sampling rate
    except Exception as e:
        print(f"Error loading audio: {e}")
        return

    interval_samples = sr * TIME_INTERVAL
    num_intervals = int(np.floor(len(y) / interval_samples))

    os.makedirs(output_folder, exist_ok= True)

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
        output_file = output_folder + '\\' + input_file.split('.')[0] + f"_{i+1}.mp3"
        sf.write(output_file, interval_audio, sr)

    print(f"Audio file '{input_file}' split and saved to '{output_folder}' successfully.")

if __name__ == "__main__":
    genres= os.listdir(INPUT_FOLDER)
    processes = []
    
    for genre in genres:
        print("Genre:",genre)
        #args = []

        for _,_,files in os.walk(INPUT_FOLDER + "\\" + genre):
            for i,file in enumerate(files):
                # print(file, i)

                #segment_mp3(file, INPUT_FOLDER + "\\" + genre, OUTPUT_FOLDER + "\\" + genre)
                p = multiprocessing.Process(target= segment_mp3, 
                                      args= (file, INPUT_FOLDER + "\\" + genre, OUTPUT_FOLDER + "\\" + genre))
                processes.append(p)
                p.start()

                #args.append((file, INPUT_FOLDER + "\\" + genre, OUTPUT_FOLDER + "\\" + genre))

        #with multiprocessing.Pool() as pool:
        #    pool.starmap(segment_mp3, args)

    for p in processes:
        p.join()

    print("Segments done!")