import os
import subprocess
# loc = EMO-DB/wav


def extract(directory_location):
    for filename in os.listdir(directory_location):
        full_file_path = os.path.join(directory_location, filename)
        output_location = "RawTrainingFeatures.csv"
        config_file_location = "opensmile-2.3.0/config/gemaps/eGeMAPSv01a.conf"
        label_index = 5
        symbol = filename[label_index]
        label = convert_label_symbol(symbol)
        name = filename.replace(".wav", "")
        cmd = "%s -C %s -I %s -O %s -class %s -N %s" % ("SMILExtract", config_file_location, full_file_path, output_location, label, name)
        print(cmd)
        subprocess.call(cmd, shell=True)


def convert_label_symbol(symbol):
    symbol_to_label = {
        "W": "anger",
        "L": "boredom",
        "E": "disgust",
        "A": "fear",
        "F": "happiness",
        "T": "sadness",
        "N": "neutral"
    }
    # also can use one-hot encoding
    return symbol_to_label[symbol]


extract("EMO-DB/wav")
