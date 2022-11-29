import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

INPUT_SEQ_LEN = 24
OUTPUT_SEQ_LEN = 6

def get_data():
    data = pd.read_csv('data/app.csv', sep=';')
    data.time = pd.to_datetime(data.time)
    data.drop('crashes', axis=1, inplace=True)
    data = data.set_index('time')
    return data

def generate_train_sequences(x, input_seq_len, output_seq_len):
    
    total_start_points = len(x) - input_seq_len - output_seq_len
    start_x_idx = np.random.choice(range(total_start_points), total_start_points, replace = False)
    
    input_batch_idxs = [(range(i, i+input_seq_len)) for i in start_x_idx]
    input_seq = np.take(x, input_batch_idxs, axis = 0)
    
    output_batch_idxs = [(range(i+input_seq_len, i+input_seq_len+output_seq_len)) for i in start_x_idx]
    output_seq = np.take(x, output_batch_idxs, axis = 0)
    
    input_seq =(input_seq.reshape(input_seq.shape[0],input_seq.shape[1],3))
    output_seq=(output_seq.reshape(output_seq.shape[0],output_seq.shape[1],3))
    
    return input_seq, output_seq

def create_train_test_data():
    data = get_data()
    input_seq, output_seq = generate_train_sequences(data.values, INPUT_SEQ_LEN, OUTPUT_SEQ_LEN)
    train_input_seq, test_input_seq, train_output_seq, test_output_seq = train_test_split(input_seq, output_seq, test_size=0.20, random_state=42)
    return train_input_seq, test_input_seq, train_output_seq, test_output_seq
