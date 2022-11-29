import preprocess
import model
import time
from memory_profiler import profile

@profile
def main():
    data = preprocess.get_data()
    train_input_seq, test_input_seq, train_output_seq, test_output_seq = preprocess.create_train_test_data()
    seq2seq = model.Seq2SeqModel(train_input_seq,train_output_seq)
    seq2seq.fit_model()
    print("Training completed ...")
    print("30.12.18 09:00:00 User Count: {}".format(int(seq2seq.predict_next_24hours(data[145:]))))
    pass

if __name__ == '__main__':

    start = time.time()
    main()
    end = time.time()
    print("System complete time: {:.2f} seconds".format((end-start)))
    pass