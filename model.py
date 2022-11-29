from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Input, BatchNormalization, RepeatVector, TimeDistributed, Activation, dot, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import RootMeanSquaredError

class Seq2SeqModel():
    def __init__(self, train_input_seq, train_output_seq):
        self.model = None
        self.train_input_seq = train_input_seq
        self.train_output_seq = train_output_seq
        self.n_hidden = 100
        self.epochs = 200
        self.patience = 50
        self.loss = 'mean_squared_error'
        self.learning_rate = 0.01
        self.metrics = ['mae', RootMeanSquaredError()]
        self.history = None

    def build_model(self):
        input_train = Input(shape=(self.train_input_seq.shape[1], self.train_input_seq.shape[2]))
        output_train = Input(shape=(self.train_output_seq.shape[1], self.train_output_seq.shape[2]))
        encoder_stack_h, encoder_last_h, encoder_last_c = LSTM(self.n_hidden, activation='elu', dropout=0.2, recurrent_dropout=0.2, return_state=True, return_sequences=True)(input_train)
        encoder_last_h = BatchNormalization(momentum=0.6)(encoder_last_h)
        encoder_last_c = BatchNormalization(momentum=0.6)(encoder_last_c)
        decoder_input = RepeatVector(output_train.shape[1])(encoder_last_h)
        decoder_stack_h = LSTM(self.n_hidden, activation='elu', dropout=0.2, recurrent_dropout=0.2, return_state=False, return_sequences=True)(decoder_input, initial_state=[encoder_last_h, encoder_last_c])
        attention = dot([decoder_stack_h, encoder_stack_h], axes=[2, 2])
        attention = Activation('softmax')(attention)
        context = dot([attention, encoder_stack_h], axes=[2,1])
        context = BatchNormalization(momentum=0.6)(context)
        decoder_combined_context = concatenate([context, decoder_stack_h])
        out = TimeDistributed(Dense(output_train.shape[2]))(decoder_combined_context)
        self.model = Model(inputs=input_train, outputs=out)

    def compile_model(self):
        self.model.compile(loss=self.loss, optimizer=Adam(learning_rate=self.learning_rate, clipnorm=1), metrics=self.metrics)

    def fit_model(self):
        self.build_model()
        self.compile_model()
        es = EarlyStopping(monitor='val_loss', mode='min', patience=self.patience)
        self.history = self.model.fit(  self.train_input_seq, self.train_output_seq, 
                                        validation_split=0.2, epochs=self.epochs, verbose=1, 
                                        callbacks=[es], batch_size=10)

    def predict(self, data_to_predict):
        pred = self.model.predict(data_to_predict)
        return pred

    def predict_next_24hours(self, data_to_predict):
        temp = data_to_predict.copy()
        for _ in range(4):
            temp_pred = self.model.predict(temp.values.reshape((1, 24, 3)))
            temp = temp.shift(-6)
            temp[18:] = temp_pred[0]
        return temp.users.values[23]