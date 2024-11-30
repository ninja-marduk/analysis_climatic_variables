def fn_get_model_convLSTM_tframe_5():
    
    model = Sequential()
    model.add(ConvLSTM2D(filters=64, kernel_size=(7, 7),
                         input_shape=(None, 101, 101, 1), padding='same', return_sequences=True, 
                         activation='tanh', recurrent_activation='hard_sigmoid',
                         kernel_initializer='glorot_uniform', unit_forget_bias=True, 
                         dropout=0.3, recurrent_dropout=0.3, go_backwards=True ))
    model.add(BatchNormalization())

    model.add(ConvLSTM2D(filters=32, kernel_size=(7, 7), padding='same', return_sequences=True, 
                         activation='tanh', recurrent_activation='hard_sigmoid', 
                         kernel_initializer='glorot_uniform', unit_forget_bias=True, 
                         dropout=0.4, recurrent_dropout=0.3, go_backwards=True ))
    model.add(BatchNormalization())
    
    model.add(ConvLSTM2D(filters=32, kernel_size=(7, 7), padding='same', return_sequences=True, 
                         activation='tanh', recurrent_activation='hard_sigmoid', 
                         kernel_initializer='glorot_uniform', unit_forget_bias=True, 
                         dropout=0.4, recurrent_dropout=0.3, go_backwards=True ))
    model.add(BatchNormalization())


    model.add(ConvLSTM2D(filters=32, kernel_size=(7, 7), padding='same', return_sequences=False, 
                         activation='tanh', recurrent_activation='hard_sigmoid', 
                         kernel_initializer='glorot_uniform', unit_forget_bias=True, 
                         dropout=0.4, recurrent_dropout=0.3, go_backwards=True ))
    model.add(BatchNormalization())
    
    model.add(Conv2D(filters=1, kernel_size=(1, 1),
                   activation='sigmoid',
                   padding='same', data_format='channels_last')) 
    
    print(model.summary())
    return model