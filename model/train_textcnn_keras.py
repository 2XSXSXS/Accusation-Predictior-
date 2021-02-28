import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Dense, Embedding, Input, Dropout
from keras.layers.merge import concatenate
from keras.layers import Conv1D, BatchNormalization, Activation, GlobalMaxPooling1D
from data_utils.evaluation import *
from sklearn.metrics import f1_score
from keras.utils.vis_utils import plot_model

embedding_dim = 256
num_words = 40000
maxlen = 400
label_type = 'accusation'

train_fact_pad_seq = np.load('./variables/pad_sequences/train_pad_%d_%d.npy' % (maxlen, num_words))
valid_fact_pad_seq = np.load('./variables/pad_sequences/valid_pad_%d_%d.npy' % (maxlen, num_words))
test_fact_pad_seq = np.load('./variables/pad_sequences/test_pad_%d_%d.npy' % (maxlen, num_words))

train_labels = np.load('./variables/labels/train_one_hot_%s.npy' % (label_type))
valid_labels = np.load('./variables/labels/valid_one_hot_%s.npy' % (label_type))
test_labels = np.load('./variables/labels/test_one_hot_%s.npy' % (label_type))

set_labels = np.load('./variables/label_set/set_%s.npy' % label_type)

num_classes = train_labels.shape[1]
num_filters = 512
num_hidden = 1000
batch_size = 64
num_epochs = 2
dropout_rate = 0.2

input = Input(shape=[train_fact_pad_seq.shape[1]], dtype='float64')
embedding_layer = Embedding(input_dim=num_words + 1,
                            input_length=maxlen,
                            output_dim=embedding_dim,
                            mask_zero=0,
                            name='Embedding')
embed = embedding_layer(input)

cnn1 = Conv1D(num_filters, 3, strides=1, padding='same')(embed)
relu1 = Activation(activation='relu')(cnn1)
cnn1 = GlobalMaxPooling1D()(relu1)
cnn2 = Conv1D(num_filters, 4, strides=1, padding='same')(embed)
relu2 = Activation(activation='relu')(cnn2)
cnn2 = GlobalMaxPooling1D()(relu2)
cnn3 = Conv1D(num_filters, 5, strides=1, padding='same')(embed)
relu3 = Activation(activation='relu')(cnn3)
cnn3 = GlobalMaxPooling1D()(relu3)
cnn4 = Conv1D(num_filters, 6, strides=1, padding='same')(embed)
relu4 = Activation(activation='relu')(cnn4)
cnn4 = GlobalMaxPooling1D()(relu4)
cnn = concatenate([cnn1, cnn2, cnn3, cnn4], axis=-1)
bn = BatchNormalization()(cnn)
drop1 = Dropout(dropout_rate)(bn)
dense = Dense(num_hidden, activation="relu")(drop1)
drop2 = Dropout(dropout_rate)(dense)
main_output = Dense(num_classes, activation='sigmoid')(drop2)
model = Model(inputs=input, outputs=main_output)

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

plot_model(model, to_file="./pics/textcnn_filter345.png",show_shapes=True)

for epoch in range(num_epochs):
    model.fit(x=train_fact_pad_seq, y=train_labels,
              batch_size=batch_size, epochs=1,
              validation_data=(valid_fact_pad_seq, valid_labels), verbose=1)

    predictions_valid = model.predict(valid_fact_pad_seq[:])

    predictions = predictions_valid

    sets = set_labels

    y1 = label2tag(valid_labels[:], sets)
    y2 = predict2toptag(predictions, sets)
    y3 = predict2half(predictions, sets)
    y4 = predict2tag(predictions, sets)

    s1 = [str(y1[i]) == str(y2[i]) for i in range(len(y1))]
    print(sum(s1) / len(s1))
    s2 = [str(y1[i]) == str(y3[i]) for i in range(len(y1))]
    print(sum(s2) / len(s2))
    s3 = [str(y1[i]) == str(y4[i]) for i in range(len(y1))]
    accuracy = int(np.round(sum(s3) / len(s3), 3) * 100)
    print(accuracy)

    predictions_one_hot = predict1hot(predictions)
    f1_micro = f1_score(valid_labels,predictions_one_hot,average='micro')
    print('f1_micro_accusation:', f1_micro)
    f1_marco = f1_score(valid_labels, predictions_one_hot, average='macro')
    print('f1_macro_accusation:', f1_marco)
    f1_average = int(np.round((f1_marco + f1_micro) / 2, 2) * 100)
    print('total:', f1_average)

    model.save('./model/textcnn_%s_token_%s_pad_%s_filter_%s_hidden_%s_epoch_%s_accu_%s_f1_%s.h5' % (
        label_type , num_words, maxlen, num_filters, num_hidden, epoch + 1, accuracy, f1_average))

    r = pd.DataFrame({'label': y1, 'predict': y4})
    r.to_excel('./results/textcnn_%s_token_%s_pad_%s_filter_%s_hidden_%s_epoch_%s_accu_%s_f1_%s.xlsx' % (
        label_type, num_words, maxlen, num_filters, num_hidden, epoch + 1, accuracy, f1_average),
               sheet_name='1', index=False)

