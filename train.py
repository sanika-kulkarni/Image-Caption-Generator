from keras.models import Model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from utils.preprocess import *
import numpy as np
import pickle

# Load dataset and features
descriptions = load_descriptions('data/Flickr8k_text/Flickr8k.token.txt')
features = extract_features('data/Flickr8k_Dataset/')

tokenizer = create_tokenizer(descriptions)
vocab_size = len(tokenizer.word_index) + 1
max_len = max_length(descriptions)

# Data generator
def data_generator(descriptions, photos, tokenizer, max_len, vocab_size):
    while True:
        for key, desc_list in descriptions.items():
            photo = photos[key][0]
            for desc in desc_list:
                seq = tokenizer.texts_to_sequences([desc])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_len)[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    yield [photo, in_seq], out_seq

# Model
inputs1 = Input(shape=(4096,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)

inputs2 = Input(shape=(max_len,))
se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256)(se2)

decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)

model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Train model
steps = sum(len(d) for d in descriptions.values())
generator = data_generator(descriptions, features, tokenizer, max_len, vocab_size)
model.fit(generator, epochs=10, steps_per_epoch=steps, verbose=1)

# Save
model.save('models/caption_model.h5')
with open('models/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
