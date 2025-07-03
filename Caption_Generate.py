from keras.models import load_model
from utils.preprocess import *
import pickle

model = load_model('models/caption_model.h5')
with open('models/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

max_len = 34  # from training
photo = extract_features('image.jpg')  # Image to be added according to preference

def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = next((w for w, i in tokenizer.word_index.items() if i == yhat), None)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text[9:-7] 

caption = generate_desc(model, tokenizer, list(photo.values())[0])
print("Generated Caption:", caption)
