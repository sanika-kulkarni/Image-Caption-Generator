# Image-Caption-Generator
This project is an Image Caption Generator that combines Computer Vision (CV) and Natural Language Processing (NLP) to generate meaningful captions for images.  It uses a Convolutional Neural Network (CNN) (VGG16) to extract features from an image and a Recurrent Neural Network (RNN)(LSTM) to generate captions. 
#Input/Inputs:
- An image from the Flickr8k dataset

#Output/Outputs:
- A caption to the image like: "A group of students sitting in the classroom."

#Key Concepts Used
- CNN (VGG16): Extracts  the high-level features from images provided.
- LSTM (RNN): Generates sequence of words (captions).
- Tokenizer & Embedding: Conversion of text to numbers for training.
- Sequence Padding: Handles different lengths of captions efficiently.
- Training Forcing: Trains the model using word-by-word predictions.

#Credits
- [Flickr8k Dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k)
- Pre-trained CNN: VGG16 (Keras Applications)
