# CaptionFlow
CaptionFlow is a custom trained Vision Language Model that generates a caption based on the image provided. It is a transformer based model

# Contents
1. Dataset
2. Model architecture
3. Training
4. Testing

# 1. Dataset
The model has been trained on the COCO 2017 dataset. [Click Here to go to Dataset](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset)

# 2. Model architecture
- CNN Encoder: Used InceptionV3 to extract image features.
- Transformer Encoder Layer: Implemented a Transformer encoder layer.
- Transformer Decoder Layer: Implemented a Transformer decoder layer.
- ImageCaptioningModel: Combined CNN and Transformer components, with methods for training and evaluation.

# 3. Training
- The model was trained for 10 epochs with a batch size of 32.
- Loss function used was BinaryCrossentropy
- Adam was used as optimizer with learning rate of 0.0001

# 4. Testing
From the predicted captions we can see that the model does good enough in some images. More accuracy and relevant output can be attained using complex model and more data
