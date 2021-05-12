<h1 align="center"> Image Captioning :camera: :arrow_right: :memo: </h1> 
 
 ***A sequence to sequence model to caption images built using PyTorch. This model uses inceptionV3 as encoder and LSTM layers as decoder. This model is trained on Flickr30k dataset.***

## Demo
  
## Running on native machine
### dependencies
* python3
* `python -m spacy download en` - for tokenizing english sentences  

### pip packages
`pip install -r requirements.txt` 

## Steps to train your own model
  ### Scripts
  `neuralnet/train.py` - is used to train the model  
  
  `engine.py` - is used to perform inference 
 
 `ui.py` - is used to build the streamlit app
    
  For more details make sure to visit these files to look at script arguments and description
  
  1. Dataset  
    i. Download the [Flickr30k](https://www.kaggle.com/hsankesara/flickr-image-dataset) dataset  
    ii. Remove the duplicate images folder and csv file  
  
  2. Training  
    use train.py to train the model  
