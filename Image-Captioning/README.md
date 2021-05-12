<h1 align="center"> Image Captioning :camera: :arrow_right: :memo: </h1> 
 
 ***A sequence to sequence model to caption images built using PyTorch. This model uses inceptionV3 as encoder and LSTM layers as decoder. This model is trained on Flickr30k dataset.***

## Demo
  <p align="center"> <img src="https://github.com/Koushik0901/AI-Portfolio/blob/master/Image-Captioning/test_examples/surfing.png" width="500" height="320"  /> </p>
<p align="center"> <b>Prediction: <i>a man in wetsuit is surfing .</i></b>
 
<p align="center"> <img src="https://github.com/Koushik0901/AI-Portfolio/blob/master/Image-Captioning/test_examples/dirt_bike.png" width="500" height="320"  /> </p>
<p align="center"> <b>Prediction: <i>a man in blue helmet is riding a dirt bike on a dirt track .</i></b>

<p align="center"> <img src="https://github.com/Koushik0901/AI-Portfolio/blob/master/Image-Captioning/test_examples/dog.png" width="500" height="320"  /> </p>
<p align="center"> <b>Prediction: <i>a dog is running on the beach .</i></b>

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
