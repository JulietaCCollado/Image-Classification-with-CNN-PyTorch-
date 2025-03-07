# Image Classification with CNN (Pytorch)

This project trains a **Convolutional Neural Network (CNN)** using **PyTorch** to classify images from the **CIFAR-10 dataset**. You can also use the train model to classify custom images.

## Project Structure
```bash
Image Classification/ 
 │──models/ #Store model definitions
    │── cnn.py # CNN model definition
 │──train.py # Training Script
 │──test.py # Model evaluation script
 │──predict.py # Make predictions on new images
 │── main.py # Runs the pipeline
 │── utils.py # Helper functions
 │── requirements.txt # Dependencies
 │── README.md # Project documentation
```
 
 ### Installation

 ## 1: **Clone the Repository**
```bash
git clone https://github.com/JulietaCCollado/Image-Classification-with-CNN-PyTorch-
```
## 2: Install Dependencies
```bash
pip install -r requirements.txt
```

## 3: Train the model
```bash
python train.py
```

## 4: Test the model
``` bash
python test.py
```

## 5: Predict a custom image
```bash
python predict.py Avion.jpg
```

## 6: Run the full pipeline
```bash
python main.py
```

## Author
👨‍💻 Created by Julieta Collado
📧 Contact: julietacollado98@hotmail.com
