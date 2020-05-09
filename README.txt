### Introduction
This image captioning is roughly based on the paper "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention" by Xu et al. (ICML2015).
The input is an image, and the output is a sentence describing the content of the image. It uses a convolutional neural network to extract visual features from the image, and uses a LSTM recurrent neural network to decode these features into a sentence.
A soft attention mechanism is incorporated to improve the quality of the caption. This project is implemented using the Tensorflow library, and allows end-to-end training of both CNN and RNN parts.

### Prerequisites
- Tensorflow (https://www.tensorflow.org/install/)
- NumPy (https://scipy.org/install.html))
- OpenCV (https://pypi.python.org/pypi/opencv-python)
- Natural Language Toolkit (NLTK) (http://www.nltk.org/install.html)
- Pandas(https://scipy.org/install.html)
- Matplotlib (https://scipy.org/install.html)
- tqdm (https://pypi.python.org/pypi/tqdm))

### Code folder structure

image_captioning-eval
├── README.txt
├── base_model.py
├── config.py
├── dataset.py
├── eval.sh
├── imgSep.py
├── main.py
├── model.py
├── models
│   ├── ResN-people-model
│   ├── VGG-People-Model
├── person_images
│   ├── images
├── summary
├── test
│   ├── images
│   ├── results
│   ├── results.csv
├── train
│   ├── images
│   ├── captions_train2014.json
│   ├── instances_train2014.json
├── val
│   ├── People_Results
│   ├── all_results
│   ├── captions_val2014.json
│   ├── images
│   ├── instances_val2014.json
│   ├── results
├── utils
│   ├── coco
│   │   ├── blue
│   │   ├── cider
│   │   ├── meteor
│   │   ├── rouge
│   │   ├── spice
│   ├── misc.py
│   ├── nn.py
│   └── vocabulary.py
├── vgg16_no_fc.npy
└── vocabulary.csv


### Usage
***Preparation:***
Download the COCO train2014 and val2014 data (http://cocodataset.org/#download).
Put the COCO train2014 images in the folder `train/images`, and put the file `captions_train2014.json` in the folder `train`.
Similarly, put the COCO val2014 images in the folder `val/images`, and put the file `captions_val2014.json` in the folder `val`.
Furthermore, use the pretrained VGG16 net 'vgg16_no_fc.npy' or ResNet50 net'resnet50_no_fc.npy' if you want to use it to initialize the CNN part.

use imgSep.py file to seperate 'People' images from rest od the dataset.

***Training:***
To train a model using the COCO train2014 data, first setup various parameters in the file `config.py` and then run a command like this:

    > python main.py --phase=train --load_cnn --cnn_model_file='./vgg16_no_fc.npy'[--train_cnn]

    Turn on `--train_cnn` if you want to jointly train the CNN and RNN parts. Otherwise, only the RNN part is trained.

The checkpoints will be saved in the folder `models`. If you want to resume the training from a checkpoint, run a command like this:

    > python main.py --phase=train --load --model_file='./models/xxxxxx.npy'\ [--train_cnn]

To monitor the progress of training, run the following command:

    > tensorboard --logdir='./summary/'


***Evaluation:***
To evaluate a trained model using the COCO val2014 data, run a command like this:

    > python main.py --phase=eval --model_file='./models/xxxxxx.npy' --beam_size=3

The result will be shown in stdout. Furthermore, the generated captions will be saved in the file `val/results.json`.

***Inference:***
You can use the trained model to generate captions for any JPEG images! Put such images in the folder `test/images`, and run a command like this:

  > python main.py --phase=test --model_file='./models/xxxxxx.npy' --beam_size=3

The generated captions will be saved in the folder `test/results`.


### References
- Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/abs/1502.03044). Kelvin Xu, Jimmy Ba, Ryan Kiros, Kyunghyun Cho, Aaron Courville, Ruslan Salakhutdinov, Richard Zemel, Yoshua Bengio. ICML 2015.
- The original implementation in Theano(https://github.com/kelvinxu/arctic-captions)
- An earlier implementation in Tensorflow(https://github.com/jazzsaxmafia/show_attend_and_tell.tensorflow)
- Microsoft COCO dataset(http://mscoco.org/)
- TensorBoard: TensorFlow's visualization toolkit (https://www.tensorflow.org/tensorboard)
- TensorFlow-Examples (https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/4_Utils/tensorboard_basic.py)
- Freezing layer (https://stackoverflow.com/questions/46610732/how-to-freeze-some-layers-when-fine-tune-resnet50)
- Deep Learning using Transfer Learning -Python Code for ResNet50 (https://towardsdatascience.com/deep-learning-using-transfer-learning-python-code-for-resnet50-8acdfb3a2d38)
