# NL-CNN A compact and fast trainable convolutional neural net
A fast yet light convolutional neural network model suitable for small/medium input image sizes (up to 64x64)

Code is avalailable as the notebook nl_cnn_demos.ipynb 

<a href="https://colab.research.google.com/github/radu-dogaru/NL-CNN-a-compact-fast-trainable-convolutional-neural-net/blob/main/nl_cnn_demos.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


It includes the function defining the NL-CNN Keras/Tensorflow model and some relevant trainied models for a diversity of datasets: MNIST, EMNIST, Fashion-MNIST, FER-2013, SVHH, GTRSB

A novel compact yet accurate convolution neural model is proposed. Particularly for small or medium image sizes (up to 128x128), NL-CNN provides a simple and efficient solution, with relatively fast training allowing a fine optimization of the hyper-parameters. Accuracies are in the range of values obtained with state-of-the art solutions and in some cases larger than what is reported so far. Our model compares favorably to well established resources-constrained  models with faster training (around 3 times) better test accuracy and up to 10 times less complexity (measured as memory occupied by the .h5 model file).   
Relevant features of the proposed architecture: 
i) Quite accurate, given the reduced complexity, compares favorably to MobileNet; 
ii) Relatively simple to understand (and implement, in HW oriented solutions such as FPGA); 
iii) Training speed is better than in MobileNet thus allowing for a rapid and careful optimization of the hyper-parameters for the best performance; 
iv) The code can run and was tested in Google COLAB

If you find this code useful please cite the following work:

Radu Dogaru, Ioana Dogaru, "NL-CNN: A Resources-Constrained Deep Learning Model based on Nonlinear Convolution", submitted to IEEE ATEE-2021 conference, 2021 
Reprint here: https://arxiv.org/ftp/arxiv/papers/2102/2102.00227.pdf

**NEW (June, 2023) The XNL-CNN model** is added (for larger image sizes). Code here: https://github.com/radu-dogaru/NL-CNN-a-compact-fast-trainable-convolutional-neural-net/blob/main/xnlcnn.py  
Details in an article to be published soon 
