# NL-CNN A compact and fast trainable convolutional neural net
A fast yet light convolutional neural network model suitable for small/medium input image sizes (up to 64x64)


A novel compact yet accurate convolution neural model is proposed. Particularly for small or medium image sizes (up to 128x128), NL-CNN provides a simple and efficient solution, with relatively fast training allowing a fine optimization of the hyper-parameters. Accuracies are in the range of values obtained with state-of-the art solutions and in some cases larger than what is reported so far. Our model compares favorably to well established resources-constrained  models with faster training (around 3 times) better test accuracy and up to 10 times less complexity (measured as memory occupied by the .h5 model file).   
Relevant features of the proposed architecture: 
i) Quite accurate, given the reduced complexity, compares favorably to MobileNet; 
ii) Relatively simple to understand (and implement, in HW oriented solutions such as FPGA); 
iii) Training speed is better than in MobileNet thus allowing for a rapid and careful optimization of the hyper-parameters for the best performance; 
iv) The code can run and was tested in Google COLAB

If you find this code useful please cite the following work:
