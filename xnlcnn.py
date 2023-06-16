Celula definire model 

# xNL_CNN MODEL 
# Returns a precompiled model with a specific optimizer included 
#-------- extended NL-CNN model (improved from https://github.com/radu-dogaru/NL-CNN-a-compact-fast-trainable-convolutional-neural-net)
# add_layer may now range from 0 to 4 to cope with large image sizes 
# k1 is a coefficient to control de filters in the additional layers (add_layers), a value ranging from 0.7 to 1 is Ok 
#-----------------------------------------------------------------------------------------------------
# Copright Radu & Ioana Dogaru ; Last update March 2023 
#==============================================================================================
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers  import BatchNormalization
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, SeparableConv2D  # straturi convolutionale si max-pooling 
from tensorflow.keras.optimizers import  SGD, Adadelta, Adam, Nadam, RMSprop


def create_xnlcnn_model(input_shape, num_classes, k=1.5, k1=1.5, separ=0, flat=0, width=80, nl=(3,2), add_layer=0):
  # Arguments: k - multiplication coefficient 
  # Structure parameteres 
  kfil=k
  filtre1=width ; filtre2=int(kfil*filtre1) ; filtre3=(kfil*filtre2)  # filters (kernels) per each layer - efic. pe primul 
  nr_conv=3 # 0, 1, 2 sau 3  (number of convolution layers)
  csize1=3; csize2=3 ; csize3=3      # convolution kernel size (square kernel) 
  psize1=4; psize2=4 ; psize3=4      # pooling size (square)
  str1=2; str2=2; str3=2             # stride pooling (downsampling rate) 
  pad='same'; # padding style ('valid' is also an alternative)
  nonlinlayers1=nl[0]  # total of layers (with RELU nonlin) in the first maxpool layer  # De parametrizat asta 
  nonlinlayers2=nl[1]  # 

  nonlin_type='relu' # may be other as well 'tanh' 'elu' 'softsign'
  bndrop=1 # include BatchNorm inainte de MaxPool si drop(0.3) dupa .. 
  cvdrop=1 # droput 
  drop_cv=0.5
  
  model = Sequential()
  # convolution layer1  ==========================================================================
  # Initially first layer was always a Conv2D one
  if separ==1:
    model.add( SeparableConv2D(filtre1, padding=pad, kernel_size=(csize1, csize1), input_shape=input_shape) )
  elif separ==0: 
    model.add( Conv2D(filtre1, padding=pad, kernel_size=(csize1, csize1), input_shape=input_shape) )

  # next are the additional layers 
  for nl in range(nonlinlayers1-1):
    model.add(Activation(nonlin_type))  # Activ NL-CNN-1
    if separ==1:
      model.add(SeparableConv2D(filtre1, padding=pad, kernel_size=(csize1, csize1) ) ) # Activ NL-CNN-2
    elif separ==0:
      model.add(Conv2D(filtre1, padding=pad, kernel_size=(csize1, csize1)) ) # Activ NL-CNN-2
  #  MaxPool in the end of the module 
  if bndrop==1:
    model.add(BatchNormalization())
  model.add(MaxPooling2D(pool_size=(psize1, psize1),strides=(str1,str1),padding=pad))
  if cvdrop==1:
    model.add(Dropout(drop_cv))
  
  # NL LAYER 2 =======================================================================================================
 
  if separ==1:
    model.add(SeparableConv2D(filtre2, padding=pad, kernel_size=(csize2, csize2)) )
  elif separ==0:
    model.add(Conv2D(filtre2, padding=pad, kernel_size=(csize2, csize2)) )
  # aici se adauga un neliniar 
    
  #=========== unul extra NL=2 pe strat 2 =====================
  for nl in range(nonlinlayers2-1):
    model.add(Activation(nonlin_type))  # Activ NL-CNN-1
    if separ==1:
        model.add(SeparableConv2D(filtre2, padding=pad, kernel_size=(csize2, csize2)) ) # Activ NL-CNN-2
    elif separ==0:
        model.add(Conv2D(filtre2, padding=pad, kernel_size=(csize2, csize2)) ) # Activ NL-CNN-2
        
  # OUTPUT OF LAYER 2 (MAX-POOL)
  if bndrop==1:
      model.add(BatchNormalization())
  model.add(MaxPooling2D(pool_size=(psize2, psize2),strides=(str2,str2),padding=pad))
  if cvdrop==1:
      model.add(Dropout(drop_cv))
  #-------------------------------------------------------------------------------------------
  # LAYER 3 
      
  if separ==1:
      model.add(SeparableConv2D(filtre3, padding=pad, kernel_size=(csize3, csize3)) )  # SeparableConv
  elif separ==0:
      model.add(Conv2D(filtre3, padding=pad, kernel_size=(csize3, csize3)) ) # Activ NL-CNN-2
  # OUTPUT OF LAYER 3 
  if bndrop==1:
      model.add(BatchNormalization())
  model.add(MaxPooling2D(pool_size=(psize3, psize3),strides=(str3,str3),padding=pad))
  if cvdrop==1:
      model.add(Dropout(drop_cv))
  #------------------- 
  # 
  # LAYER 4  (only if requested - for large images ?? )
  if add_layer>=1:    
    if separ==1:
      model.add(SeparableConv2D(k1*filtre3, padding=pad, kernel_size=(csize3, csize3)) )  # SeparableConv
    elif separ==0:
      model.add(Conv2D(k1*filtre3, padding=pad, kernel_size=(csize3, csize3)) ) # Activ NL-CNN-2
    # OUTPUT OF LAYER 4
    if bndrop==1:
      model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(psize3, psize3),strides=(str3,str3),padding=pad))
    if cvdrop==1:
      model.add(Dropout(drop_cv))
  if add_layer>=2:
    if separ==1:
      model.add(SeparableConv2D(k1*k1*filtre3, padding=pad, kernel_size=(csize3, csize3)) )  # SeparableConv
    elif separ==0:
      model.add(Conv2D(k1*k1*filtre3, padding=pad, kernel_size=(csize3, csize3)) ) # Activ NL-CNN-2
    # OUTPUT OF LAYER 5
    if bndrop==1:
      model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(psize3, psize3),strides=(str3,str3),padding=pad))
    if cvdrop==1:
      model.add(Dropout(drop_cv))
  if add_layer>=3:
    if separ==1:
      model.add(SeparableConv2D(k1*k1*k1*filtre3, padding=pad, kernel_size=(csize3, csize3)) )  # SeparableConv
    elif separ==0:
      model.add(Conv2D(k1*k1*k1*filtre3, padding=pad, kernel_size=(csize3, csize3)) ) # Activ NL-CNN-2
    # OUTPUT OF LAYER 5
    if bndrop==1:
      model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(psize3, psize3),strides=(str3,str3),padding=pad))
    if cvdrop==1:
      model.add(Dropout(drop_cv))
  if add_layer==4:
    if separ==1:
      model.add(SeparableConv2D(k1*k1*k1*k1*filtre3, padding=pad, kernel_size=(csize3, csize3)) )  # SeparableConv
    elif separ==0:
      model.add(Conv2D(k1*k1*k1*filtre3, padding=pad, kernel_size=(csize3, csize3)) ) # Activ NL-CNN-2
    # OUTPUT OF LAYER 5
    if bndrop==1:
      model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(psize3, psize3),strides=(str3,str3),padding=pad))
    if cvdrop==1:
      model.add(Dropout(drop_cv))
  #========================================================================================
  # INPUT TO DENSE LAYER (FLATTEN - more data can overfit / GLOBAL - less data - may be a good choice ) 
  if flat==1:
      model.add(Flatten())  # 
  elif flat==0:
      model.add(GlobalAveragePooling2D()) # Global average 
  #model.add(Dense(500,activation='relu')) 
  model.add(Dense(num_classes, activation='softmax'))
  # END OF MODEL DESCRIPTION 

  return model


# Constructing a XNL-CNN model 
with strategy.scope():

    model=create_xnlcnn_model(input_shape=[*IMAGE_SIZE, 3], num_classes=NUM_CLAS, k=2, k1=0.7, separ=0, flat=0, width=50, nl=(2,2), add_layer=4)
    # - cele de mai sus se comenteaza daca s-a invocat modelul de mai sus (fara apel functie)

 

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss = 'sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy'],
    #steps_per_execution=16  
)
model.summary()
