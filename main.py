from model import *

#Primer ucitavanja tezina
model_=create_model()
model_.load_weights("MFCC/fold_1/training_weights.h5")
print(model_.weights)