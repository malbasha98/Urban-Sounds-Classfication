from model import *
from settings import *
from data import *
import matplotlib.pyplot as plt

########################################################################################################################
# START OF K-FOLD CROSS VALIDATION
########################################################################################################################

print(features.shape)
acc_per_fold=[]
loss_per_fold=[]
histories=[]
fold_idx=np.arange(1, 11)
for fold in fold_idx:
    test_X=features[np.where(folds==fold)]
    test_Y=labels[np.where(folds==fold)]
    train_X=features[np.where(folds!=fold)]
    train_Y=labels[np.where(folds!=fold)]

    if(len(train_X.shape)==3): # if audio files are mono
        print("reshapeing")
        test_X=tf.reshape(test_X, [test_X.shape[0], test_X.shape[1], test_X.shape[2],1])
        train_X=tf.reshape(train_X, [train_X.shape[0], train_X.shape[1], train_X.shape[2],1])

    model = create_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=[tf.keras.metrics.categorical_accuracy])
    path=MFCC_folder+"/fold_"+str(fold)+"/"
    save_model = tf.keras.callbacks.ModelCheckpoint(path + "training_weights.h5",
                                                    monitor='val_categorical_accuracy', mode='max',
                                                    verbose=1, save_best_only=True,save_weights_only=True)

    csv_logger = tf.keras.callbacks.CSVLogger(path + 'training.csv')

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', mode='max', verbose=1,
                                                      patience=early_stopping_patience, restore_best_weights=True)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_categorical_accuracy', mode='max', verbose=1,
                                                     factor=0.1, patience=reduce_lr_patience)

    # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold} ...')

    start = datetime.now()

    history = model.fit(train_X,train_Y,
                        batch_size = batch_size,
                        validation_data=(test_X, test_Y),
                        epochs=50,
                        callbacks=[save_model, csv_logger, early_stopping, reduce_lr])

    duration = datetime.now() - start

    histories.append(history)

    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.savefig(path + "training_loss_plot"+str(fold)+".png")

    plt.clf()
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.savefig(path + "training_accuracy_plot"+str(fold)+".png")

    ########################################################################################################################
    # EVALUATE MODEL
    ########################################################################################################################

    scores = model.evaluate(test_X, test_Y, verbose=0)
    print(f'Score for fold {fold}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])
