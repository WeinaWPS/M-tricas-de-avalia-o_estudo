# M-tricas-de-avalia-o_estudo

!pip install --upgrade pip setuptools wheel


!pip install tensorflow==2.12.0


from tensorflow.keras import datasets, layers, models # Use layers instead of layer
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import seaborn as sns
from sklearn.metrics import confusion_matrix

tf.__version__
'2.12.0'

%load_ext tensorboard

lodir='log'

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000,28,28,1))

train_images, test_images = train_images / 255.0, test_images / 255.0

classes=[0,1,2,3,4,5,6,7,8,9]

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

logdir = 'log' 
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x=train_images,
          y=train_labels,
          epochs=5,
          validation_data=(test_images, test_labels),
          callbacks=[tensorboard_callback])

y_true=test_labels
y_pred=model.predict(test_images)
y_pred=np.argmax(y_pred,axis=1)

classes=[0,1,2,3,4,5,6,7,8,9]

import pandas as pd
con_matr=tf.math.confusion_matrix(labels=y_true,predictions=y_pred).numpy()
con_matr_norm=np.around(con_mat.astype('float')/con_mat.sum(axis=1)[:,np.newaxis],decimals=2)

con_matr_df = pd.DataFrame(con_mat_norm,
                        index=classes,
                        columns=classes)

figure = plt.figure(figsize=(8, 8))
sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

model_=models.Sequential()
model_.add(layers.Conv2D(22, (2, 2), activation='relu', input_shape=(29, 29, 1)))
model_.add(layers.MaxPooling2D((2, 2)))
model_.add(layers.Conv2D(44, (2, 2), activation='relu'))
model_.add(layers.MaxPooling2D((2, 2)))
model_.add(layers.Conv2D(66, (2, 2), activation='relu'))

model_.add(layers.Flatten())
model_.add(layers.Dense(66, activation='relu'))
model_.add(layers.Dense(10, activation='softmax'))

model_.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

file_writer = tf.summary.create_file_writer(logdir + "/metrics")
file_writer.set_as_default()

def lr_scheduler(epoch):
    if epoch < 10:
        return 0.001
    else:
        return 0.001 * tf.math.exp(0.1 * (10 - epoch))

    def log_confusion_matrix(epoch, logs):
   
        test_pred = model_.predict_classes(test_images)
        con_mat = tf.math.confusion_matrix(labels=test_labels,predictions=test_pred).numpy()
        con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)

        con_mat_df = pd.DataFrame(con_mat_norm,
                                  index=classes,
                                  columns=classes)
        
        fig = plt.figure(figsize=(8, 8))
        sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        but = io.BytesIO()
        plt.savefig(but, format='png')

        plt.close(fig)
        but.seek(0)
        image = tf.image.decode_png(but.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)

        with file_writer.as_default():
            tf.summary.image("Confusion Matrix", image, step=epoch)

            logdir = "logs/image/" + datetime.now().strftime("%Y%m%d-%H%M%S")
            file_writer = tf.summary.create_file_writer(logdir)

            file_writer.set_as_default()

            tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)



import tensorflow as tf
from tensorflow import keras
  
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logdir)

cm_callback = keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)


 #importando a biblioteca necessária
from sklearn.metrics import classification_report

  #exemplo de valores
y_true = [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
y_pred = [0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1]

  #evocando o report de classificação
print(classification_report(y_true, y_pred))

mat = confusion_matrix(y_true, y_pred)
sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel('Previsão do modelo')
plt.ylabel('Valor verdadeiro');

# problema de classificação com múltiplas classes
y_true = ["cat", "ant", "cat", "cat", "ant", "bird"]
y_pred = ["ant", "ant", "cat", "cat", "ant", "cat"]

mat = confusion_matrix(y_true, y_pred)
sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel('Previsão do modelo')
plt.ylabel('Valor verdadeiro');

# exemplo do classification Report
print(classification_report(y_true, y_pred))

from sklearn.model_selection import cross_val_score
clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf, X, y, cv=5)
scores



print("Acuracia: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
