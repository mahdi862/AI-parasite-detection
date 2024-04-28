# TF 1.14 gives lots of warnings for deprecations ready for the switch to TF 2.0
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import glob

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds

from datetime import datetime
from tensorflow.keras.models import model_from_json
from tensorflow.keras.callbacks import Callback, CSVLogger, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Activation, Input, concatenate
from tensorflow.keras.layers import Layer, BatchNormalization, MaxPooling2D, Concatenate, Lambda, Flatten, Dense
from tensorflow.keras.initializers import glorot_uniform, he_uniform
from tensorflow.keras.regularizers import l2
# from tensorflow.keras.utils import multi_gpu_model

from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, roc_auc_score
import math
# from pylab import dist
import json

from tensorflow.python.client import device_lib
import matplotlib.gridspec as gridspec

import matplotlib.pyplot as plt

from skimage import data, color, io
from skimage.transform import rescale, resize, downscale_local_mean
import os, glob
import random
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from skimage import data, filters
from skimage.transform import rotate
import cv2, shutil

# from google.colab import drive
# drive.mount('/content/drive')
# drive.mount('/content/drive/',force_remount=True)

# base_path=

# output_folder=r"/Users/mahdi/Documents/mahdi/python_projects/oneShot_Model/HealthAI/normalize_image"
# input_folder=r"/Users/mahdi/Documents/mahdi/python_projects/oneShot_Model/HealthAI/images/"
# input_folder=r"/content/drive/MyDrive/HealthAI/Colabs/Baseline_augmented_photos"
# input_folder=r"/content/drive/MyDrive/images"
# input_folder_base=r"/content/drive/MyDrive/HealthAI/training_images"
import os

this_path = os.path.dirname(os.path.realpath(__file__))

# input_folder_base = r"./training_images"
input_folder_base= os.path.join(this_path,"training_images_best")
# input_folder=r"/content/drive/MyDrive/HealthAI/training_images_augmented"


input_folder = os.path.join(this_path,"training_images_augmented")
# if (os.path.exists(input_folder)):
#     print("input path exist")
#     shutil.rmtree(input_folder)
#     # os.rmdir(input_folder)
#     os.mkdir(input_folder)
# img_path={'Ascaris':0,   'Trichuris_trichiura':3,   'Enterobius_Vermicularis':7, 'Schistosoma_Mansoni':8, 'Taenia':9}

# pickle_file=r"/content/drive/MyDrive/HealthAI/Colabs/Baseline_augmented_photos/training_validation_dataset.pkl"
# pickle_file=r"/content/drive/MyDrive/images/training_validation_dataset.pkl"
pickle_file = os.path.join(input_folder,"training_validation_dataset.pkl")

# img_tag={'Ascaris':0,   'Trichuris_trichiura':3,   'Enterobius_Vermicularis':7, 'Schistosoma_Mansoni':8, 'Taenia':9}
# img_tag={'Ascaris':0}#,   'Trichuris_trichiura':3,   'Enterobius_Vermicularis':7, 'Schistosoma_Mansoni':8, 'Taenia':9}
# img_tag={'Ascaris':0,   'Trichuris_trichiura':1,   'Enterobius_Vermicularis':2, 'Schistosoma_Mansoni':3, 'Taenia':4}
img_tag = {'Ascaris': 0, 'Dicrocoelium': 1, 'Hymenolepis_nana': 2, 'Trichuris_trichiura': 3,
           'Schistosoma_Haematobium': 4, 'Dipylidium_Caninum': 5, 'Hookworm': 6, 'Enterobius_Vermicularis': 7,
           'Schistosoma_Mansoni': 8, 'Taenia': 9, 'Fasciola': 10}

# img_tag = {'Ascaris': 0,  'Trichuris_trichiura': 1,
#            'Schistosoma_Haematobium': 2,  'Taenia': 3}
#
# img_tag = {'Ascaris': 0,  'Hymenolepis_nana': 1,
#            'Schistosoma_Haematobium': 2, 'Dipylidium_Caninum': 3,
#            'Schistosoma_Mansoni': 4,  'Fasciola': 5}
# img_tag = {'Ascaris': 0,  'Schistosoma_Mansoni': 1}


# img_tag = {'Dicrocoelium': 0, 'Hymenolepis_nana': 1, 'Trichuris_trichiura': 2, 'Schistosoma_Haematobium': 3,
#            'Schistosoma_Mansoni': 4}

# img_tag = { 'Hymenolepis_nana': 0, 'Trichuris_trichiura': 1, 'Schistosoma_Haematobium': 2,
#            'Schistosoma_Mansoni': 3}

# img_tag={ 'Schistosoma_Mansoni':0,  'Fasciola':1}



# if (os.path.exists(input_folder)):
#   print ("input path exist")
# else:
#   print ("input path does not exist")


# if (os.path.exists(input_folder_base)):
#   print ("input base path exist")
# else:
#   print ("input base path does not exist")
load_model=0
img_size = 128
read_from_pickle = 0  # if it is 1, the data will be read from pickle file
training_percent = 0.9  # x60% of data will be used for training
generate_augmented_image = 1

if (generate_augmented_image == 1):
    if (os.path.exists(input_folder)):
        print("input path exist")
        shutil.rmtree(input_folder)
        # os.rmdir(input_folder)
        os.mkdir(input_folder)
    else:
        print("input path does not exist")
        os.mkdir(input_folder)

    all_image_folders = [x[1] for x in os.walk(input_folder_base)][0]
    # print(all_image_folders)

    for img_folder in all_image_folders:
        if (img_folder not in img_tag.keys()):
            continue
        image_folder_path = os.path.join(input_folder, img_folder)
        if (not os.path.exists(image_folder_path)):
            os.mkdir(image_folder_path)

        all_images_name = glob.glob(os.path.join(input_folder_base, img_folder, '*.*'))
        count = 0
        for img in all_images_name:
            img_file_name = os.path.basename(img)
            if (img_file_name.find("augmented") >= 0):
                continue
            img_path = os.path.join(input_folder_base, img_folder, img_file_name)
            # raw_img=data.load(img_path)
            try:
                raw_img = io.imread(img_path, as_gray=True)
            except:
                continue

            img0 = filters.sobel(raw_img)
            img0 = img0 / np.max(img0)

            m, n = img0.shape
            img1=img0

            if (np.abs(m - n) > 3):
                # img0 = filters.sobel(raw_img)
                # img0 = img0 / np.max(img0)
                # m, n = img0.shape
                h = plt.hist(img0.reshape((m * n, 1)), bins=np.arange(0, 1.0, 0.01))
                idx = np.where(np.max(h[0]) == h[0])[0][0]
                color_val = h[1][:-1][4]

                if (m < n):
                    d1 = int((n - m) / 2)
                    d2 = (n - m) - d1
                    v0 = np.random.rand(d1, n) * color_val
                    v1 = np.random.rand(d2, n) * color_val
                    img1 = np.vstack([v0, img0, v1])

                else:
                    d1 = int((m - n) / 2)
                    d2 = (m - n) - d1
                    v0 = np.random.rand(m, d1) * color_val
                    v1 = np.random.rand(m, d2) * color_val
                    img1 = np.hstack([v0, img0, v1])


            image_resized = resize(img1, (img_size, img_size), anti_aliasing=True) * 256
            # edges = filters.sobel(image_resized)
            edges = image_resized

            suffix = img.split('.')[-1]
            # print(suffix)
            ln_suffix = len(suffix) + 1

            img_path_augmented = os.path.join(input_folder, img_folder, img_file_name)

            # edges =edges /np.max(edges)

            img1_name = img_path_augmented[:-ln_suffix] + "_augmented1.jpg"
            cv2.imwrite(img1_name, edges)

            edge1 = rotate(edges, 90, resize=False)

            img2_name = img_path_augmented[:-ln_suffix] + "_augmented2.jpg"
            cv2.imwrite(img2_name, edge1)

            edge2 = rotate(edges, 180, resize=False)
            img3_name = img_path_augmented[:-ln_suffix] + "_augmented3.jpg"
            cv2.imwrite(img3_name, edge2)

            edge3 = rotate(edges, 270, resize=False)
            img4_name = img_path_augmented[:-ln_suffix] + "_augmented4.jpg"
            cv2.imwrite(img4_name, edge3)

            # print(img1_name ,img2_name)
    print("finish generating augmented images!")




if (read_from_pickle==0):
    # all_image_folders = [x[0] for x in os.walk(input_folder)]
    all_image_folders=[x[1] for x in os.walk(input_folder)][0]
    # print(all_image_folders)

    x_data=[]
    y_data=[]

    # img_tag={'mansoni':0, 'fasciola':1}
    # img_tag={'Ascaris':0, 'Dicrocoelium':1, 'Hymenolepis_nana':2, 'Trichuris_trichiura':3, 'Schistosoma_Haematobium':4, 'Dipylidium_Caninum':5, 'Hookworm':6, 'Enterobius_Vermicularis':7, 'Schistosoma_Mansoni':8, 'Taenia':9, 'Fasciola':10}
    # img_tag={'Ascaris':0,   'Trichuris_trichiura':3,   'Enterobius_Vermicularis':7, 'Schistosoma_Mansoni':8, 'Taenia':9}

    for img_folder in all_image_folders:
      if (img_folder not in img_tag.keys()):
        continue
      all_images_name = glob.glob(os.path.join(input_folder,img_folder,'*.*'))
      count=0
      for img in all_images_name:
          img_path=os.path.join(input_folder,img_folder,img)
          # raw_img=data.load(img_path)
          try:
              raw_img=io.imread(img_path,as_gray = True)
          except:
              continue
          # image = color.rgb2gray(raw_img)
          image_resized = resize(raw_img, (img_size, img_size),anti_aliasing=True)
          # image_resized=image_resized/np.max(image_resized)

          # edges = filters.sobel(image_resized)
          # edges=edges/np.max(edges)
          edges=image_resized

          x_data.append(edges)
          y_data.append(img_tag[img_folder])

    for i in range(len(y_data) - 1, 0, -1):
        # Pick a random index from 0 to i
        j = random.randint(0, i + 1)

        # Swap arr[i] with the element at random index
        y_data[i], y_data[j] = y_data[j], y_data[i]
        x_data[i], x_data[j] = x_data[j], x_data[i]

    n=len(y_data)
    training_len=int(training_percent*n)
    x_train=x_data[:training_len]
    y_train=y_data[:training_len]

    # x_val = x_data[training_len:]
    # y_val = y_data[training_len:]

    # x_train = x_data
    # y_train = y_data


    x_val = x_data
    y_val = y_data

    x_train=np.stack(x_train,axis=0)
    x_test=np.stack(x_val,axis=0)

    y_train=np.stack(y_train,axis=0)
    y_test=np.stack(y_val,axis=0)

    data_pkl=[x_train,y_train,x_test,y_test]
    with open(pickle_file, 'wb') as f:
        pkl.dump(data_pkl, f)

    # random.shuffle(y_data)
    # random.shuffle(y_data)
else:
    if (os.path.exists(pickle_file)):
        with open(pickle_file, "rb") as f:
            data_pkl=pkl.load(f)
            x_train=data_pkl[0]
            y_train=data_pkl[1]
            x_test=data_pkl[2]
            y_test=data_pkl[3]

            print (x_train.shape,x_test.shape)
    else:
        print ("pickle file does not exis!!!")

num_classes = len(np.unique(y_train))

x_train_w = x_train.shape[1] # (60000, 28, 28)
x_train_h = x_train.shape[2]
x_test_w = x_test.shape[1]
x_test_h = x_test.shape[2]

x_train_w_h = x_train_w * x_train_h # 28 * 28 = 784
x_test_w_h = x_test_w * x_test_h

print(x_train_w,x_train_h,x_train_w_h)
# x_train = x_train/255. # (60000, 784)
# x_test = x_test/255.
print(x_train.shape,x_test.shape)

print("catagory are: ",np.unique(y_train))
print("test catagory are: ",np.unique(y_test))

# plt.imshow(x_train[100,:,:],cmap='gray')
# edges = filters.sobel(x_train[100,:,:])
# plt.figure()
# plt.imshow(edges,cmap='gray')



def _normalize_img(img, label):
    img = tf.cast(img, tf.float32) / 1.

    # tf.print(label)
    return (img, label)

# train_dataset, test_dataset = tfds.load(name="mnist", split=['train', 'test'], as_supervised=True)

# Build your input pipelines
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
# print(len(train_dataset))

# with tf.compat.v1.Session()  as sess:
#     for adataset in train_dataset:
#       print(adataset[1].eval())


train_dataset = train_dataset.shuffle(len(train_dataset)).batch(256)
train_dataset = train_dataset.map(_normalize_img)

test_dataset = test_dataset.batch(256)
test_dataset = test_dataset.map(_normalize_img)



input_shape = (x_train_w, x_train_h,1)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=64, kernel_size=5, activation='relu', input_shape=input_shape),
    tf.keras.layers.MaxPooling2D(pool_size=5),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=3),
    tf.keras.layers.Dropout(0.3),
    # tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu'),
    # tf.keras.layers.MaxPooling2D(pool_size=3),
    # tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(2048, activation=None),  # No activation on final dense layer
    tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))  # L2 normalize embeddings

])

print(model.summary())

batch_size = 256
epochs = 2500
steps_per_epoch = int(x_train.shape[0] / batch_size)
val_steps = int(x_test.shape[0] / batch_size)
alpha = 0.3
num_hard = int(batch_size * 0.8)  # Number of semi-hard triplet examples in the batch
# lr = 0.01
lr = 0.003
optimiser = 'Adam'
emb_size = num_classes
validation_loss=[]
optimiser_obj = Adam(lr=lr)

## Use a custom non-dt name:
name = "snn-parasite-Allcatagory-run0-lessparams-Aug25"
log_main_dir=os.path.join(this_path,"traninig_log")
if not os.path.exists(log_main_dir):
    os.mkdir(log_main_dir)
logdir = os.path.join(log_main_dir,name)
if not os.path.exists(logdir):
    os.mkdir(logdir)

## Callbacks:
# Create the TensorBoard callback
tensorboard = tf.keras.callbacks.TensorBoard(
    log_dir=logdir,
    histogram_freq=0,
    batch_size=batch_size,
    write_graph=True,
    write_grads=True,
    write_images=True,
    update_freq='epoch',
    profile_batch=0
)


# Save the embedding mode weights based on the main model's val loss
# This is needed to reecreate the emebedding model should we wish to visualise
# the latent space at the saved epoch
class SaveEmbeddingModelWeights(Callback):
    def __init__(self, filepath, monitor='val_loss', verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.best = np.Inf
        self.filepath = filepath

    def on_epoch_end(self, epoch, logs={}):
        # v_loss = logs.get('val_loss')
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("SaveModelWeights requires %s available!" % self.monitor, RuntimeWarning)
        v_loss = logs.get('val_loss')
        # print ("current los, val_los, best_loss",current,v_loss,self.best)
        validation_loss.append(v_loss)
        # plt.plot(validation_loss)
        if current < self.best:
            filepath = self.filepath.format(epoch=epoch + 1,cal_loss=v_loss, **logs)
            # if self.verbose == 1:
            # print("Saving embedding model weights at %s" % filepath)
            model.save_weights(filepath, overwrite=True)
            self.best = current


# Training logger
csv_log = os.path.join(logdir, 'training.csv')
csv_logger = CSVLogger(csv_log, separator=',', append=True)

# Only save the best model weights based on the val_loss
checkpoint = ModelCheckpoint(os.path.join(logdir, 'snn_model-{epoch:02d}-{val_loss:.4f}.h5'),
                             monitor='val_loss', verbose=0,
                             save_best_only=True, save_weights_only=True,
                             mode='auto')

emb_weight_saver = SaveEmbeddingModelWeights(os.path.join(logdir, 'emb_model-{epoch:02d}-{cal_loss:0.4f}.h5'))

callbacks = [tensorboard, csv_logger, checkpoint, emb_weight_saver]

hyperparams = {'batch_size': batch_size,
               'epochs': epochs,
               'steps_per_epoch': steps_per_epoch,
               'val_steps': val_steps,
               'alpha': alpha,
               'num_hard': num_hard,
               'optimiser': optimiser,
               'lr': lr,
               'emb_size': emb_size
               }

with open(os.path.join(logdir, "hyperparams.json"), "w") as json_file:
    json.dump(hyperparams, json_file)

# Compile the model
# model=create_embedding_model(emb_size)
model.compile(
    optimizer=optimiser_obj,
    loss=tfa.losses.TripletSemiHardLoss())





# Train the network
# history = model.fit(
#     train_dataset,
#     epochs=5)

# history = model.fit(
#         # x=train_x,y=train_y,
#         train_dataset,
#         # steps_per_epoch=steps_per_epoch,
#         epochs=epochs,
#         verbose=1,
#         callbacks=callbacks,
#         # workers=0,
#         # validation_data=(val_x,val_y),
#         validation_data = test_dataset,
#         validation_steps=val_steps,shuffle = 1
#         )

def json_to_dict(json_src):
    with open(json_src, 'r') as j:
        return json.loads(j.read())


if (load_model):
    ## Load in best trained SNN and emb model

    # The best performing model weights has the higher epoch number due to only saving the best weights
    highest_epoch = 0
    dir_list = os.listdir(logdir)

    for file in dir_list:
        if file.endswith(".h5"):
            epoch_num = int(file.split("-")[1].split(".h5")[0])
            if epoch_num > highest_epoch:
                highest_epoch = epoch_num

    # Find the embedding and SNN weights src for the highest_epoch (best) model
    for file in dir_list:
        # Zfill ensure a leading 0 on number < 10
        if ("-" + str(highest_epoch).zfill(2)) in file:
            if file.startswith("emb"):
                embedding_weights_src = os.path.join(logdir, file)
            elif file.startswith("snn"):
                snn_weights_src = os.path.join(logdir, file)

    model.load_weights(embedding_weights_src)
    # hyperparams = os.path.join(logdir, "hyperparams.json")
    # snn_config = os.path.join(logdir, "siamese_config.json")
    # emb_config = os.path.join(logdir, "embedding_config.json")
    #
    # snn_config = json_to_dict(snn_config)
    # emb_config = json_to_dict(emb_config)
    #
    # # json.dumps to make the dict a string, as required by model_from_json
    # loaded_snn_model = model_from_json(json.dumps(snn_config))
    # loaded_snn_model.load_weights(snn_weights_src)
    #
    # loaded_emb_model = model_from_json(json.dumps(emb_config))
    # loaded_emb_model.load_weights(embedding_weights_src)

    # Store visualisations of the embeddings using PCA for display next to "after training" for comparisons


else:

    if (os.path.exists(logdir)):
            print("log path exist")
            shutil.rmtree(logdir)
            # os.rmdir(input_folder)
            os.mkdir(logdir)

    history = model.fit(
            # x=train_x,y=train_y,
            train_dataset,
            # steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            verbose=1,
            callbacks=callbacks,
            # workers=0,
            # validation_data=(val_x,val_y),
            validation_data = test_dataset,
            validation_steps=val_steps,shuffle = 1
            )

    highest_epoch = 0
    dir_list = os.listdir(logdir)

    for file in dir_list:
        if file.endswith(".h5"):
            epoch_num = int(file.split("-")[1].split(".h5")[0])
            if epoch_num > highest_epoch:
                highest_epoch = epoch_num

    # Find the embedding and SNN weights src for the highest_epoch (best) model
    for file in dir_list:
        # Zfill ensure a leading 0 on number < 10
        if ("-" + str(highest_epoch).zfill(2)) in file:
            if file.startswith("emb"):
                embedding_weights_src = os.path.join(logdir, file)
            elif file.startswith("snn"):
                snn_weights_src = os.path.join(logdir, file)

    print("best model found is", embedding_weights_src)
    model.load_weights(embedding_weights_src)




# Evaluate the network
val_results = model.predict(test_dataset)
train_results = model.predict(train_dataset)

# print(results.shape)



import io,os
# Save test embeddings for visualization in projector
np.savetxt(os.path.join(this_path,"val_vecs.tsv"), val_results, delimiter="\t")
np.savetxt(os.path.join(this_path,"train_vecs.tsv"), train_results, delimiter="\t")

val_meta = os.path.join(this_path,"val_meta.tsv")
out_m = io.open(val_meta , 'w', encoding='utf-8')
for img, labels in tfds.as_numpy(test_dataset):
    [out_m.write(str(x) + "\n") for x in labels]
out_m.close()


train_meta = os.path.join(this_path,"train_meta.tsv")
out_m = io.open(train_meta , 'w', encoding='utf-8')
for img, labels in tfds.as_numpy(train_dataset):
    [out_m.write(str(x) + "\n") for x in labels]
out_m.close()



try:
  from google.colab import files
  files.download('val_vecs.tsv')
  files.download('val_meta.tsv')

  files.download('train_vecs.tsv')
  files.download('train_meta.tsv')

except:
  pass



# import pandas as pd
# from sklearn.decomposition import PCA
#
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
#
# pca = PCA(n_components=2)
#
# df_val_data=pd.read_csv(os.path.join(this_path,"val_vecs.tsv"),sep='\t',header=None)
# df_val_class=pd.read_csv(os.path.join(this_path,"val_meta.tsv"),sep='\t',header=None)
#
# # df_val_data["class"]=df_val_class[0]
# df_val_data=df_val_data.reset_index()
# df_val_class["class"]=df_val_class[0]
#
# # pca = PCA()
# pipe = Pipeline([('scaler', StandardScaler()), ('pca', pca)])
# principalComponents = pipe.fit_transform(df_val_data)
#
# # principalComponents = pca.fit_transform(df_val_data)
#
# principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
#
# finalDf2 = pd.concat([principalDf, df_val_class["class"]], axis = 1)
# finalDf2.head()
#
# plt.figure(figsize=(7,7))
# plt.scatter(finalDf2['principal component 1'],finalDf2['principal component 2'],c=finalDf2['class'],cmap='prism', s =5)
# plt.xlabel('pc1')
# plt.ylabel('pc2')
#
#
# pca3 = PCA(n_components=3)
# pipe = Pipeline([('scaler', StandardScaler()), ('pca', pca3)])
# principalComponents = pipe.fit_transform(df_val_data)
# # principalComponents = pca3.fit_transform(df_val_data)
#
# principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2', 'principal component 3'])
#
# finalDf3 = pd.concat([principalDf, df_val_class["class"]], axis = 1)
# finalDf3.head()
#
# from mpl_toolkits.mplot3d import Axes3D
#
# fig = plt.figure(figsize=(9,9))
# axes = Axes3D(fig)
# axes.set_title('PCA Representation', size=14)
# axes.set_xlabel('PC1')
# axes.set_ylabel('PC2')
# axes.set_zlabel('PC3')
#
# axes.scatter(finalDf3['principal component 1'],finalDf3['principal component 2'],finalDf3['principal component 3'],c=finalDf3['class'], cmap = 'prism', s=10)
#
#
#
# flag=1
