
#define the path to the input parasite image
import sys

# input_parasite_photo_path=r"/Users/mahdi/Documents/mahdi/parasite_detection/Aug25_update/Fasiola_hepatica_egg_zAAwN2Xrwpg_137-7.jpeg"
# input_parasite_photo_path=r"/Users/mahdi/Documents/mahdi/parasite_detection/Aug25_update/python_self_install/Ascaris (14).jpg"
# input_parasite_photo_path=r"/Users/mahdi/Documents/mahdi/parasite_detection/Sep14/training_images_best/Schistosoma_Haematobium/S.haematobium_egg2_1.jpg"
# input_parasite_photo_folder=r"/Users/mahdi/Documents/mahdi/parasite_detection/Sep14/training_images_best/"

from skimage import data, color, io
from skimage.transform import rescale, resize, downscale_local_mean
import os, glob
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from skimage import data, filters
from skimage.transform import rotate
import cv2, shutil

import matplotlib.pyplot as plt
import seaborn as sns
import umap
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
import pandas as pd

# from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import os

# %matplotlib inline

this_script_path = os.path.dirname(os.path.realpath(__file__))


#define global parameter
tmp_folder=os.path.join(this_script_path,r"tmp")
img_size = 128
img_tag = {'Ascaris': 0, 'Dicrocoelium': 1, 'Hymenolepis_nana': 2, 'Trichuris_trichiura': 3,
           'Schistosoma_Haematobium': 4, 'Dipylidium_Caninum': 5, 'Hookworm': 6, 'Enterobius_Vermicularis': 7,
           'Schistosoma_Mansoni': 8, 'Taenia': 9, 'Fasciola': 10}
img_tag0 = img_tag

rev_parasite_tag = {v: k for k, v in img_tag.items()}

#model path
model_path=os.path.join(this_script_path,r"emb_model-19308-0.0102.h5")  ##r"emb_model-9507-0.0083.h5")

val_meta=os.path.join(this_script_path,r"val_meta.tsv")
val_vec=os.path.join(this_script_path,r"val_vecs.tsv")

df0=pd.read_csv(val_meta,header=None,delimiter="\t")
df1=pd.read_csv(val_vec,header=None,delimiter="\t")
val_labels=df0.values
val_data=df1.values

#define the path to the input parasite image
# this_path = os.path.dirname(input_parasite_photo_path)

# accuracy_test_result=os.path.join(this_script_path,"verification_accuracy_sep14Script.csv")
# df_accuracy_test_result=pd.DataFrame()
# #generate a temprory folder and copy the parasite images and augmented image into this folder
# tmp_cnt=0
# while True:
#     this_tmp_folder=os.path.join(tmp_folder,"tmp%d"%tmp_cnt)
#     if (os.path.exists(this_tmp_folder)):
#         tmp_cnt+=1
#         continue
#     else:
#         break
# os.mkdir(this_tmp_folder)

input_shape = (img_size, img_size, 1)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=32, kernel_size=5, activation='relu', input_shape=input_shape),
    tf.keras.layers.MaxPooling2D(pool_size=5),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=3),
    tf.keras.layers.Dropout(0.3),
    # tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu'),
    # tf.keras.layers.MaxPooling2D(pool_size=3),
    # tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation=None),  # No activation on final dense layer
    tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))  # L2 normalize embeddings

])

model.load_weights(model_path)
pca_comp=10

def detect_parasite(input_parasite_photo_path):
#-------------------------------------------------------------------#
#generate augmented images
# for input_parasite_photo_path in glob.glob(os.path.join(input_parasite_photo_folder,'*.*')):
# for pca_comp in [8, 9, 10, 11, 12]:
#     for parasite_name in img_tag0.keys():
#         for input_parasite_photo_path in glob.glob(os.path.join(input_parasite_photo_folder, parasite_name, '*.*')):
    print ("processing ",input_parasite_photo_path)
    # input_parasite_photo_path
    img_file_name = os.path.basename(input_parasite_photo_path)
    img_path = input_parasite_photo_path
    # raw_img=data.load(img_path)
    try:
        raw_img = io.imread(img_path, as_gray=True)
    except:
        print ("can not open the image file", input_parasite_photo_path)
        return "NA"
        # sys.exit("Error opening the image file")

    # generate a temprory folder and copy the parasite images and augmented image into this folder
    tmp_cnt = 0
    while True:
        this_tmp_folder = os.path.join(tmp_folder, "tmp%d" % tmp_cnt)
        if (os.path.exists(this_tmp_folder)):
            tmp_cnt += 1
            continue
        else:
            break
    os.mkdir(this_tmp_folder)

    img0 = filters.sobel(raw_img)
    img0 = img0 / np.max(img0)

    m, n = img0.shape
    img1=img0

    if (np.abs(m - n) > 3):
        # img0 = filters.sobel(raw_img)
        # img0 = img0 / np.max(img0)
        # m, n = img0.shape
        # h = plt.hist(img0.reshape((m * n, 1)), bins=np.arange(0, 1.0, 0.01))
        h = np.histogram(img0.reshape((m * n, 1)), bins=np.arange(0, 1.0, 0.01))

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

    suffix = img_file_name.split('.')[-1]
    # print(suffix)
    ln_suffix = len(suffix) + 1

    img_path_augmented = os.path.join(this_tmp_folder, img_file_name)

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


    #----------------------------------------------
    #read all the images for testing
    x_data=[]
    y_data=[]

    for img_folder in [this_tmp_folder]:
      all_images_name = glob.glob(os.path.join(this_tmp_folder,'*.*'))
      count=0
      for img in all_images_name:
          # img_path=os.path.join(input_folder,img_folder,img)
          # raw_img=data.load(img_path)
          try:
              raw_img=io.imread(img,as_gray = True)
          except:
              continue
          # image = color.rgb2gray(raw_img)
          image_resized = resize(raw_img, (img_size, img_size),anti_aliasing=True)
          # image_resized=image_resized/np.max(image_resized)

          # edges = filters.sobel(image_resized)
          # edges=edges/np.max(edges)
          edges=image_resized

          x_data.append(edges)
          y_data.append(-100)
          # y_data.append(img_tag[img_folder])
    x_data=np.stack(x_data,axis=0)
    y_data=np.stack(y_data,axis=0)
    y_data=y_data.reshape((len(y_data),1))
    len_data=len(x_data)
    #--------------------------------------------------
    #load the Siamese network model


    # model = tf.keras.Sequential([
    #     tf.keras.layers.Conv2D(filters=64, kernel_size=5, activation='relu', input_shape=input_shape),
    #     tf.keras.layers.MaxPooling2D(pool_size=5),
    #     tf.keras.layers.Dropout(0.3),
    #     tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu'),
    #     tf.keras.layers.MaxPooling2D(pool_size=3),
    #     tf.keras.layers.Dropout(0.3),
    #     # tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu'),
    #     # tf.keras.layers.MaxPooling2D(pool_size=3),
    #     # tf.keras.layers.Dropout(0.3),
    #     tf.keras.layers.Flatten(),
    #     tf.keras.layers.Dense(2048, activation=None),  # No activation on final dense layer
    #     tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))  # L2 normalize embeddings
    #
    # ])

    # model = tf.keras.Sequential([
    #     tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
    #     tf.keras.layers.MaxPooling2D(pool_size=3),
    #     tf.keras.layers.Dropout(0.3),
    #     tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu'),
    #     tf.keras.layers.MaxPooling2D(pool_size=3),
    #     tf.keras.layers.Dropout(0.3),
    #     tf.keras.layers.Conv2D(filters=256, kernel_size=2, activation='relu'),
    #     tf.keras.layers.MaxPooling2D(pool_size=2),
    #     tf.keras.layers.Dropout(0.3),
    #     tf.keras.layers.Flatten(),
    #     tf.keras.layers.Dense(512, activation=None),  # No activation on final dense layer
    #     tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))  # L2 normalize embeddings
    #
    # ])



    #---------------------------------------------------------------
    #validate input prasite image
    val_results = model.predict(x_data)

    total_data=np.vstack([val_data,val_results])
    total_label=np.vstack([val_labels,y_data])
    # trans = umap.UMAP( n_neighbors=5,
    #     min_dist=0.0,
    #     n_components=2,
    #     random_state=1,).fit(total_data)
    #
    # # plt.scatter(trans.embedding_[:, 0], trans.embedding_[:, 1], s= 5, c=y_train, cmap='Spectral')
    # plt.scatter(trans.embedding_[:, 0], trans.embedding_[:, 1], s= 30, c=total_label, cmap='Spectral')
    # import pandas as pd


    from sklearn.decomposition import PCA

    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    #
    # df=pd.DataFrame(total_data)
    # pca = PCA(n_components=2)
    # pca3 = PCA(n_components=3)

    # scalar = StandardScaler()
    # scaled_data = pd.DataFrame(scalar.fit_transform(df))
    # pca.fit(scaled_data)
    # data_pca = pca.transform(scaled_data)
    # data_pca = pd.DataFrame(data_pca,columns=['PC1','PC2'])
    # data_pca.head()


    # pipe = Pipeline([('scaler', StandardScaler()), ('pca', pca)])
    # principalComponents2 = pipe.fit_transform(total_data)

    pca = PCA(n_components=pca_comp)
    pipe = Pipeline([('scaler', StandardScaler()), ('pca', pca)])
    principalComponents = pipe.fit_transform(total_data)

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
    # Standardizing the features
    # x = StandardScaler().fit_transform(total_data)
    # principalComponents = pca.fit_transform(x)
    # principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])

    # finalDf = pd.concat([principalDf, df[['class']]], axis = 1)
    # finalDf.head()

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

    # plt.title('Embedding of the training set by UMAP', fontsize=24);

    X=np.zeros((len(principalComponents)-len_data,2))
    X=principalComponents[:-len_data,:]
    # X[:,0]=trans.embedding_[:-len_data, 0]
    # X[:,1]=trans.embedding_[:-len_data, 1]
    #
    x=np.zeros((len_data,2))
    x=principalComponents[-len_data:,:]

    # x[:,0]=trans.embedding_[-len_data:, 0]
    # x[:,1]=trans.embedding_[-len_data:, 1]

    kmeans = KMeans(n_clusters=11)
    kmeans.fit(X)
    # print(kmeans.cluster_centers_)

    # kmeans.predict(X[100:102,:])
    x_res=kmeans.predict(x)
    cnts=np.bincount(x_res)
    p_idx=np.argmax(cnts)


    y_p=kmeans.predict(X)
    df=pd.DataFrame()
    df["X0"]=y_p
    df["X1"]=val_labels
    # df=df.sort_values(by=['X0']).value_counts()
    df=df.value_counts()

    # match_max=df.groupby(["X0"]).value_counts().sort_values(ascending=False)

    match_label={}
    match_keys=[]
    i=0
    # for x0 in np.unique(y_p):
    #     i=0
    #     # print ("x0 is ",x0)
    #     # df_x=df[df["X0"]==x0]
    #     # print (df_x.value_counts().sort_values(ascending=False))
    #     # print(df_x[0:(1)])
    #     # print(df_x[0:(2)])
    #     # print(df_x[1:(2)])
    #
    #     res=str(df[i:(i+1)]).split('\n')[1].split()
    #     # print (res)
    #     idx1=int(res[1])
    #     idx2=int(res[2])
    #     if (idx1 not in match_keys):
    #         match_keys.append(idx1)
    #         match_label[idx1]=idx2
    while True:
        if (len(match_keys)>=len(np.unique(y_p))):
            break
        res=str(df[i:(i+1)]).split('\n')[1].split()

        idx1=int(res[0])
        idx2=int(res[1])
        if (idx1 not in match_keys):
            match_keys.append(idx1)
            match_label[idx1]=idx2
        i=i+1

    detected_parasites=[]
    for p_x in x_res:
        # print("detected parasite is: ", rev_parasite_tag[match_label[p_x]])
        detected_parasites.append(rev_parasite_tag[match_label[p_x]])

    detected_parasite=set(detected_parasites)
    # print("len of parasites, ",len(detected_parasite))
    # print ("parasite, ",detected_parasite)

    # data_dic = {"PCA_n": pca_comp, "input_file": input_parasite_photo_path, "parasite": parasite_name,
    #             "detected": ('-'.join(detected_parasite))}
    # tmp_df = pd.DataFrame(data_dic, index=[0])
    # if len(df_accuracy_test_result) == 0:
    #     df_accuracy_test_result = tmp_df
    # else:
    #     df_accuracy_test_result = pd.concat([df_accuracy_test_result, tmp_df])
    msg=""
    if (len(detected_parasite)==0):
        msg = "parasite is not detetable!"
    elif (len(detected_parasite)==1):
        msg="detected parasite is: " +str(detected_parasite)
    elif (len(detected_parasite)==2):
        msg="Prasite is one of these two parasites "+', '.join(list(detected_parasite))
    elif (len(detected_parasite)==3):
        for parasite in detected_parasite:
            if (detected_parasites.count(parasite) == 2):
                break
            # print ("Prasite with 50% chance is %s and  with 25% chance one of these two parasites %s"%(parasite,detected_parasite.discard(parasite)))
        # print (parasite)
        detected_parasite.remove(parasite)
        print ("left dp, ",detected_parasite)

        # print(','.join(list(detected_parasite.discard(parasite))))
        msg="Prasite with 50% chance is {} and  with 25% chance one of these two parasites ".format(parasite)+', '.join(list(detected_parasite))
    else:
        msg="Prasite with 25% chance is one of these parasites "+", ".join(list(detected_parasite))

    return msg



    # predicted_idx=match_label[p_idx]
    # parasite=rev_parasite_tag[predicted_idx]
    # print ("detected parasite is: ",parasite)

# detect_parasite(os.path.join(this_script_path,r"test_images/S_mansoni_egg26_1.jpg"))
# flag=1
# df_accuracy_test_result.to_csv(accuracy_test_result)
