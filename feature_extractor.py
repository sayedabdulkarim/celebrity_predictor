#### FOR getting filenames.pkl ####

# import os
# import pickle
#
# actors = os.listdir('data')
# # print(actors)
#
# fileNames = []
#
# for actor in actors:
#     for file in os.listdir(os.path.join('data', actor)):
#         fileNames.append(os.path.join('data', actor, file))
#
# # print(fileNames)
# # print(len(fileNames), ' length')
#
# # we will dump it to binary obj
# pickle.dump(fileNames, open('filenames.pkl', 'wb'))


#######################################
#######################################

from tensorflow.keras.preprocessing import image
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import numpy as np
import pickle
from tqdm import tqdm

# Load filenames from the pickle file
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Initialize the VGGFace model using ResNet50 backbone
model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')


# Print model summary to verify the architecture
# model.summary()
def feature_extractor(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)

    result = model.predict(preprocessed_img).flatten()

    return result


features = []

for file in tqdm(filenames):
    features.append(feature_extractor(file, model))
    # result = feature_extractor(file, model)

    # print(result.shape)
    # break

    pickle.dump(features, open('embedding.pkl', 'wb'))