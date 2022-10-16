# data process
import numpy as np
# tensorflow utils
import tensorflow as tf
# image processing
from PIL import Image
import cv2


class TanaminModels:
  
  # directory from google drive (SHOULD BE EDITED)
  LEAF_MODEL_DIR = "/content/drive/Shareddrives/Tanamin Team/Machine Learning/models/leaf_models/saved_model/BestMC_DenseNet121"
  PLANT_MODEL_DIR = "/content/drive/Shareddrives/Tanamin Team/Machine Learning/models/plant_models/new_model/saved_model/best_modelDense_Klasifikasi_Daun"
  POTATO_MODEL_DIR = "/content/drive/Shareddrives/Tanamin Team/Machine Learning/models/potato_models/saved_model/BestMC_DenseNet121"
  CORN_MODEL_DIR = "/content/drive/Shareddrives/Tanamin Team/Machine Learning/models/corn_models/saved_model/BestMC_DenseNet121"
  RICE_MODEL_DIR = "/content/drive/Shareddrives/Tanamin Team/Machine Learning/models/rice_models/saved_model/BestMC_DenseNet121"
  CASSAVA_MODEL_DIR = "/content/drive/Shareddrives/Tanamin Team/Machine Learning/models/cassv_models/saved_model/BestMC_DenseNetModel"
  CHILI_MODEL_DIR = "/content/drive/Shareddrives/Tanamin Team/Machine Learning/models/chili_models/saved_model/BestMC_bestmodelTFLEARNING"
  TOMATO_MODEL_DIR = "/content/drive/Shareddrives/Tanamin Team/Machine Learning/models/tomat/saved_model/best_modelDense"

  LEAF_CLASSES = ['leaf', 'not_leaf']
  PLANT_CLASSES = ['cassava', 'chili', 'corn', 'not_available', 'potato', 'rice', 'tomato']
  POTATO_CLASSES = ['early_blight', 'late_blight', 'healthy']
  CORN_CLASSES = ['common_rust', 'gray_leaf_spot', 'healthy', 'northern_leaf_blight']
  RICE_CLASSES = ['blight', 'brown_spot', 'healthy', 'tungro']
  CASSAVA_CLASSES = ['bacterial_blight', 'green_mottle', 'mosaic', 'healthy']
  CHILI_CLASSES = ['leaf_curl', 'healthy', 'yellowish', 'kekurangan_magnesium', 'cercospora']
  TOMATO_CLASSES = ['early_blight', 'kurang_magnesium', 'lalat_pengarat', 'leaf_mold', 'sehat', 'tomato_yellow_leaf_curl', 'septoria_leaf_spot', 'target_spot', 'bacterial_spot', 'embun_tepung', 'late_blight', 'mosaic']


  def __init__(self):
    self.leafModel = tf.keras.models.load_model(TanaminModels.LEAF_MODEL_DIR)
    self.plantModel = tf.keras.models.load_model(TanaminModels.PLANT_MODEL_DIR)
    self.potatoModel = tf.keras.models.load_model(TanaminModels.POTATO_MODEL_DIR)
    self.cornModel = tf.keras.models.load_model(TanaminModels.CORN_MODEL_DIR)
    self.riceModel = tf.keras.models.load_model(TanaminModels.RICE_MODEL_DIR)
    self.cassavaModel = tf.keras.models.load_model(TanaminModels.CASSAVA_MODEL_DIR) # Warning: no compile
    self.chiliModel = tf.keras.models.load_model(TanaminModels.CHILI_MODEL_DIR)
    self.tomatoModel = tf.keras.models.load_model(TanaminModels.TOMATO_MODEL_DIR)


  def leaf_predict(self, img):
    result = self.leafModel.predict(img)
    result = np.argmax(result)
    print("leaf_predict_result:", result)
    return result


  def plant_predict(self, img):
    result = self.plantModel.predict(img)
    result = np.argmax(result)
    print("plant_predict_result:", result)
    return result


  def disease_predict(self, img, plant):
    if plant == "cassava":
      result = self.cassavaModel.predict(img)
      result_class = TanaminModels.CASSAVA_CLASSES[np.argmax(result)]

    elif plant == "chili":
      result = self.chiliModel.predict(img)
      result_class = TanaminModels.CHILI_CLASSES[np.argmax(result)]

    elif plant == "corn":
      result = self.cornModel.predict(img)
      result_class = TanaminModels.CORN_CLASSES[np.argmax(result)]

    elif plant == "potato":
      result = self.potatoModel.predict(img)
      result_class = TanaminModels.POTATO_CLASSES[np.argmax(result)]

    elif plant == "rice":
      result = self.riceModel.predict(img)
      result_class = TanaminModels.RICE_CLASSES[np.argmax(result)]

    elif plant == "tomato":
      result = self.tomatoModel.predict(img)
      result_class = TanaminModels.TOMATO_CLASSES[np.argmax(result)]

    accuracy = max(result[0]) * 100
    print("disease_predict_result:", result_class, accuracy)
    
    return result_class, accuracy


class TanaminPrediction:
  
  def __init__(self, models):
    self.models = models


  def __crop_image(self, img):
    width, height, _ = img.shape
    if width == height:
        return img
    img = np.array(img)
    offset  = int(abs(height-width)/2)
    if width>height:
        img = img[:,offset:(width-offset),:]
    else:
        img = img[offset:(height-offset),:,:]
    return img


  def __preprocessing(self, img):
    img = self.__crop_image(img)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = img / 255
    return img


  def start_predict(self, img):
    plant = None
    accuracy = None
    
    img = self.__preprocessing(img)
    result = self.models.leaf_predict(img)

    if self.models.LEAF_CLASSES[result] == "not_leaf":
      result = "Image is not leaf" 
    else:
      result = self.models.plant_predict(img)
      if self.models.PLANT_CLASSES[result] == "not_available":
        result = "This plant is not yet available" 
      else:
        plant = self.models.PLANT_CLASSES[result]
        result, accuracy = self.models.disease_predict(img, plant)
    
    final_result = {
        'result': result,
        'accuracy': accuracy,
        'plant': plant
    }

    return final_result


## Testing Code Below (Remove Comments)
# tanaminModels = TanaminModels()
# tanaminPrediction = TanaminPrediction(tanaminModels)

# image_file = '/content/potato-early-blight.jpeg'
# image = np.asarray(Image.open(image_file))

# tanaminPrediction.start_predict(image)