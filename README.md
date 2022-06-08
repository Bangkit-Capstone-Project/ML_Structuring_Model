<h1 align="center">
  <img align="center" src="images\Logo Tanamin 3-01.png"  width="500"></img>
<br>
TANAMIN MACHINE LEARNING PROJECT README
</h1>
<div align="center">

[![ML Contributors](https://img.shields.io/github/contributors/Bangkit-Capstone-Project/ML_Structuring_Model?color=red)](#mlcontributors)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/Bangkit-Capstone-Project/ML_Structuring_Model)
![GitHub last commit](https://img.shields.io/github/last-commit/Bangkit-Capstone-Project/ML_Structuring_Model)
[![GitHub forks](https://img.shields.io/github/forks/Bangkit-Capstone-Project/ML_Structuring_Model)](https://github.com/Bangkit-Capstone-Project/ML_Structuring_Model)
</div>

# Team Profile

### Team ID : C22-PS177

### This is our Members

* (ML) M2297G2522 - Rama Tri Agung - UPN Veteran Yogyakarta
* (ML) M2007F0755 - Brylian Fandhi Safsalta - Universitas Dian Nuswantoro
* (ML) M2007F0765 - Achmad Naila Muna Ramadhani - Universitas Dian Nuswantoro
* (MD) A2306F2629 - Abraham Pardomuan Naiborhu - President University
* (MD) A2007F0748 - Nanang Febrianto - Universitas Dian Nuswantoro
* (CC) C7297F2544 - Shazi Awaludin  - UPN Veteran Yogyakarta

### Roles/personnel

* Project Plan (A2007F0748 - Nanang Febrianto, A2306F2629 - Abraham Pardomuan Naiborhu, C7297F2544 - Shazi Awaludin, M2007F0755 - Brylian Fandhi Safsalta, M2297G2522 - Rama Tri Agung, M2007F0765 - Achmad Naila Muna Ramadhani )
* UI/UX (A2306F2629 - Abraham Pardomuan Naiborhu, A2007F0748 - Nanang Febrianto)
* Build Machine Learning Model (M2007F0765 - Achmad Naila Muna Ramadhani, M2007F0755 - Brylian Fandhi Safsalta, M2297G2522 - Rama Tri Agung)
* Android Apps Development (A2306F2629 - Abraham Pardomuan Naiborhu, A2007F0748 - Nanang Febrianto)
* Deployment Application (C7297F2544 - Shazi Awaludin)




# Tanamin Machine Learing Project
This ML project is our final project for Google Bangkit Academy 2022.

**Android:**
[Tanamin Mobile Apps Developments](https://github.com/Bangkit-Capstone-Project/MobileDev_AppsProject)

**Cloud:**
[Tanamin Cloud Service](https://github.com/DocBot-Bangkit-2021/DocBot-Cloud)

**Project Background:**

Rice and cassava are staple foods for all Indonesian people. Planting these plants takes a long time to be harvested, so special attention needs to be paid to these plants so as not to contract the disease and the occurrence of crop failure or quality degradation. To prevent this requires the ability of humans to select and treat plant diseases. However, manual checking takes a lot of time and effort and requires a lot of human resources.

Based on these problems, can be prevented by providing technological solutions, namely building machine learning applications that can detect diseases of rice and cassava plants. The solution aims to assist farmers in increasing the efficiency of checking for the disease to prevent crop failures and improve society's welfare.


**Machine Learning:** 

Machine Learning path build four kinds of models that include [Cassava Diseases](https://github.com/Bangkit-Capstone-Project/ML_Structuring_Model/tree/main/Cassava_disease), [Rice Diseases](https://github.com/Bangkit-Capstone-Project/ML_Structuring_Model/tree/main/Rice_disease), [Tomato Leaf disease detection](https://github.com/Bangkit-Capstone-Project/ML_Structuring_Model/tree/main/Tomato_disease) and [Vegetables detection](https://github.com/Bangkit-Capstone-Project/ML_Structuring_Model/tree/main/Vagetable). Build process using *baseline experiment, early stopping, checkpoint*. Pre-trained model or transfer learning by *resnet152v2, densenet121, inceptionv3, mobilenetv2*. The model was saved and deployed with *saved_model*.

**Project Case :**

- CassavaLeaf Diseases
- Rice Plan Diseases
- Vegetable Classification
- Tomato Leaf Diseases

**Our Dataset Link:**

* Cassava Diseases 
  * [Cassava Leaf Diseases Datasets](https://www.kaggle.com/c/cassava-disease)
  
* Tomato Leaf  Diseases
  * [Tomato Leaf Disease](https://www.kaggle.com/datasets/noulam/tomato)

* Rice Plant Diseases
  * [syahyanriyaz Dataset](https://www.kaggle.com/datasets/shayanriyaz/riceleafs)
  * [Chandrug Dataset](https://www.kaggle.com/datasets/chandrug/riceleafdisease)
  * [Tedisetiady Dataset](https://www.kaggle.com/datasets/tedisetiady/leaf-rice-disease-indonesia)
  * [vbookshelf Dataset](https://www.kaggle.com/datasets/vbookshelf/rice-leaf-diseases)

* Vegetable Classification 
  * [Vegetables Dataset](https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset)

Preview of the image and data used are shown in the picture below.

<img align="center" src="images\cassav.png"></img>

<p align="center">Cassava Leaf Diseases preview dataset </p>

<img align="center" src="images\rice_image.jpg"></img>

<p align="center">Rice Plant Disease preview dataset</p>

<img align="center" src="images\tomato_image.jpg"></img>

<p align="center">Tomato Leaf Dataset</p>

<img align="center" src="images\vegetable_image.jpg"></img>
<p align="center">Vegetables Dataset</p>


## Notebook for each case

### Cassava Disease
- [Cassava_disease.ipynb](https://github.com/Bangkit-Capstone-Project/ML_Structuring_Model/blob/7f98fc705b111da08900434fbf9e7758c403b9e6/Cassava_disease/Copy_of_model_baseline_v2_(1).ipynb)


### Rice Plant Disease

- [Rice_disease.ipynb](https://github.com/Bangkit-Capstone-Project/ML_Structuring_Model/blob/7f98fc705b111da08900434fbf9e7758c403b9e6/Rice_disease/rice_disease_prediction.ipynb)

### Tomatoes 

- [Tomatoes_Leaf_Disease.ipynb](https://github.com/Bangkit-Capstone-Project/ML_Structuring_Model/blob/7f98fc705b111da08900434fbf9e7758c403b9e6/Tomato_disease/Tomato.ipynb)

###  Vegetables

- [Vegetables.ipynb](https://github.com/Bangkit-Capstone-Project/ML_Structuring_Model/blob/7f98fc705b111da08900434fbf9e7758c403b9e6/Vagetable/vagetable.ipynb)


## Prerequisites
1. [Jupyter Notebook](https://test-jupyter.readthedocs.io/en/latest/install.html) or [Google Colab](https://colab.research.google.com/)
2. Kaggle API Token → [Generate](https://github.com/Kaggle/kaggle-api#api-credentials)
3. [Python](https://www.python.org/downloads/) version 3.6 or above
4. Latest version of Tensorflow 2.5 (or you can update again by rerunning .ipynb and updating models)

## How to use
1. [Create `kaggle.json` from Kaggle](https://github.com/Kaggle/kaggle-api#api-credentials)
2. Go to your Kaggle profile then download your Kaggle API.
    - My Account  →  Look for API section  →  Create New API Token
3. Open `.ipynb` with Google Colaboratory using `open in colab`
4. "Save a copy in Drive" to run and edit with your account. Click `File` > `Save a copy in Drive` in your Google Colaboratory.
5. Upload your `kaggle.json` if asked to upload it.
6. Will download the dataset on kaggle
7. If using GoogleDrive for the dataset, `GoogleAuth` click on the given link and sign in with your Google Account.
8. Done
