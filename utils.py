from keras.preprocessing import image
import numpy as np
from extract_bottleneck_features import *
import cv2
from keras.models import load_model
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input

bottleneck_features = np.load('bottleneck_features/DogVGG19Data.npz')
train_VGG19 = bottleneck_features['train']
valid_VGG19 = bottleneck_features['valid']
test_VGG19 = bottleneck_features['test']

# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')

def ResNet50_predict_labels(img_path):
    """
    Predict Label from ResNet50 model.
    :param img_path:
    :return: Most likely label value
    """
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))


dog_names = ['project/dogimages/dogImages/train/001.Affenpinscher',
 'project/dogimages/dogImages/train/002.Afghan_hound',
 'project/dogimages/dogImages/train/003.Airedale_terrier',
 'project/dogimages/dogImages/train/004.Akita',
 'project/dogimages/dogImages/train/005.Alaskan_malamute',
 'project/dogimages/dogImages/train/006.American_eskimo_dog',
 'project/dogimages/dogImages/train/007.American_foxhound',
 'project/dogimages/dogImages/train/008.American_staffordshire_terrier',
 'project/dogimages/dogImages/train/009.American_water_spaniel',
 'project/dogimages/dogImages/train/010.Anatolian_shepherd_dog',
 'project/dogimages/dogImages/train/011.Australian_cattle_dog',
 'project/dogimages/dogImages/train/012.Australian_shepherd',
 'project/dogimages/dogImages/train/013.Australian_terrier',
 'project/dogimages/dogImages/train/014.Basenji',
 'project/dogimages/dogImages/train/015.Basset_hound',
 'project/dogimages/dogImages/train/019.Beagle',
 'project/dogimages/dogImages/train/017.Bearded_collie',
 'project/dogimages/dogImages/train/018.Beauceron',
 'project/dogimages/dogImages/train/019.Bedlington_terrier',
 'project/dogimages/dogImages/train/020.Belgian_malinois',
 'project/dogimages/dogImages/train/021.Belgian_sheepdog',
 'project/dogimages/dogImages/train/022.Belgian_tervuren',
 'project/dogimages/dogImages/train/023.Bernese_mountain_dog',
 'project/dogimages/dogImages/train/024.Bichon_frise',
 'project/dogimages/dogImages/train/025.Black_and_tan_coonhound',
 'project/dogimages/dogImages/train/026.Black_russian_terrier',
 'project/dogimages/dogImages/train/027.Bloodhound',
 'project/dogimages/dogImages/train/028.Bluetick_coonhound',
 'project/dogimages/dogImages/train/029.Border_collie',
 'project/dogimages/dogImages/train/030.Border_terrier',
 'project/dogimages/dogImages/train/031.Borzoi',
 'project/dogimages/dogImages/train/032.Boston_terrier',
 'project/dogimages/dogImages/train/033.Bouvier_des_flandres',
 'project/dogimages/dogImages/train/034.Boxer',
 'project/dogimages/dogImages/train/035.Boykin_spaniel',
 'project/dogimages/dogImages/train/036.Briard',
 'project/dogimages/dogImages/train/037.Brittany',
 'project/dogimages/dogImages/train/038.Brussels_griffon',
 'project/dogimages/dogImages/train/039.Bull_terrier',
 'project/dogimages/dogImages/train/040.Bulldog',
 'project/dogimages/dogImages/train/041.Bullmastiff',
 'project/dogimages/dogImages/train/042.Cairn_terrier',
 'project/dogimages/dogImages/train/043.Canaan_dog',
 'project/dogimages/dogImages/train/044.Cane_corso',
 'project/dogimages/dogImages/train/045.Cardigan_welsh_corgi',
 'project/dogimages/dogImages/train/046.Cavalier_king_charles_spaniel',
 'project/dogimages/dogImages/train/047.Chesapeake_bay_retriever',
 'project/dogimages/dogImages/train/048.Chihuahua',
 'project/dogimages/dogImages/train/049.Chinese_crested',
 'project/dogimages/dogImages/train/050.Chinese_shar-pei',
 'project/dogimages/dogImages/train/051.Chow_chow',
 'project/dogimages/dogImages/train/052.Clumber_spaniel',
 'project/dogimages/dogImages/train/053.Cocker_spaniel',
 'project/dogimages/dogImages/train/054.Collie',
 'project/dogimages/dogImages/train/055.Curly-coated_retriever',
 'project/dogimages/dogImages/train/056.Dachshund',
 'project/dogimages/dogImages/train/057.Dalmatian',
 'project/dogimages/dogImages/train/058.Dandie_dinmont_terrier',
 'project/dogimages/dogImages/train/059.Doberman_pinscher',
 'project/dogimages/dogImages/train/060.Dogue_de_bordeaux',
 'project/dogimages/dogImages/train/061.English_cocker_spaniel',
 'project/dogimages/dogImages/train/062.English_setter',
 'project/dogimages/dogImages/train/063.English_springer_spaniel',
 'project/dogimages/dogImages/train/064.English_toy_spaniel',
 'project/dogimages/dogImages/train/065.Entlebucher_mountain_dog',
 'project/dogimages/dogImages/train/066.Field_spaniel',
 'project/dogimages/dogImages/train/067.Finnish_spitz',
 'project/dogimages/dogImages/train/068.Flat-coated_retriever',
 'project/dogimages/dogImages/train/069.French_bulldog',
 'project/dogimages/dogImages/train/070.German_pinscher',
 'project/dogimages/dogImages/train/071.German_shepherd_dog',
 'project/dogimages/dogImages/train/072.German_shorthaired_pointer',
 'project/dogimages/dogImages/train/073.German_wirehaired_pointer',
 'project/dogimages/dogImages/train/074.Giant_schnauzer',
 'project/dogimages/dogImages/train/075.Glen_of_imaal_terrier',
 'project/dogimages/dogImages/train/076.Golden_retriever',
 'project/dogimages/dogImages/train/077.Gordon_setter',
 'project/dogimages/dogImages/train/078.Great_dane',
 'project/dogimages/dogImages/train/079.Great_pyrenees',
 'project/dogimages/dogImages/train/080.Greater_swiss_mountain_dog',
 'project/dogimages/dogImages/train/081.Greyhound',
 'project/dogimages/dogImages/train/082.Havanese',
 'project/dogimages/dogImages/train/083.Ibizan_hound',
 'project/dogimages/dogImages/train/084.Icelandic_sheepdog',
 'project/dogimages/dogImages/train/085.Irish_red_and_white_setter',
 'project/dogimages/dogImages/train/086.Irish_setter',
 'project/dogimages/dogImages/train/087.Irish_terrier',
 'project/dogimages/dogImages/train/088.Irish_water_spaniel',
 'project/dogimages/dogImages/train/089.Irish_wolfhound',
 'project/dogimages/dogImages/train/090.Italian_greyhound',
 'project/dogimages/dogImages/train/091.Japanese_chin',
 'project/dogimages/dogImages/train/092.Keeshond',
 'project/dogimages/dogImages/train/093.Kerry_blue_terrier',
 'project/dogimages/dogImages/train/094.Komondor',
 'project/dogimages/dogImages/train/095.Kuvasz',
 'project/dogimages/dogImages/train/096.Labrador_retriever',
 'project/dogimages/dogImages/train/097.Lakeland_terrier',
 'project/dogimages/dogImages/train/098.Leonberger',
 'project/dogimages/dogImages/train/099.Lhasa_apso',
 'project/dogimages/dogImages/train/100.Lowchen',
 'project/dogimages/dogImages/train/101.Maltese',
 'project/dogimages/dogImages/train/102.Manchester_terrier',
 'project/dogimages/dogImages/train/103.Mastiff',
 'project/dogimages/dogImages/train/104.Miniature_schnauzer',
 'project/dogimages/dogImages/train/105.Neapolitan_mastiff',
 'project/dogimages/dogImages/train/106.Newfoundland',
 'project/dogimages/dogImages/train/107.Norfolk_terrier',
 'project/dogimages/dogImages/train/108.Norwegian_buhund',
 'project/dogimages/dogImages/train/109.Norwegian_elkhound',
 'project/dogimages/dogImages/train/110.Norwegian_lundehund',
 'project/dogimages/dogImages/train/111.Norwich_terrier',
 'project/dogimages/dogImages/train/112.Nova_scotia_duck_tolling_retriever',
 'project/dogimages/dogImages/train/113.Old_english_sheepdog',
 'project/dogimages/dogImages/train/114.Otterhound',
 'project/dogimages/dogImages/train/115.Papillon',
 'project/dogimages/dogImages/train/119.Parson_russell_terrier',
 'project/dogimages/dogImages/train/117.Pekingese',
 'project/dogimages/dogImages/train/118.Pembroke_welsh_corgi',
 'project/dogimages/dogImages/train/119.Petit_basset_griffon_vendeen',
 'project/dogimages/dogImages/train/120.Pharaoh_hound',
 'project/dogimages/dogImages/train/121.Plott',
 'project/dogimages/dogImages/train/122.Pointer',
 'project/dogimages/dogImages/train/123.Pomeranian',
 'project/dogimages/dogImages/train/124.Poodle',
 'project/dogimages/dogImages/train/125.Portuguese_water_dog',
 'project/dogimages/dogImages/train/126.Saint_bernard',
 'project/dogimages/dogImages/train/127.Silky_terrier',
 'project/dogimages/dogImages/train/128.Smooth_fox_terrier',
 'project/dogimages/dogImages/train/129.Tibetan_mastiff',
 'project/dogimages/dogImages/train/130.Welsh_springer_spaniel',
 'project/dogimages/dogImages/train/131.Wirehaired_pointing_griffon',
 'project/dogimages/dogImages/train/132.Xoloitzcuintli',
 'project/dogimages/dogImages/train/133.Yorkshire_terrier']

VGG19_model = load_model('./saved_models/VGG19.hdf5')


def VGG19_predict_breed(img_path):
    """
    Use VGG19 model to predict dog breed. Depends on dog_names list.
    :param img_path:
    :return: dog names
    """
    # extract bottleneck features
    bottleneck_feature = extract_VGG19(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = VGG19_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    print(dog_names[np.argmax(predicted_vector)])
    return dog_names[np.argmax(predicted_vector)]


def path_to_tensor(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        # Normalize the image tensor
        return np.expand_dims(x, axis=0)#.astype('float32')/255
    except IOError:
        print(f"Warning: Skipping corrupted image {img_path}")
        return None


def paths_to_tensor(img_paths):
    batch_tensors = []
    for img_path in img_paths:
        tensor = path_to_tensor(img_path)
        if tensor is not None:
            batch_tensors.append(tensor[0])
    return np.array(batch_tensors)


face_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_alt.xml')


def face_detector(img_path):
 """
 Returns True if face is detected with haarcascade.
 :param img_path:
 :return: Boolean
 """
 img = cv2.imread(img_path)
 gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 faces = face_cascade.detectMultiScale(gray)
 return len(faces) > 0


def dog_detector(img_path):
 """
 Returns True if dog is detected. The label values are fixed and might change over time.
 :param img_path:
 :return: Boolean
 """
 prediction = ResNet50_predict_labels(img_path)
 return ((prediction <= 268) & (prediction >= 151))


def detect_human_or_dog(file_path):
 """
 Returns information if the provided image contains human or dog. Also gives information which dog breed is most likely.

 :param file_path: file path to the image
 :return String with predicted information
 """

 face_detected = face_detector(file_path)
 dog_detected = dog_detector(file_path)
 breed_detected = VGG19_predict_breed(file_path)
 breed_only = breed_detected.split('.')[-1]
 if face_detected:
  return f"If you were a dog, you would be a {breed_only}"

 elif dog_detected:
  return f"The dog in the image is a {breed_only}"

 else:
  return "Ooops, the algorithm could not find a human or dog..."