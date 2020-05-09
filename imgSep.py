import os
import pandas as pd
from tqdm import tqdm
import shutil

from utils.coco.coco import COCO
from config import Config


def create_data_subset(folder_name, category):
    """
    Assuming that the dataset has already been downloaded. Here is an example of getting all images and the
    corresponding captions from the training dataset for the category horse. The images and the csv file containing
    the image captions are saved into a new folder. You can easily generalize this approach to separate all
    categories into new folders
    """
    # note this only refers to  the training set and not the validation set
    # coco = COCO('train/instances_train2014.json')

    coco = COCO('val/instances_val2014.json')

    # note this only refers to the captions of the training set and not the validation set
    #caps = COCO('train/captions_train2014.json')

    caps = COCO('val/captions_val2014.json')

    categories = coco.loadCats(coco.getCatIds())
    names = [cat['name'] for cat in categories]

    print("Available categories: ")
    for index, n in enumerate(names):
        print(index, n)

    category_ids = coco.getCatIds(catNms=[category])
    image_ids = coco.getImgIds(catIds=category_ids)
    images = coco.loadImgs(image_ids)
    annIds = caps.getAnnIds(imgIds=image_ids)
    annotations = caps.loadAnns(annIds)

    # Split the annotations every 5 captions since there are 5 captions for each image
    annotations = [annotations[x:x + 5] for x in range(0, len(annotations), 5)]

    # Create empty dataframe with two columns for the image file name and the corresponding captions
    df = pd.DataFrame(columns=['image_id', 'image_file', 'caption'])

    # Create folder in for the images of the selected category
    # os.mkdir(folder_name)

    print('Folder created to store images....')

    # Create map for image id (key) to captions (values)
    captions_dict = {}
    for i, n in enumerate(annotations):
        captions_dict[annotations[i][0]['image_id']] = annotations[i]

    print('Retrieving images....')

    person_file_names = []
    for img in tqdm(images):
        person_file_names.append(img['file_name'])
        for entry in captions_dict[img['id']]:
            df.loc[len(df)] = [entry['image_id'], img['file_name'], entry['caption']]

    print('Caption csv creation in progress....')

    # Convert dataframe to csv file and save to folder
    df.to_csv(folder_name + "/val_person_captions.csv", index=False)

    print('Csv created....')

    print('Storing images....')

    # # Copy all images of given category to new folder
    # for filename in tqdm(os.listdir('train/images')):
    #     if filename in person_file_names:
    #         shutil.copy(os.path.join('train/images', filename), folder_name)

    print('Done creating data subset with images....')


def prepare_data_subset(filepath):
    file_data = pd.read_csv(filepath)
    image_ids = file_data['image_id'].values
    image_files = file_data['image_file'].values
    captions = file_data['caption'].values

    # captions = [coco.anns[ann_id]['caption'] for ann_id in coco.anns]
    # image_ids = [coco.anns[ann_id]['image_id'] for ann_id in coco.anns]

    new_image_files = [os.path.join('val/images/',
                                    image_file)
                       for image_file in image_files]

    file_data = pd.DataFrame({'image_id': image_ids,
                              'image_file': new_image_files,
                              'caption': captions})

    # file_data = file_data[:1000]
    newfilepath = filepath + '2'
    file_data.to_csv('person_images/val_person_captions2.csv')


def Preapre_text_file(filepath):
    file_data = pd.read_csv(filepath)
    image_ids = file_data['image_id'].values
    captions = file_data['caption'].values

    file_data = pd.DataFrame({'image_id': image_ids,
                              'caption': captions})

    train_output = file_data.iloc[:, 0]
    train_output_dup = train_output.drop_duplicates()
    train_output_final = train_output_dup.iloc[0:6000]
    train_output_final.to_csv(r'person_images/mscoco_people.trainImages.txt', header=None, index=None, sep='\t',
                              mode='a')

    val_output = file_data.iloc[:, 0]
    val_output_dup = val_output.drop_duplicates()
    val_output_final = val_output_dup.iloc[6001:7001]
    val_output_final.to_csv(r'person_images/mscoco_people.valImages.txt', header=None, index=None, sep='\t', mode='a')


def Preapre_token_file(filepath):
    file_data = pd.read_csv(filepath)
    image_ids = file_data['image_id'].values
    captions = file_data['caption'].values

    token_file_data = pd.DataFrame({'image_id': image_ids,
                                    'caption': captions})

    # column_names = ["image_id", "caption"]
    # token_file_data = pd.DataFrame(columns=column_names)

    token_file_data.to_csv(r'person_images/mscoco_people.token.txt', header=None, index=None, sep='\t', mode='a')


def Preapre_train_val_Img(train,val):


    file_data = pd.read_csv(train)

    file_names = file_data.iloc[:, 0]

    person_file_names = []
    for img in tqdm(file_names):
        person_file_names.append(img)

    # Copy all images of given category to new folder
    for filename in tqdm(os.listdir('person_images/images')):
        if filename in person_file_names:
            shutil.copy(os.path.join('person_images/images', filename), 'person_images/train-imgs')

    file_data_val = pd.read_csv(val)

    print('Done creating data subset with train images....')

    file_names_val = file_data_val.iloc[:, 0]

    person_file_names_val = []
    for img in tqdm(file_names_val):
        person_file_names_val.append(img)

    # Copy all images of given category to new folder
    for filename_val in tqdm(os.listdir('person_images/images')):
        if filename_val in person_file_names_val:
            shutil.copy(os.path.join('person_images/images', filename_val), 'person_images/train-imgs')

    print('Done creating data subset with val images....')


# image_id	image_file	caption
# 318556	./train/images/COCO_train2014_000000318556.jpg	a very clean and well decorated empty bathroom.

# ---------------------------------------
# usage

create_data_subset(folder_name='person_images', category='person')
prepare_data_subset('person_images/val_person_captions.csv')

#Preapre_text_file('person_images/person_captions.csv')
#Preapre_token_file('person_images/person_captions.csv')
#Preapre_train_val_Img('person_images/mscoco_people.trainImages.txt','person_images/mscoco_people.valImages.txt')