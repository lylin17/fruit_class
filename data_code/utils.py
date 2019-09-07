import os
import pickle

from keras.applications.vgg16 import preprocess_input
import numpy as np
from PIL import Image

def preprocess(fruit, num_images):
    '''preprocess scrapped images to select RGB images and resize to 244 X 244
    
    Args:
        fruit(str): class of fruit images
        num_images(int): first num_image images to be output
    
    Returns:
        fruit_arr(np array): 244*244*3 array for each fruit image     
    '''
    
    fruitlist = []
    
    for filename in os.listdir("./downloads/{0} fruit".format(fruit)):
        try:
            img = Image.open("./downloads/{0} fruit/{1}".format(fruit,filename))
            width, height = img.size

            if height >= width:
                basewidth = 244
                wpercent = (basewidth / float(img.size[0]))
                hsize = int((float(img.size[1]) * float(wpercent)))

            if height < width:
                hsize = 244
                hpercent = (hsize / float(img.size[1]))
                basewidth = int((float(img.size[0]) * float(hpercent)))    

            img = img.resize((basewidth,hsize), Image.ANTIALIAS)
            width, height = img.size
            left = (width - 244) / 2
            top = (height - 244) / 2
            right = (width + 244) / 2
            bottom = (height + 244) / 2
            img = img.crop((left, top, right, bottom))

            arr = np.array(img)
            if arr.shape == (244, 244, 3): fruitlist.append(np.array(img))           
            
        except:
            # Images with errors are ignored
            pass    
        
        #Stop if reach num_images of processed images
        if np.stack(fruitlist).shape[0] + 1 == num_images: break
        
    allimg = np.stack(fruitlist)   
    
    #preprocess input array to required format by keras pre-trained model for transfer learning   
    if allimg.shape[0] + 1 >= num_images:
        fruit_arr = preprocess_input(allimg)[:num_images, :, :, :]   
    else:
        print('Available images less than num images!')
        fruit_arr = None
        
    return fruit_arr

def preprocess_all(num_images, fruit_classes=['apples', 'oranges', 'pears']):
    '''preprocess scrapped images for all fruit classes
    
    Wrapper around preprocess function to process all fruit classes and save them as pickle files in ./data directory
    
    Args:
        num_images(int): first num_image images to be output. This number is the same for all fruit classes
        fruit_classes(list): list of strings of all fruit classes. Fruit class directory must be present in ./downloads as ./downloads/<fruit_class> fruit/ Defaults to ['apples','oranges','pears'].
    
    Returns:
        None, but output a pickle file for each fruit class     
    '''
    #setup data directory in current directory if not present
    if not os.path.isdir('./data'):
        os.mkdir('./data')
    
    for fruit in fruit_classes:
    
        fruit_arr = preprocess(fruit, num_images)
        
        #save to picke file only if data meet num_images criteria
        if fruit_arr is not None:
            with open('./data/x_{0}.pkl'.format(fruit), 'wb') as file:
                pickle.dump(fruit_arr, file)