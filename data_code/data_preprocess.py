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
    fruit_count = 0
    
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
            
            fruit_count +=1
            
            if fruit_count == num_images: break

        except:
            # Images with errors are ignored
            None     
    
    if fruit_count < num_images: raise ValueError('Available images less than num images!')
    
    allimg = np.stack(fruitlist) 
    
    #preprocess input array to required format by keras pre-trained model for transfer learning
    fruit_arr = preprocess_input(allimg)[:num_images, :, :, :]
                
    return fruit_arr


#Data preprocessing

#setup data directory in current directory if not present
if not os.path.isdir('./data'):
    os.mkdir('./data')

num_image = 350

for fruit in ['apples', 'oranges', 'pears']:
    
    fruit_arr = preprocess(fruit, num_image)
    
    with open('./data/x_{0}.pkl'.format(fruit), 'wb') as file:
            pickle.dump(fruit_arr, file)

      
 
