import numpy as np

from flask_wtf import FlaskForm
from wtforms import validators, SubmitField, FileField
from keras.applications.vgg16 import preprocess_input
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

class ReusableForm(FlaskForm):
    """User entry form for entering specifics for generation"""
    
    # Starting seed
    file = FileField("Image File to Classify:", validators=[validators.InputRequired()])

    # Submit button
    submit = SubmitField("Classify")

def process_image(file, errpath):
    '''Reize input image to required format.
    
    Args:
        file (image): Input image by User
        errpath(str): path for image to display if dim error or read error

    Returns:
        img(PIL image): Result image if 
        arr(np array): Formated image, None if image read error (e.g. not RGB)
    '''
    
    try:         
        img = Image.open(file)
        width, height = img.size

        if height >= width:
            basewidth = 244
            wpercent = (basewidth / float(img.size[0]))
            hsize = int((float(img.size[1])  *float(wpercent)))

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
        
    except:
        img = Image.open(errpath)
        arr = None
        
    return img, arr
        
def pred_fruit(model, img, arr, graph, filepath, fontpath):
    '''Predict fruit class and output result as an image.
    
    Args:
        model(h5 file): Trained fruit classification model
        arr(np array): formatted array from input image
        graph(tf object): default tensorflow graph
        filepath(str): path to save result image
        fontpath(str): path to font file

    Returns:
        None, but output result image as png file
    '''        
    
    if arr is None:
        text = 'Image read error!' 
    
    elif arr.shape == (244, 244, 3):
        with graph.as_default():
            testfruit = arr
            testfruit = np.expand_dims(testfruit,axis=0)
            testfruit = preprocess_input(testfruit)
            pred  = model.predict(testfruit)
            text = str(int(np.max(pred)*100))+'% '+ ['Apple','Orange','Pear'][pred.argmax()]          

    else: 
        text = 'Image dim error!'
    
    new_im = Image.new('RGB', (344, 360),'ivory')
    draw = ImageDraw.Draw(new_im)
    new_im.paste(img, (50,50))
    
    font = ImageFont.truetype(fontpath, size=24)
    w, h = draw.textsize(text,font)
    draw.text(((244-w)/2+50,310),text,(0,0,0),font=font) 
    
    new_im.save(filepath)