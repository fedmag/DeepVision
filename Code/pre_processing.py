### CREATED BY anhanat Deesamutara
import numpy as np
from PIL import Image, ImageOps
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import vectorize

##############
## Subtasks ##
##############

def makefolder(path):
    try:
        os.mkdir(path)
    except:
        pass
def makesubfolder(path):
    try:
        os.makedirs(path)
    except:
        pass
    
def add_artifact(image,occult_ratio,aspect_ratio=1):
    mat_1 = np.array(image)[:,:,0]
    size,_ = mat_1.shape
    kernel_size = int(occult_ratio * size)
    px,py = [np.random.randint(0,high = size-kernel_size),np.random.randint(0,high = size-kernel_size)]
    # px2,py2 = [np.random.randint(0,high = size-kernel_size),np.random.randint(0,high = size-kernel_size)] # uncomment to have two damages
    
    artifact = np.zeros((kernel_size,kernel_size))
    # artifact2 = np.zeros((kernel_size,kernel_size)) # uncomment to have two damages
    
    mat_1[px:px + kernel_size , py:py + kernel_size] = artifact
    # mat_1[px2:px2 + kernel_size , py2:py2 + kernel_size] = artifact2  # this adds a second 'damage'
    
    return mat_1

###############################
### Preprocessing Pipelines ###
###############################

def preprocessing_pipeline():
    size = [100] # you can create several folders each of them with a different image size just by adding the desired size to this list
    occult_ratio = [0.20] # you can create several folders each of them with a different damage size just by adding the (% of the damage)/100 to this list
    
    dir_1 = os.listdir('C:\\Users\\Federico\\Desktop\\Computer Science\\Deep Vision\\Denoise for inpainting\\Code FACES\\Images')
    dir_2 = ['C:\\Users\\Federico\\Desktop\\Computer Science\\Deep Vision\\Denoise for inpainting\\Code FACES\\Images\\'+str(d) for d in dir_1]
    files = np.array([os.listdir(d) for d in dir_2])
    makefolder('groundtruth_output_40') # name of the folder for the GT
    makefolder('occulted_output_40') # name of the folder for the damaged images
    
    index = 0
    for s in size:
        desired_size = int(s)
        print(desired_size)
        dir_3 = 'groundtruth_output_40/resized_' + str(desired_size)
        makesubfolder(dir_3) 
        
        for i in range(len(files)):
            for j in tqdm(range(len(files[i])),position=0):
                
                im = Image.open(dir_2[i]+"/"+files[i][j])
                img_greyscaled = im.convert('LA')
                img_resized = img_greyscaled.resize((desired_size,desired_size))    
                img_resized.save(dir_3 + '/' + files[i][j], "PNG")
                
                for r in occult_ratio:
                    path = 'occulted_output_40/'+ str(int(100*r)) + '_percent/' + str(int(size[index]))
                    makesubfolder(path)
                    test = img_resized
                    img_occulted = add_artifact(test,r)
                    img_final = Image.fromarray(img_occulted)
                    img_final.save(path +'/'+ files[i][j],"PNG")
        index += 1
        
preprocessing_pipeline()

