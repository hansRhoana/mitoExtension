#
#
# Mitochondria 3D Reconstruction
#
# 

import mahotas
import scipy.ndimage
import scipy.misc
import numpy as np
import glob
import h5py
import pylab
import pymorph
import matplotlib

def normalize_image(original_image, saturation_level=0.005):
    sorted_image = np.sort( np.uint8(original_image).ravel() )
    minval = np.float32( sorted_image[ len(sorted_image) * ( saturation_level / 2 ) ] )
    maxval = np.float32( sorted_image[ len(sorted_image) * ( 1 - saturation_level / 2 ) ] )
    norm_image = np.float32(original_image - minval) * ( 255 / (maxval - minval))
    norm_image[norm_image < 0] = 0
    norm_image[norm_image > 255] = 255
    return np.uint8(norm_image)

img_search_string = 'images\\*.png'
mito_search_string = 'images\\moarimg\\*.png'

img_files = sorted( glob.glob( img_search_string ) )
mito_files = sorted( glob.glob( mito_search_string ) )

for img_num in range(1):
# for img_num in range(len(img_files)):

    img_filename = img_files[img_num]
    mito_filename = mito_files[img_num]
    
    # Open the images
    print img_filename
    source_img = mahotas.imread(img_filename)
    if len(source_img.shape) == 3:
        source_img = source_img[:,:,0]
    source_img = normalize_image(source_img)
    

    print mito_filename
    mito_img = mahotas.imread(mito_filename)
    mito_img = normalize_img(mito_img)
    pylab.imshow(mito_img)
    pylab.gray()
    pylab.show()

    print 'Done'

    # Values for the erode/dilate functions
    
    radius = 2
    y,x = np.ogrid[-radius:radius+1, -radius:radius+1]
    disc = x*x + y*y <= radius*radius

    # Make some classification
    
    #ilastik_filename = img_filename.replace('.png', '.png_processed.h5')
    #prob_file = h5py.File('Thousands_mito_em_s1152.png_processed (1).h5', 'r')
    #label_index = 1
    #mito_prob = prob_file['/volume/prediction'][0,0,:,:,label_index]
    #prob_file.close()
    
    # loop over the results
    
    i = 50
    for i in range (50, 99)
      ilastik_filename = img_filename.replace('.png', '.png_processed.h5')
      prob_file = h5py.File('Thousands_mito_em_s11[i].png_processed (1).h5', 'r')
      label_index = 1
      mito_prob[i] = prob_file['/volume/prediction'][0,0,:,:,label_index]
      prob_file.close()
   
    for i in range (50, 99)
      mito_prob[i] = mito_prob[i]*100
      blur_img[i] = scipy.ndimage.gaussian_filter(mito_prob[i], 16)
      mito_pred2[i] = blur_img[i]<85
      mito_pred2[i] = mahotas.erode(mito_pred2[i], disc)
    # load
    ilastik_filename = img_filename.replace('.png', '.png_processed.h5')
    prob_file = h5py.File('Thousands_mito_em_s1152.png_processed (1).h5', 'r')
    label_index = 1
    mito_prob[i] = prob_file['/volume/prediction'][0,0,:,:,label_index]
    prob_file.close()
    blur_img[i] = scipy.ndimage.gaussian_filter(mito_prob[i], 13)
    mito_pred2[i] = blur_img<.85
    mito_pred2[i] = mahotas.erode(mito_pred2[i], disc)
    
    ilastik_filename = img_filename.replace('.png', '.png_processed.h5')
    prob_file = h5py.File('Thousands_mito_em_s1152.png_processed (1).h5', 'r')
    label_index = 1
    mito_prob[i] = prob_file['/volume/prediction'][0,0,:,:,label_index]
    prob_file.close()
    blur_img[i] = scipy.ndimage.gaussian_filter(mito_prob[i], 13)
    mito_pred2[i] = blur_img<.85
    mito_pred2[i] = mahotas.erode(mito_pred2[i], disc)
    
    ilastik_filename = img_filename.replace('.png', '.png_processed.h5')
    prob_file = h5py.File('Thousands_mito_em_s1152.png_processed (1).h5', 'r')
    label_index = 1
    mito_prob[i] = prob_file['/volume/prediction'][0,0,:,:,label_index]
    prob_file.close()
    blur_img[i] = scipy.ndimage.gaussian_filter(mito_prob[i], 13)
    mito_pred2[i] = blur_img<.85
    mito_pred2[i] = mahotas.erode(mito_pred2[i], disc)
    
    ilastik_filename = img_filename.replace('.png', '.png_processed.h5')
    prob_file = h5py.File('Thousands_mito_em_s1152.png_processed (1).h5', 'r')
    label_index = 1
    mito_prob[i] = prob_file['/volume/prediction'][0,0,:,:,label_index]
    prob_file.close()
    blur_img[i] = scipy.ndimage.gaussian_filter(mito_prob[i], 13)
    mito_pred2[i] = blur_img<.85
    mito_pred2[i] = mahotas.erode(mito_pred2[i], disc)
    
    ilastik_filename = img_filename.replace('.png', '.png_processed.h5')
    prob_file = h5py.File('Thousands_mito_em_s1152.png_processed (1).h5', 'r')
    label_index = 1
    mito_prob[i] = prob_file['/volume/prediction'][0,0,:,:,label_index]
    prob_file.close()
    blur_img[i] = scipy.ndimage.gaussian_filter(mito_prob[i], 13)
    mito_pred2[i] = blur_img<.85
    mito_pred2[i] = mahotas.erode(mito_pred2[i], disc)
    
    ilastik_filename = img_filename.replace('.png', '.png_processed.h5')
    prob_file = h5py.File('Thousands_mito_em_s1152.png_processed (1).h5', 'r')
    label_index = 1
    mito_prob[i] = prob_file['/volume/prediction'][0,0,:,:,label_index]
    prob_file.close()
    blur_img[i] = scipy.ndimage.gaussian_filter(mito_prob[i], 13)
    mito_pred2[i] = blur_img<.85
    mito_pred2[i] = mahotas.erode(mito_pred2[i], disc)
    
    ilastik_filename = img_filename.replace('.png', '.png_processed.h5')
    prob_file = h5py.File('Thousands_mito_em_s1152.png_processed (1).h5', 'r')
    label_index = 1
    mito_prob[i] = prob_file['/volume/prediction'][0,0,:,:,label_index]
    prob_file.close()
    blur_img[i] = scipy.ndimage.gaussian_filter(mito_prob[i], 13)
    mito_pred2[i] = blur_img<.85
    mito_pred2[i] = mahotas.erode(mito_pred2[i], disc)
    
    ilastik_filename = img_filename.replace('.png', '.png_processed.h5')
    prob_file = h5py.File('Thousands_mito_em_s1152.png_processed (1).h5', 'r')
    label_index = 1
    mito_prob[i] = prob_file['/volume/prediction'][0,0,:,:,label_index]
    prob_file.close()
    blur_img[i] = scipy.ndimage.gaussian_filter(mito_prob[i], 13)
    mito_pred2[i] = blur_img<.85
    mito_pred2[i] = mahotas.erode(mito_pred2[i], disc)
    
    ilastik_filename = img_filename.replace('.png', '.png_processed.h5')
    prob_file = h5py.File('Thousands_mito_em_s1152.png_processed (1).h5', 'r')
    label_index = 1
    mito_prob[i] = prob_file['/volume/prediction'][0,0,:,:,label_index]
    prob_file.close()
    blur_img[i] = scipy.ndimage.gaussian_filter(mito_prob[i], 13)
    mito_pred2[i] = blur_img<.85
    mito_pred2[i] = mahotas.erode(mito_pred2[i], disc)
    
    ilastik_filename = img_filename.replace('.png', '.png_processed.h5')
    prob_file = h5py.File('Thousands_mito_em_s1152.png_processed (1).h5', 'r')
    label_index = 1
    mito_prob[i] = prob_file['/volume/prediction'][0,0,:,:,label_index]
    prob_file.close()
    blur_img[i] = scipy.ndimage.gaussian_filter(mito_prob[i], 13)
    mito_pred2[i] = blur_img<.85
    mito_pred2[i] = mahotas.erode(mito_pred2[i], disc)
    
    ilastik_filename = img_filename.replace('.png', '.png_processed.h5')
    prob_file = h5py.File('Thousands_mito_em_s1152.png_processed (1).h5', 'r')
    label_index = 1
    mito_prob[i] = prob_file['/volume/prediction'][0,0,:,:,label_index]
    prob_file.close()
    blur_img[i] = scipy.ndimage.gaussian_filter(mito_prob[i], 13)
    mito_pred2[i] = blur_img<.85
    mito_pred2[i] = mahotas.erode(mito_pred2[i], disc)
    
    ilastik_filename = img_filename.replace('.png', '.png_processed.h5')
    prob_file = h5py.File('Thousands_mito_em_s1152.png_processed (1).h5', 'r')
    label_index = 1
    mito_prob[i] = prob_file['/volume/prediction'][0,0,:,:,label_index]
    prob_file.close()
    blur_img[i] = scipy.ndimage.gaussian_filter(mito_prob[i], 13)
    mito_pred2[i] = blur_img<.85
    mito_pred2[i] = mahotas.erode(mito_pred2[i], disc)
    
    ilastik_filename = img_filename.replace('.png', '.png_processed.h5')
    prob_file = h5py.File('Thousands_mito_em_s1152.png_processed (1).h5', 'r')
    label_index = 1
    mito_prob[i] = prob_file['/volume/prediction'][0,0,:,:,label_index]
    prob_file.close()
    blur_img[i] = scipy.ndimage.gaussian_filter(mito_prob[i], 13)
    mito_pred2[i] = blur_img<.85
    mito_pred2[i] = mahotas.erode(mito_pred2[i], disc)
    
    ilastik_filename = img_filename.replace('.png', '.png_processed.h5')
    prob_file = h5py.File('Thousands_mito_em_s1152.png_processed (1).h5', 'r')
    label_index = 1
    mito_prob[i] = prob_file['/volume/prediction'][0,0,:,:,label_index]
    prob_file.close()
    blur_img[i] = scipy.ndimage.gaussian_filter(mito_prob[i], 13)
    mito_pred2[i] = blur_img<.85
    mito_pred2[i] = mahotas.erode(mito_pred2[i], disc)
    
    ilastik_filename = img_filename.replace('.png', '.png_processed.h5')
    prob_file = h5py.File('Thousands_mito_em_s1152.png_processed (1).h5', 'r')
    label_index = 1
    mito_prob[i] = prob_file['/volume/prediction'][0,0,:,:,label_index]
    prob_file.close()
    blur_img[i] = scipy.ndimage.gaussian_filter(mito_prob[i], 13)
    mito_pred2[i] = blur_img<.85
    mito_pred2[i] = mahotas.erode(mito_pred2[i], disc)
    
    ilastik_filename = img_filename.replace('.png', '.png_processed.h5')
    prob_file = h5py.File('Thousands_mito_em_s1152.png_processed (1).h5', 'r')
    label_index = 1
    mito_prob[i] = prob_file['/volume/prediction'][0,0,:,:,label_index]
    prob_file.close()
    blur_img[i] = scipy.ndimage.gaussian_filter(mito_prob[i], 13)
    mito_pred2[i] = blur_img<.85
    mito_pred2[i] = mahotas.erode(mito_pred2[i], disc)
    
    ilastik_filename = img_filename.replace('.png', '.png_processed.h5')
    prob_file = h5py.File('Thousands_mito_em_s1152.png_processed (1).h5', 'r')
    label_index = 1
    mito_prob[i] = prob_file['/volume/prediction'][0,0,:,:,label_index]
    prob_file.close()
    blur_img[i] = scipy.ndimage.gaussian_filter(mito_prob[i], 13)
    mito_pred2[i] = blur_img<.85
    mito_pred2[i] = mahotas.erode(mito_pred2[i], disc)
    
    ilastik_filename = img_filename.replace('.png', '.png_processed.h5')
    prob_file = h5py.File('Thousands_mito_em_s1152.png_processed (1).h5', 'r')
    label_index = 1
    mito_prob[i] = prob_file['/volume/prediction'][0,0,:,:,label_index]
    prob_file.close()
    blur_img[i] = scipy.ndimage.gaussian_filter(mito_prob[i], 13)
    mito_pred2[i] = blur_img<.85
    mito_pred2[i] = mahotas.erode(mito_pred2[i], disc)
    
    ilastik_filename = img_filename.replace('.png', '.png_processed.h5')
    prob_file = h5py.File('Thousands_mito_em_s1152.png_processed (1).h5', 'r')
    label_index = 1
    mito_prob[i] = prob_file['/volume/prediction'][0,0,:,:,label_index]
    prob_file.close()
    blur_img[i] = scipy.ndimage.gaussian_filter(mito_prob[i], 13)
    mito_pred2[i] = blur_img<.85
    mito_pred2[i] = mahotas.erode(mito_pred2[i], disc)
    
    ilastik_filename = img_filename.replace('.png', '.png_processed.h5')
    prob_file = h5py.File('Thousands_mito_em_s1152.png_processed (1).h5', 'r')
    label_index = 1
    mito_prob[i] = prob_file['/volume/prediction'][0,0,:,:,label_index]
    prob_file.close()
    blur_img[i] = scipy.ndimage.gaussian_filter(mito_prob[i], 13)
    mito_pred2[i] = blur_img<.85
    mito_pred2[i] = mahotas.erode(mito_pred2[i], disc)
    
    ilastik_filename = img_filename.replace('.png', '.png_processed.h5')
    prob_file = h5py.File('Thousands_mito_em_s1152.png_processed (1).h5', 'r')
    label_index = 1
    mito_prob[i] = prob_file['/volume/prediction'][0,0,:,:,label_index]
    prob_file.close()
    blur_img[i] = scipy.ndimage.gaussian_filter(mito_prob[i], 13)
    mito_pred2[i] = blur_img<.85
    mito_pred2[i] = mahotas.erode(mito_pred2[i], disc)
    
    ilastik_filename = img_filename.replace('.png', '.png_processed.h5')
    prob_file = h5py.File('Thousands_mito_em_s1152.png_processed (1).h5', 'r')
    label_index = 1
    mito_prob[i] = prob_file['/volume/prediction'][0,0,:,:,label_index]
    prob_file.close()
    blur_img[i] = scipy.ndimage.gaussian_filter(mito_prob[i], 13)
    mito_pred2[i] = blur_img<.85
    mito_pred2[i] = mahotas.erode(mito_pred2[i], disc)
    
    ilastik_filename = img_filename.replace('.png', '.png_processed.h5')
    prob_file = h5py.File('Thousands_mito_em_s1152.png_processed (1).h5', 'r')
    label_index = 1
    mito_prob[i] = prob_file['/volume/prediction'][0,0,:,:,label_index]
    prob_file.close()
    blur_img[i] = scipy.ndimage.gaussian_filter(mito_prob[i], 13)
    mito_pred2[i] = blur_img<.85
    mito_pred2[i] = mahotas.erode(mito_pred2[i], disc)
    
    
    ilastik_filename = img_filename.replace('.png', '.png_processed.h5')
    prob_file = h5py.File('Thousands_mito_em_s1152.png_processed (1).h5', 'r')
    label_index = 1
    mito_prob[i] = prob_file['/volume/prediction'][0,0,:,:,label_index]
    prob_file.close()
    blur_img[i] = scipy.ndimage.gaussian_filter(mito_prob[i], 13)
    mito_pred2[i] = blur_img<.85
    mito_pred2[i] = mahotas.erode(mito_pred2[i], disc)
    

    
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    
    
    # Connected Components Phase - optimization
    
    #print mito_prob.max()
    #print mito_prob.min()
    #mito_prob = mito_prob*100
    #blur_img = scipy.ndimage.gaussian_filter(mito_prob, 16)
    #print blur_img.max()
    #print blur_img.min()
    #mito_pred2 = blur_img<85
    #pylab.imshow(mito_pred2)
    #pylab.gray()
    #pylab.show()
    
    # Values for the erode/dilate functions

    #radius = 2
    #,x = np.ogrid[-radius:radius+1, -radius:radius+1]
    #disc = x*x + y*y <= radius*radius

    

    
    
  
    
   
##  # Erode makes everything smaller and removes small objects
    #mito_pred2 = mahotas.erode(mito_pred2, disc)
##  # Dilate makes everything bigger and removes small holes
    # mito_pred2 = mahotas.dilate(mito_pred2, disc)

    #################################
    #
    # Optimization
    #
    ##
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    ######################################
   

    
print 'Done.'
