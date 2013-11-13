#
#
# The Mitochondria Segmentation Base of the Rhoana Extension
#
# Current result:.3711

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

    


    # Make some classification
    
    ilastik_filename = img_filename.replace('.png', '.png_processed.h5')
    prob_file = h5py.File('Thousands_mito_em_s1152.png_processed (1).h5', 'r')
    label_index = 1
    mito_prob = prob_file['/volume/prediction'][0,0,:,:,label_index]
    prob_file.close()
    print mito_prob.max()
    print mito_prob.min()
    mito_prob = mito_prob*100
    blur_img = scipy.ndimage.gaussian_filter(mito_prob, 16)
    print blur_img.max()
    print blur_img.min()
    mito_pred2 = blur_img<85
    pylab.imshow(mito_pred2)
    pylab.gray()
    pylab.show()
    
    # Values for the erode/dilate functions

    radius = 2
    y,x = np.ogrid[-radius:radius+1, -radius:radius+1]
    disc = x*x + y*y <= radius*radius

   

    
    
  
    
   
##  # Erode makes everything smaller and removes small objects
    mito_pred2 = mahotas.erode(mito_pred2, disc)
##  # Dilate makes everything bigger and removes small holes
    # mito_pred2 = mahotas.dilate(mito_pred2, disc)

    # Predictions
    pylab.imshow(mito_pred2)
    pylab.gray()
    pylab.show()

    # Display the target output
    pylab.imshow(mito_img)
    pylab.gray()
    pylab.show()

    # Measure the result
    true_positives_h5 = np.sum(np.logical_and(mito_pred2 > 0, mito_img > 0)) 
    false_positives_h5 = np.sum(np.logical_and(mito_pred2 > 0, mito_img  == 0))
    true_negatives_h5 = np.sum(np.logical_and(mito_pred2 == 0, mito_img  == 0))
    false_negatives_h5 = np.sum(np.logical_and(mito_pred2 == 0, mito_img > 0))

    total_true_h5 = true_positives_h5 + false_negatives_h5
    total_true_not_h5 = true_negatives_h5 + false_positives_h5

    print 'Image {0}: Found {1} mitochondria pixels out of {2} ({3:0.2f}%) with {4} false positives ({5:0.2f}% of true mito).'.format(
        img_num,
        true_positives_h5,
        total_true_h5,
        float(true_positives_h5) / float(total_true_h5) * 100.0,
        false_positives_h5,
        float(false_positives_h5) / float(total_true_h5) * 100.0
        ) 

    precision_h5 = float(true_positives_h5) / float(true_positives_h5 + false_positives_h5)
    recall_h5 = float(true_positives_h5) / float(true_positives_h5 + false_negatives_h5)
    Fscore_h5 = 2 * precision_h5 * recall_h5 / (precision_h5 + recall_h5)
    print 'Precision = {0:0.4f}, Recall = {1:0.4f}, F-Score = {2:0.4f}'.format(precision_h5, recall_h5, Fscore_h5,)

    print ''
    #
    #
    # Current result:.3711
    #
    #
    
    # Predictions with regmax method

    blur_imgH = scipy.ndimage.gaussian_filter(mito_prob, 16)
    mito_pred3 = blur_imgH<90
    blur_imgH = blur_imgH.astype(np.uint8)
    
    mito_pred3 = mahotas.erode(mito_pred3, disc)
    mito_pred3 = mito_pred3.astype(np.uint8)
    #labeled, nr_objects = scipy.ndimage.label(mito_pred3)
    #print nr_objects
    #labeled = labeled.astype(np.uint8)

    rmax = pymorph.regmax(blur_imgH)
    pylab.imshow(pymorph.overlay(mito_prob, rmax))
    pylab.gray()
    pylab.show()
    seeds,nr_nuclei = scipy.ndimage.label(rmax)
    print nr_nuclei
    pylab.imshow(mito_pred3)
    pylab.show()
    dist = scipy.ndimage.distance_transform_edt(mito_pred3)
    dist = dist.max() - dist
    dist-=dist.min()
    dist = dist/float(dist.ptp())*255
    dist = dist.astype(np.uint8)
    pylab.imshow(dist)
    pylab.gray()
    pylab.show()
    nuclei = pymorph.cwatershed(dist, seeds)
    nuclei = mahotas.erode(nuclei, disc)
    pylab.imshow(nuclei)
    pylab.gray()
    pylab.show()




    pylab.imshow(nuclei)
    pylab.gray()
    pylab.show()

    pylab.imshow(mito_img)
    pylab.gray()
    pylab.show()
    
    true_positives = np.sum(np.logical_and(nuclei > 0, mito_img > 0)) #hansmod/same deal
    false_positives = np.sum(np.logical_and(nuclei > 0, mito_img == 0))
    true_negatives = np.sum(np.logical_and(nuclei == 0, mito_img == 0))
    false_negatives = np.sum(np.logical_and(nuclei == 0, mito_img > 0))

    total_true_mito = true_positives + false_negatives
    total_true_not_mito = true_negatives + false_positives

    print 'Image {0}: Found {1} mitochondria pixels out of {2} ({3:0.2f}%) with {4} false positives ({5:0.2f}% of true mito).'.format(
        img_num,
        true_positives,
        total_true_mito,
        float(true_positives) / float(total_true_mito) * 100.0,
        false_positives,
        float(false_positives) / float(total_true_mito) * 100.0
        )

    precision = float(true_positives) / float(true_positives + false_positives)
    recall = float(true_positives) / float(true_positives + false_negatives)
    Fscore = 2 * precision * recall / (precision + recall)

    print 'Precision = {0:0.4f}, Recall = {1:0.4f}, F-Score = {2:0.4f}'.format(precision, recall, Fscore,)

    print ''

    #testing 
    #
    #Regmax method outputs grossly inadequate results for our purposes
    
    
print 'Done.'
