#
#
# Mitochondria 3D Reconstruction + neurite detection, segmentation and reconstruction
# Update: OUTDATED SCRIPT - We now just merge the stack of mitochondria labeling results from mito_classify_3D with Rhoana's neurite prediction stack and segment and reconstruct the resulting stack
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
from enthought.mayavi import mlab

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
    
    radius = 1.5
    y,x = np.ogrid[-radius:radius+1, -radius:radius+1]
    disc = x*x + y*y <= radius*radius

    # Make some classification
    
    #ilastik_filename = img_filename.replace('.png', '.png_processed.h5')
    #prob_file = h5py.File('Thousands_mito_em_s1152.png_processed (1).h5', 'r')
    #label_index = 1
    #mito_prob = prob_file['/volume/prediction'][0,0,:,:,label_index]
    #prob_file.close()
    
    # load the results
    
    ilastik_filename = img_filename.replace('.png', '.png_processed.h5')
    prob_file = h5py.File('Thousands_mito_em_s1150.png_processed.h5', 'r')
    label_index = 1
    mito_prob = prob_file['/volume/prediction'][0,0,:,:,label_index]
    prob_file.close()
    blur_img = scipy.ndimage.gaussian_filter(mito_prob, 13)
    mito_pred2 = blur_img<.85
    mito_pred2 = mahotas.erode(mito_pred2, disc)

    
    prob_file2 = h5py.File('Thousands_mito_em_s1151.png_processed.h5', 'r')
    label_index = 1
    mito_prob2 = prob_file2['/volume/prediction'][0,0,:,:,label_index]
    prob_file2.close()
    blur_img2 = scipy.ndimage.gaussian_filter(mito_prob2, 13)
    mito_pred22 = blur_img2<.85
    mito_pred22 = mahotas.erode(mito_pred22, disc)

    
    prob_file3 = h5py.File('Thousands_mito_em_s1152.png_processed.h5', 'r')
    label_index = 1
    mito_prob3 = prob_file3['/volume/prediction'][0,0,:,:,label_index]
    prob_file3.close()
    blur_img3 = scipy.ndimage.gaussian_filter(mito_prob3, 13)
    mito_pred23 = blur_img3<.85
    mito_pred23 = mahotas.erode(mito_pred23, disc)

    
    prob_file4 = h5py.File('Thousands_mito_em_s1153.png_processed.h5', 'r')
    label_index = 1
    mito_prob4 = prob_file4['/volume/prediction'][0,0,:,:,label_index]
    prob_file4.close()
    blur_img4 = scipy.ndimage.gaussian_filter(mito_prob4, 13)
    mito_pred24 = blur_img4<.85
    mito_pred24 = mahotas.erode(mito_pred24, disc)

    
    prob_file5 = h5py.File('Thousands_mito_em_s1154.png_processed.h5', 'r')
    label_index = 1
    mito_prob5 = prob_file5['/volume/prediction'][0,0,:,:,label_index]
    prob_file5.close()
    blur_img5 = scipy.ndimage.gaussian_filter(mito_prob5, 13)
    mito_pred25 = blur_img5<.85
    mito_pred25 = mahotas.erode(mito_pred25, disc)//

    
    prob_file6 = h5py.File('Thousands_mito_em_s1155.png_processed.h5', 'r')
    label_index = 1
    mito_prob6 = prob_file6['/volume/prediction'][0,0,:,:,label_index]
    prob_file6.close()
    blur_img6 = scipy.ndimage.gaussian_filter(mito_prob6, 13)
    mito_pred26 = blur_img6<.85
    mito_pred26 = mahotas.erode(mito_pred26, disc)

    
    prob_file7 = h5py.File('Thousands_mito_em_s1156.png_processed.h5', 'r')
    label_index = 1
    mito_prob7 = prob_file7['/volume/prediction'][0,0,:,:,label_index]
    prob_file7.close()
    blur_img7 = scipy.ndimage.gaussian_filter(mito_prob7, 13)
    mito_pred27 = blur_img7<.85
    mito_pred27 = mahotas.erode(mito_pred27, disc)

    
    prob_file8 = h5py.File('Thousands_mito_em_s1157.png_processed.h5', 'r')
    label_index = 1
    mito_prob8 = prob_file8['/volume/prediction'][0,0,:,:,label_index]
    prob_file8.close()
    blur_img8 = scipy.ndimage.gaussian_filter(mito_prob8, 13)
    mito_pred28 = blur_img8<.85
    mito_pred28 = mahotas.erode(mito_pred28, disc)

    
    prob_file9 = h5py.File('Thousands_mito_em_s1158.png_processed.h5', 'r')
    label_index = 1
    mito_prob9 = prob_file9['/volume/prediction'][0,0,:,:,label_index]
    prob_file9.close()
    blur_img9 = scipy.ndimage.gaussian_filter(mito_prob9, 13)
    mito_pred29 = blur_img9<.85
    mito_pred29 = mahotas.erode(mito_pred29, disc)

    
    prob_file10 = h5py.File('Thousands_mito_em_s1159.png_processed.h5', 'r')
    label_index = 1
    mito_prob10 = prob_file10['/volume/prediction'][0,0,:,:,label_index]
    prob_file10.close()
    blur_img10 = scipy.ndimage.gaussian_filter(mito_prob10, 13)
    mito_pred210 = blur_img10<.85
    mito_pred210 = mahotas.erode(mito_pred210, disc)

    
    prob_file11 = h5py.File('Thousands_mito_em_s1160.png_processed.h5', 'r')
    label_index = 1
    mito_prob11 = prob_file11['/volume/prediction'][0,0,:,:,label_index]
    prob_file11.close()
    blur_img11 = scipy.ndimage.gaussian_filter(mito_prob11, 13)
    mito_pred211 = blur_img11<.85
    mito_pred211 = mahotas.erode(mito_pred211, disc)

    
    prob_file12 = h5py.File('Thousands_mito_em_s1161.png_processed.h5', 'r')
    label_index = 1
    mito_prob12 = prob_file12['/volume/prediction'][0,0,:,:,label_index]
    prob_file12.close()
    blur_img12 = scipy.ndimage.gaussian_filter(mito_prob12, 13)
    mito_pred212 = blur_img12<.85
    mito_pred212 = mahotas.erode(mito_pred212, disc)

   
    prob_file13 = h5py.File('Thousands_mito_em_s1162.png_processed.h5', 'r')
    label_index = 1
    mito_prob13 = prob_file13['/volume/prediction'][0,0,:,:,label_index]
    prob_file13.close()
    blur_img13 = scipy.ndimage.gaussian_filter(mito_prob13, 13)
    mito_pred213 = blur_img13<.85
    mito_pred213 = mahotas.erode(mito_pred213, disc)

    
    prob_file14 = h5py.File('Thousands_mito_em_s1163.png_processed.h5', 'r')
    label_index = 1
    mito_prob14 = prob_file14['/volume/prediction'][0,0,:,:,label_index]
    prob_file14.close()
    blur_img14 = scipy.ndimage.gaussian_filter(mito_prob14, 13)
    mito_pred214 = blur_img14<.85
    mito_pred214 = mahotas.erode(mito_pred214, disc)

   
    prob_file15 = h5py.File('Thousands_mito_em_s1164.png_processed.h5', 'r')
    label_index = 1
    mito_prob15 = prob_file15['/volume/prediction'][0,0,:,:,label_index]
    prob_file15.close()
    blur_img15 = scipy.ndimage.gaussian_filter(mito_prob15, 13)
    mito_pred215 = blur_img15<.85
    mito_pred215 = mahotas.erode(mito_pred215, disc)

    
    prob_file16 = h5py.File('Thousands_mito_em_s1165.png_processed.h5', 'r')
    label_index = 1
    mito_prob16 = prob_file16['/volume/prediction'][0,0,:,:,label_index]
    prob_file16.close()
    blur_img16 = scipy.ndimage.gaussian_filter(mito_prob16, 13)
    mito_pred216 = blur_img16<.85
    mito_pred216 = mahotas.erode(mito_pred216, disc)

    
    prob_file17 = h5py.File('Thousands_mito_em_s1166.png_processed.h5', 'r')
    label_index = 1
    mito_prob17 = prob_file17['/volume/prediction'][0,0,:,:,label_index]
    prob_file17.close()
    blur_img17 = scipy.ndimage.gaussian_filter(mito_prob17, 13)
    mito_pred217 = blur_img17<.85
    mito_pred217 = mahotas.erode(mito_pred217, disc)

    
    prob_file18 = h5py.File('Thousands_mito_em_s1167.png_processed.h5', 'r')
    label_index = 1
    mito_prob18 = prob_file18['/volume/prediction'][0,0,:,:,label_index]
    prob_file18.close()
    blur_img18 = scipy.ndimage.gaussian_filter(mito_prob18, 13)
    mito_pred218 = blur_img18<.85
    mito_pred218 = mahotas.erode(mito_pred218, disc)

    
    prob_file19 = h5py.File('Thousands_mito_em_s1168.png_processed.h5', 'r')
    label_index = 1
    mito_prob19 = prob_file19['/volume/prediction'][0,0,:,:,label_index]
    prob_file19.close()
    blur_img19 = scipy.ndimage.gaussian_filter(mito_prob19, 13)
    mito_pred219 = blur_img19<.85
    mito_pred219 = mahotas.erode(mito_pred219, disc)

    
    prob_file20 = h5py.File('Thousands_mito_em_s1169.png_processed.h5', 'r')
    label_index = 1
    mito_prob20 = prob_file20['/volume/prediction'][0,0,:,:,label_index]
    prob_file20.close()
    blur_img20 = scipy.ndimage.gaussian_filter(mito_prob20, 13)
    mito_pred220 = blur_img20<.85
    mito_pred220 = mahotas.erode(mito_pred220, disc)

    
    prob_file21 = h5py.File('Thousands_mito_em_s1170.png_processed.h5', 'r')
    label_index = 1
    mito_prob21 = prob_file21['/volume/prediction'][0,0,:,:,label_index]
    prob_file21.close()
    blur_img21 = scipy.ndimage.gaussian_filter(mito_prob21, 13)
    mito_pred221 = blur_img21<.85
    mito_pred221 = mahotas.erode(mito_pred221, disc)

    
    prob_file22 = h5py.File('Thousands_mito_em_s1171.png_processed.h5', 'r')
    label_index = 1
    mito_prob22 = prob_file22['/volume/prediction'][0,0,:,:,label_index]
    prob_file22.close()
    blur_img22 = scipy.ndimage.gaussian_filter(mito_prob22, 13)
    mito_pred222 = blur_img22<.85
    mito_pred222 = mahotas.erode(mito_pred222, disc)


    prob_file23 = h5py.File('Thousands_mito_em_s1172.png_processed.h5', 'r')
    label_index = 1
    mito_prob23 = prob_file23['/volume/prediction'][0,0,:,:,label_index]
    prob_file23.close()
    blur_img23 = scipy.ndimage.gaussian_filter(mito_prob23, 13)
    mito_pred223 = blur_img23<.85
    mito_pred223 = mahotas.erode(mito_pred223, disc)

    
    prob_file24 = h5py.File('Thousands_mito_em_s1173.png_processed.h5', 'r')
    label_index = 1
    mito_prob24 = prob_file24['/volume/prediction'][0,0,:,:,label_index]
    prob_file24.close()
    blur_img24 = scipy.ndimage.gaussian_filter(mito_prob24, 13)
    mito_pred224 = blur_img24<.85
    mito_pred224 = mahotas.erode(mito_pred224, disc)

    
    prob_file25 = h5py.File('Thousands_mito_em_s1174.png_processed.h5', 'r')
    label_index = 1
    mito_prob25 = prob_file25['/volume/prediction'][0,0,:,:,label_index]
    prob_file25.close()
    blur_img25 = scipy.ndimage.gaussian_filter(mito_prob25, 13)
    mito_pred225 = blur_img25<.85
    mito_pred225 = mahotas.erode(mito_pred225, disc)


    prob_file26 = h5py.File('Thousands_mito_em_s1175.png_processed.h5', 'r')
    label_index = 1
    mito_prob26 = prob_file26['/volume/prediction'][0,0,:,:,label_index]
    prob_file26.close()
    blur_img26 = scipy.ndimage.gaussian_filter(mito_prob26, 13)
    mito_pred226 = blur_img26<.85
    mito_pred226 = mahotas.erode(mito_pred226, disc)

    
    prob_file27 = h5py.File('Thousands_mito_em_s1176.png_processed.h5', 'r')
    label_index = 1
    mito_prob27 = prob_file27['/volume/prediction'][0,0,:,:,label_index]
    prob_file27.close()
    blur_img27 = scipy.ndimage.gaussian_filter(mito_prob27, 13)
    mito_pred227 = blur_img27<.85
    mito_pred227 = mahotas.erode(mito_pred227, disc)


    prob_file28 = h5py.File('Thousands_mito_em_s1177.png_processed.h5', 'r')
    label_index = 1
    mito_prob28 = prob_file28['/volume/prediction'][0,0,:,:,label_index]
    prob_file28.close()
    blur_img28 = scipy.ndimage.gaussian_filter(mito_prob28, 13)
    mito_pred228 = blur_img28<.85
    mito_pred228 = mahotas.erode(mito_pred228, disc)

    
    prob_file29 = h5py.File('Thousands_mito_em_s1178.png_processed.h5', 'r')
    label_index = 1
    mito_prob29 = prob_file29['/volume/prediction'][0,0,:,:,label_index]
    prob_file29.close()
    blur_img29 = scipy.ndimage.gaussian_filter(mito_prob29, 13)
    mito_pred229 = blur_img29<.85
    mito_pred229 = mahotas.erode(mito_pred229, disc)

   
    prob_file30 = h5py.File('Thousands_mito_em_s1179.png_processed.h5', 'r')
    label_index = 1
    mito_prob30 = prob_file30['/volume/prediction'][0,0,:,:,label_index]
    prob_file30.close()
    blur_img30 = scipy.ndimage.gaussian_filter(mito_prob30, 13)
    mito_pred230 = blur_img30<.85
    mito_pred230 = mahotas.erode(mito_pred230, disc)

    
    prob_file31 = h5py.File('Thousands_mito_em_s1180.png_processed.h5', 'r')
    label_index = 1
    mito_prob31 = prob_file31['/volume/prediction'][0,0,:,:,label_index]
    prob_file31.close()
    blur_img31 = scipy.ndimage.gaussian_filter(mito_prob31, 13)
    mito_pred231 = blur_img31<.85
    mito_pred231 = mahotas.erode(mito_pred231, disc)

    
    prob_file32 = h5py.File('Thousands_mito_em_s1181.png_processed.h5', 'r')
    label_index = 1
    mito_prob32 = prob_file32['/volume/prediction'][0,0,:,:,label_index]
    prob_file32.close()
    blur_img32 = scipy.ndimage.gaussian_filter(mito_prob32, 13)
    mito_pred232 = blur_img32<.85
    mito_pred232 = mahotas.erode(mito_pred232, disc)

    
    prob_file33 = h5py.File('Thousands_mito_em_s1182.png_processed.h5', 'r')
    label_index = 1
    mito_prob33 = prob_file33['/volume/prediction'][0,0,:,:,label_index]
    prob_file33.close()
    blur_img33 = scipy.ndimage.gaussian_filter(mito_prob33, 13)
    mito_pred233 = blur_img33<.85
    mito_pred233 = mahotas.erode(mito_pred233, disc)

    
    prob_file34 = h5py.File('Thousands_mito_em_s1183.png_processed.h5', 'r')
    label_index = 1
    mito_prob34 = prob_file34['/volume/prediction'][0,0,:,:,label_index]
    prob_file34.close()
    blur_img34 = scipy.ndimage.gaussian_filter(mito_prob34, 13)
    mito_pred234 = blur_img34<.85
    mito_pred234 = mahotas.erode(mito_pred234, disc)

    
    prob_file35 = h5py.File('Thousands_mito_em_s1184.png_processed.h5', 'r')
    label_index = 1
    mito_prob35 = prob_file35['/volume/prediction'][0,0,:,:,label_index]
    prob_file35.close()
    blur_img35 = scipy.ndimage.gaussian_filter(mito_prob35, 13)
    mito_pred235 = blur_img35<.85
    mito_pred235 = mahotas.erode(mito_pred235, disc)

    
    prob_file36 = h5py.File('Thousands_mito_em_s1185.png_processed.h5', 'r')
    label_index = 1
    mito_prob36 = prob_file36['/volume/prediction'][0,0,:,:,label_index]
    prob_file36.close()
    blur_img36 = scipy.ndimage.gaussian_filter(mito_prob36, 13)
    mito_pred236 = blur_img36<.85
    mito_pred236 = mahotas.erode(mito_pred236, disc)

    
    prob_file37 = h5py.File('Thousands_mito_em_s1186.png_processed.h5', 'r')
    label_index = 1
    mito_prob37 = prob_file37['/volume/prediction'][0,0,:,:,label_index]
    prob_file37.close()
    blur_img37 = scipy.ndimage.gaussian_filter(mito_prob37, 13)
    mito_pred237 = blur_img37<.85
    mito_pred237 = mahotas.erode(mito_pred237, disc)

    
    prob_file38 = h5py.File('Thousands_mito_em_s1187.png_processed.h5', 'r')
    label_index = 1
    mito_prob38 = prob_file38['/volume/prediction'][0,0,:,:,label_index]
    prob_file38.close()
    blur_img38 = scipy.ndimage.gaussian_filter(mito_prob38, 13)
    mito_pred238 = blur_img38<.85
    mito_pred238 = mahotas.erode(mito_pred238, disc)

    
    prob_file39 = h5py.File('Thousands_mito_em_s1188.png_processed.h5', 'r')
    label_index = 1
    mito_prob39 = prob_file39['/volume/prediction'][0,0,:,:,label_index]
    prob_file39.close()
    blur_img39 = scipy.ndimage.gaussian_filter(mito_prob39, 13)
    mito_pred239 = blur_img39<.85
    mito_pred239 = mahotas.erode(mito_pred239, disc)

    
    prob_file40 = h5py.File('Thousands_mito_em_s1189.png_processed.h5', 'r')
    label_index = 1
    mito_prob40 = prob_file40['/volume/prediction'][0,0,:,:,label_index]
    prob_file40.close()
    blur_img40 = scipy.ndimage.gaussian_filter(mito_prob40, 13)
    mito_pred240 = blur_img40<.85
    mito_pred240 = mahotas.erode(mito_pred240, disc)

    
    prob_file41 = h5py.File('Thousands_mito_em_s1190.png_processed.h5', 'r')
    label_index = 1
    mito_prob41 = prob_file41['/volume/prediction'][0,0,:,:,label_index]
    prob_file41.close()
    blur_img41 = scipy.ndimage.gaussian_filter(mito_prob41, 13)
    mito_pred241 = blur_img41<.85
    mito_pred241 = mahotas.erode(mito_pred241, disc)

    
    prob_file42 = h5py.File('Thousands_mito_em_s1191.png_processed.h5', 'r')
    label_index = 1
    mito_prob42 = prob_file42['/volume/prediction'][0,0,:,:,label_index]
    prob_file42.close()
    blur_img42 = scipy.ndimage.gaussian_filter(mito_prob42, 13)
    mito_pred242 = blur_img42<.85
    mito_pred242 = mahotas.erode(mito_pred242, disc)

   
    prob_file43 = h5py.File('Thousands_mito_em_s1192.png_processed.h5', 'r')
    label_index = 1
    mito_prob43 = prob_file43['/volume/prediction'][0,0,:,:,label_index]
    prob_file43.close()
    blur_img43 = scipy.ndimage.gaussian_filter(mito_prob43, 13)
    mito_pred243 = blur_img43<.85
    mito_pred243 = mahotas.erode(mito_pred243, disc)

    
    prob_file44 = h5py.File('Thousands_mito_em_s1193.png_processed.h5', 'r')
    label_index = 1
    mito_prob44 = prob_file44['/volume/prediction'][0,0,:,:,label_index]
    prob_file44.close()
    blur_img44 = scipy.ndimage.gaussian_filter(mito_prob44, 13)
    mito_pred244 = blur_img44<.85
    mito_pred244 = mahotas.erode(mito_pred244, disc)

   
    prob_file45 = h5py.File('Thousands_mito_em_s1194.png_processed.h5', 'r')
    label_index = 1
    mito_prob45 = prob_file45['/volume/prediction'][0,0,:,:,label_index]
    prob_file45.close()
    blur_img45 = scipy.ndimage.gaussian_filter(mito_prob45, 13)
    mito_pred245 = blur_img45<.85
    mito_pred245 = mahotas.erode(mito_pred245, disc)

    
    prob_file46 = h5py.File('Thousands_mito_em_s1195.png_processed.h5', 'r')
    label_index = 1
    mito_prob46 = prob_file46['/volume/prediction'][0,0,:,:,label_index]
    prob_file46.close()
    blur_img46 = scipy.ndimage.gaussian_filter(mito_prob46, 13)
    mito_pred246 = blur_img46<.85
    mito_pred246 = mahotas.erode(mito_pred246, disc)

    
    prob_file47 = h5py.File('Thousands_mito_em_s1196.png_processed.h5', 'r')
    label_index = 1
    mito_prob47 = prob_file47['/volume/prediction'][0,0,:,:,label_index]
    prob_file47.close()
    blur_img47 = scipy.ndimage.gaussian_filter(mito_prob47, 13)
    mito_pred247 = blur_img47<.85
    mito_pred247 = mahotas.erode(mito_pred247, disc)


    prob_file48 = h5py.File('Thousands_mito_em_s1197.png_processed.h5', 'r')
    label_index = 1
    mito_prob48 = prob_file48['/volume/prediction'][0,0,:,:,label_index]
    prob_file48.close()
    blur_img48 = scipy.ndimage.gaussian_filter(mito_prob48, 13)
    mito_pred248 = blur_img48<.85
    mito_pred248 = mahotas.erode(mito_pred248, disc)


    prob_file49 = h5py.File('Thousands_mito_em_s1198.png_processed.h5', 'r')
    label_index = 1
    mito_prob49 = prob_file49['/volume/prediction'][0,0,:,:,label_index]
    prob_file49.close()
    blur_img49 = scipy.ndimage.gaussian_filter(mito_prob49, 13)
    mito_pred249 = blur_img49<.85
    mito_pred249 = mahotas.erode(mito_pred249, disc)


    prob_file50 = h5py.File('Thousands_mito_em_s1199.png_processed.h5', 'r')
    label_index = 1
    mito_prob50 = prob_file50['/volume/prediction'][0,0,:,:,label_index]
    prob_file50.close()
    blur_img50 = scipy.ndimage.gaussian_filter(mito_prob50, 13)
    mito_pred250 = blur_img50<.85
    mito_pred250 = mahotas.erode(mito_pred250, disc)
    
    # Make a 3D rendering of the mitochondria - outputed replaced with Vaa3d's
    
    mlab.clf()
    values = mito_pred2+mito_pred22+mito_pred23+mito_pred24+mito_pred25+mito_pred26+mito_pred27+mito_pred28+mito+pred29+
    mito_pred210+mito_pred211+mito_pred212+mito_pred213+mito_pred214+mito_pred215+mito_pred216+mito_pred217+mito_pred218
    + mito_pred219+mito_pred220+mito_pred221+mito_pred222+mito_pred223+mito_pred224+mito_pred225+mito_pred226+mito_pred227+
    mito_pred228+mito_pred229+mito_pred230+mito_pred231+mito_pred232+mito_pred233+mito_pred234+mito_pred235+mito_pred236+
    mito_pred237+mito_pred238+mito_pred239+mito_pred240+mito_pred241+mito_pred242+mito_pred243+mito_pred244+mito_pred245+
    mito_pred246+mito_pred247+mito_pred248+mito_pred249+mito_pred250
    mlab.contour3d(values)
    mlab.show()
    
  
    # We can also feed these probabilities into the Vaa3D application to generate 3D renderings
    

    
    #
    #
    #
    #
    #
    #
    #
    
    
    # Connected Components Phase - optimization of the mlab method - taken out because better 3D
    # renderings were made with Vaa3d (http://www.vaa3d.org/) using the adjusted
    # segmentations above 
    
    # ##
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
    
    # We feed the mitochondria segmentation results into a Rhoana script for neurite detection, segmentation and
    # 3D reconstruction, but the following ndimage.label approach could also be used to perform this task:
    # Update: OUTDATED SCRIPT - We now just merge the stack of mitochondria labeling results from mito_classify_3D with Rhoana's neurite prediction stack and segment and reconstruct the resulting stack
    #
    #
    
    mito_pred2 = mito_pred2.astype(np.uint8)
    labeled, nr_objects = scipy.ndimage.label(mito_pred2)
    labeled = labeled.astype(np.uint8)
    obj = 0
    for labeled[obj] in labeled:
        robj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].lenth()/2+5)
        if robj == labeled[obj] :
            newobj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2 +5) - 
            labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2) # bordering neurite
            nl, nr_nobjects = scipy.ndimage.label(newobj) # segment the neurite
    
    mito_pred22 = mito_pred22.astype(np.uint8)
    labeled, nr_objects = scipy.ndimage.label(mito_pred22)
    labeled = labeled.astype(np.uint8)
    obj = 0
    for labeled[obj] in labeled:
        robj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].lenth()/2+5)
        if robj == labeled[obj] :
            newobj = labeled2[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled2[obj].length()/2 +5) - 
            labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2) # bordering neurite
            nl2, nr_nobjects2 = scipy.ndimage.label(newobj) # segment the neurite
            
    mito_pred23 = mito_pred23.astype(np.uint8)
    labeled, nr_objects = scipy.ndimage.label(mito_pred23)
    labeled = labeled.astype(np.uint8)
    obj = 0
    for labeled[obj] in labeled:
        robj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].lenth()/2+5)
        if robj == labeled[obj] :
            newobj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2 +5) - 
            labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2) # bordering neurite
            nl3, nr_nobjects3 = scipy.ndimage.label(newobj) # segment the neurite
    
    mito_pred24 = mito_pred24.astype(np.uint8)
    labeled, nr_objects = scipy.ndimage.label(mito_pred24)
    labeled = labeled.astype(np.uint8)
    obj = 0
    for labeled[obj] in labeled:
        robj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].lenth()/2+5)
        if robj == labeled[obj] :
            newobj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2 +5) - 
            labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2) # bordering neurite
            nl4, nr_nobjects4 = scipy.ndimage.label(newobj) # segment the neurite
            
    mito_pred25 = mito_pred25.astype(np.uint8)
    labeled, nr_objects = scipy.ndimage.label(mito_pred25)
    labeled = labeled.astype(np.uint8)
    obj = 0
    for labeled[obj] in labeled:
        robj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].lenth()/2+5)
        if robj == labeled[obj] :
            newobj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2 +5) - 
            labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2) # bordering neurite
            nl5, nr_nobjects5 = scipy.ndimage.label(newobj) # segment the neurite
            
    mito_pred26 = mito_pred26.astype(np.uint8)
    labeled, nr_objects = scipy.ndimage.label(mito_pred26)
    labeled = labeled.astype(np.uint8)
    obj = 0
    for labeled[obj] in labeled:
        robj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].lenth()/2+5)
        if robj == labeled[obj] :
            newobj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2 +5) - 
            labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2) # bordering neurite
            nl6, nr_nobjects6 = scipy.ndimage.label(newobj) # segment the neurite
            
    mito_pred27 = mito_pred27.astype(np.uint8)
    labeled, nr_objects = scipy.ndimage.label(mito_pred27)
    labeled = labeled.astype(np.uint8)
    obj = 0
    for labeled[obj] in labeled:
        robj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].lenth()/2+5)
        if robj == labeled[obj] :
            newobj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2 +5) - 
            labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2) # bordering neurite
            nl7, nr_nobjects7 = scipy.ndimage.label(newobj) # segment the neurite
            
    mito_pred28 = mito_pred28.astype(np.uint8)
    labeled, nr_objects = scipy.ndimage.label(mito_pred28)
    labeled = labeled.astype(np.uint8)
    obj = 0
    for labeled[obj] in labeled:
        robj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].lenth()/2+5)
        if robj == labeled[obj] :
            newobj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2 +5) - 
            labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2) # bordering neurite
            nl8, nr_nobjects8 = scipy.ndimage.label(newobj) # segment the neurite
            
    mito_pred29 = mito_pred29.astype(np.uint8)
    labeled, nr_objects = scipy.ndimage.label(mito_pred29)
    labeled = labeled.astype(np.uint8)
    obj = 0
    for labeled[obj] in labeled:
        robj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].lenth()/2+5)
        if robj == labeled[obj] :
            newobj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2 +5) - 
            labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2) # bordering neurite
            nl9, nr_nobjects9 = scipy.ndimage.label(newobj) # segment the neurite
            
    mito_pred210 = mito_pred210.astype(np.uint8)
    labeled, nr_objects = scipy.ndimage.label(mito_pred210)
    labeled = labeled.astype(np.uint8)
    obj = 0
    for labeled[obj] in labeled:
        robj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].lenth()/2+5)
        if robj == labeled[obj] :
            newobj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2 +5) - 
            labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2) # bordering neurite
            nl10, nr_nobjects10 = scipy.ndimage.label(newobj) # segment the neurite
            
    mito_pred211 = mito_pred211.astype(np.uint8)
    labeled, nr_objects = scipy.ndimage.label(mito_pred211)
    labeled = labeled.astype(np.uint8)
    obj = 0
    for labeled[obj] in labeled:
        robj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].lenth()/2+5)
        if robj == labeled[obj] :
            newobj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2 +5) - 
            labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2) # bordering neurite
            nl1, nr_nobjects11 = scipy.ndimage.label(newobj) # segment the neurite
            
    mito_pred212 = mito_pred212.astype(np.uint8)
    labeled, nr_objects = scipy.ndimage.label(mito_pred212)
    labeled = labeled.astype(np.uint8)
    obj = 0
    for labeled[obj] in labeled:
        robj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].lenth()/2+5)
        if robj == labeled[obj] :
            newobj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2 +5) - 
            labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2) # bordering neurite
            nl12, nr_nobjects12 = scipy.ndimage.label(newobj) # segment the neurite
            
    mito_pred213 = mito_pred213.astype(np.uint8)
    labeled, nr_objects = scipy.ndimage.label(mito_pred213)
    labeled = labeled.astype(np.uint8)
    obj = 0
    for labeled[obj] in labeled:
        robj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].lenth()/2+5)
        if robj == labeled[obj] :
            newobj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2 +5) - 
            labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2) # bordering neurite
            nl13, nr_nobjects13 = scipy.ndimage.label(newobj) # segment the neurite
            
    mito_pred214 = mito_pred214.astype(np.uint8)
    labeled, nr_objects = scipy.ndimage.label(mito_pred214)
    labeled = labeled.astype(np.uint8)
    obj = 0
    for labeled[obj] in labeled:
        robj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].lenth()/2+5)
        if robj == labeled[obj] :
            newobj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2 +5) - 
            labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2) # bordering neurite
            nl14, nr_nobjects14 = scipy.ndimage.label(newobj) # segment the neurite
            
    mito_pred215 = mito_pred215.astype(np.uint8)
    labeled, nr_objects = scipy.ndimage.label(mito_pred215)
    labeled = labeled.astype(np.uint8)
    obj = 0
    for labeled[obj] in labeled:
        robj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].lenth()/2+5)
        if robj == labeled[obj] :
            newobj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2 +5) - 
            labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2) # bordering neurite
            nl15, nr_nobjects15 = scipy.ndimage.label(newobj) # segment the neurite
            
    mito_pred216 = mito_pred216.astype(np.uint8)
    labeled, nr_objects = scipy.ndimage.label(mito_pred216)
    labeled = labeled.astype(np.uint8)
    obj = 0
    for labeled[obj] in labeled:
        robj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].lenth()/2+5)
        if robj == labeled[obj] :
            newobj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2 +5) - 
            labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2) # bordering neurite
            nl16, nr_nobjects16 = scipy.ndimage.label(newobj) # segment the neurite
            
    mito_pred217 = mito_pred217.astype(np.uint8)
    labeled, nr_objects = scipy.ndimage.label(mito_pred217)
    labeled = labeled.astype(np.uint8)
    obj = 0
    for labeled[obj] in labeled:
        robj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].lenth()/2+5)
        if robj == labeled[obj] :
            newobj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2 +5) - 
            labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2) # bordering neurite
            nl17, nr_nobjects17 = scipy.ndimage.label(newobj) # segment the neurite
            
    mito_pred218 = mito_pred218.astype(np.uint8)
    labeled, nr_objects = scipy.ndimage.label(mito_pred218)
    labeled = labeled.astype(np.uint8)
    obj = 0
    for labeled[obj] in labeled:
        robj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].lenth()/2+5)
        if robj == labeled[obj] :
            newobj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2 +5) - 
            labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2) # bordering neurite
            nl18, nr_nobjects18 = scipy.ndimage.label(newobj) # segment the neurite
            
    mito_pred219 = mito_pred219.astype(np.uint8)
    labeled, nr_objects = scipy.ndimage.label(mito_pred219)
    labeled = labeled.astype(np.uint8)
    obj = 0
    for labeled[obj] in labeled:
        robj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].lenth()/2+5)
        if robj == labeled[obj] :
            newobj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2 +5) - 
            labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2) # bordering neurite
            nl19, nr_nobjects19 = scipy.ndimage.label(newobj) # segment the neurite
            
    mito_pred220 = mito_pred220.astype(np.uint8)
    labeled, nr_objects = scipy.ndimage.label(mito_pred220)
    labeled = labeled.astype(np.uint8)
    obj = 0
    for labeled[obj] in labeled:
        robj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].lenth()/2+5)
        if robj == labeled[obj] :
            newobj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2 +5) - 
            labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2) # bordering neurite
            nl20, nr_nobjects20 = scipy.ndimage.label(newobj) # segment the neurite
            
    mito_pred221 = mito_pred221.astype(np.uint8)
    labeled, nr_objects = scipy.ndimage.label(mito_pred221)
    labeled = labeled.astype(np.uint8)
    obj = 0
    for labeled[obj] in labeled:
        robj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].lenth()/2+5)
        if robj == labeled[obj] :
            newobj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2 +5) - 
            labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2) # bordering neurite
            nl21, nr_nobjects21 = scipy.ndimage.label(newobj) # segment the neurite
            
    mito_pred222 = mito_pred222.astype(np.uint8)
    labeled, nr_objects = scipy.ndimage.label(mito_pred222)
    labeled = labeled.astype(np.uint8)
    obj = 0
    for labeled[obj] in labeled:
        robj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].lenth()/2+5)
        if robj == labeled[obj] :
            newobj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2 +5) - 
            labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2) # bordering neurite
            nl22, nr_nobjects22 = scipy.ndimage.label(newobj) # segment the neurite
            
    mito_pred223 = mito_pred223.astype(np.uint8)
    labeled, nr_objects = scipy.ndimage.label(mito_pred223)
    labeled = labeled.astype(np.uint8)
    obj = 0
    for labeled[obj] in labeled:
        robj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].lenth()/2+5)
        if robj == labeled[obj] :
            newobj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2 +5) - 
            labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2) # bordering neurite
            nl23, nr_nobjects23 = scipy.ndimage.label(newobj) # segment the neurite
            
    mito_pred224 = mito_pred224.astype(np.uint8)
    labeled, nr_objects = scipy.ndimage.label(mito_pred224)
    labeled = labeled.astype(np.uint8)
    obj = 0
    for labeled[obj] in labeled:
        robj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].lenth()/2+5)
        if robj == labeled[obj] :
            newobj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2 +5) - 
            labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2) # bordering neurite
            nl24, nr_nobjects24 = scipy.ndimage.label(newobj) # segment the neurite
    
    mito_pred225 = mito_pred225.astype(np.uint8)
    labeled, nr_objects = scipy.ndimage.label(mito_pred225)
    labeled = labeled.astype(np.uint8)
    obj = 0
    for labeled[obj] in labeled:
        robj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].lenth()/2+5)
        if robj == labeled[obj] :
            newobj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2 +5) - 
            labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2) # bordering neurite
            nl25, nr_nobjects25 = scipy.ndimage.label(newobj) # segment the neurite
            
    mito_pred226 = mito_pred226.astype(np.uint8)
    labeled, nr_objects = scipy.ndimage.label(mito_pred226)
    labeled = labeled.astype(np.uint8)
    obj = 0
    for labeled[obj] in labeled:
        robj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].lenth()/2+5)
        if robj == labeled[obj] :
            newobj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2 +5) - 
            labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2) # bordering neurite
            nl26, nr_nobjects26 = scipy.ndimage.label(newobj) # segment the neurite
            
    mito_pred227 = mito_pred227.astype(np.uint8)
    labeled, nr_objects = scipy.ndimage.label(mito_pred227)
    labeled = labeled.astype(np.uint8)
    obj = 0
    for labeled[obj] in labeled:
        robj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].lenth()/2+5)
        if robj == labeled[obj] :
            newobj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2 +5) - 
            labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2) # bordering neurite
            nl27, nr_nobjects27 = scipy.ndimage.label(newobj) # segment the neurite
            
    mito_pred228 = mito_pred228.astype(np.uint8)
    labeled, nr_objects = scipy.ndimage.label(mito_pred228)
    labeled = labeled.astype(np.uint8)
    obj = 0
    for labeled[obj] in labeled:
        robj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].lenth()/2+5)
        if robj == labeled[obj] :
            newobj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2 +5) - 
            labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2) # bordering neurite
            nl28, nr_nobjects28 = scipy.ndimage.label(newobj) # segment the neurite
            
    mito_pred229 = mito_pred229.astype(np.uint8)
    labeled, nr_objects = scipy.ndimage.label(mito_pred229)
    labeled = labeled.astype(np.uint8)
    obj = 0
    for labeled[obj] in labeled:
        robj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].lenth()/2+5)
        if robj == labeled[obj] :
            newobj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2 +5) - 
            labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2) # bordering neurite
            nl29, nr_nobjects29 = scipy.ndimage.label(newobj) # segment the neurite
            
    mito_pred230 = mito_pred230.astype(np.uint8)
    labeled, nr_objects = scipy.ndimage.label(mito_pred230)
    labeled = labeled.astype(np.uint8)
    obj = 0
    for labeled[obj] in labeled:
        robj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].lenth()/2+5)
        if robj == labeled[obj] :
            newobj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2 +5) - 
            labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2) # bordering neurite
            nl30, nr_nobjects30 = scipy.ndimage.label(newobj) # segment the neurite
            
    mito_pred231 = mito_pred231.astype(np.uint8)
    labeled, nr_objects = scipy.ndimage.label(mito_pred231)
    labeled = labeled.astype(np.uint8)
    obj = 0
    for labeled[obj] in labeled:
        robj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].lenth()/2+5)
        if robj == labeled[obj] :
            newobj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2 +5) - 
            labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2) # bordering neurite
            nl31, nr_nobjects31 = scipy.ndimage.label(newobj) # segment the neurite
            
    mito_pred232 = mito_pred232.astype(np.uint8)
    labeled, nr_objects = scipy.ndimage.label(mito_pred232)
    labeled = labeled.astype(np.uint8)
    obj = 0
    for labeled[obj] in labeled:
        robj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].lenth()/2+5)
        if robj == labeled[obj] :
            newobj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2 +5) - 
            labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2) # bordering neurite
            nl32, nr_nobjects32 = scipy.ndimage.label(newobj) # segment the neurite
            
    mito_pred233 = mito_pred233.astype(np.uint8)
    labeled, nr_objects = scipy.ndimage.label(mito_pred233)
    labeled = labeled.astype(np.uint8)
    obj = 0
    for labeled[obj] in labeled:
        robj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].lenth()/2+5)
        if robj == labeled[obj] :
            newobj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2 +5) - 
            labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2) # bordering neurite
            nl33, nr_nobjects33 = scipy.ndimage.label(newobj) # segment the neurite
            
    mito_pred234 = mito_pred234.astype(np.uint8)
    labeled, nr_objects = scipy.ndimage.label(mito_pred234)
    labeled = labeled.astype(np.uint8)
    obj = 0
    for labeled[obj] in labeled:
        robj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].lenth()/2+5)
        if robj == labeled[obj] :
            newobj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2 +5) - 
            labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2) # bordering neurite
            nl34, nr_nobjects34 = scipy.ndimage.label(newobj) # segment the neurite

    mito_pred235 = mito_pred235.astype(np.uint8)
    labeled, nr_objects = scipy.ndimage.label(mito_pred235)
    labeled = labeled.astype(np.uint8)
    obj = 0
    for labeled[obj] in labeled:
        robj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].lenth()/2+5)
        if robj == labeled[obj] :
            newobj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2 +5) - 
            labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2) # bordering neurite
            nl35, nr_nobjects35 = scipy.ndimage.label(newobj) # segment the neurite
            
    mito_pred236 = mito_pred236.astype(np.uint8)
    labeled, nr_objects = scipy.ndimage.label(mito_pred236)
    labeled = labeled.astype(np.uint8)
    obj = 0
    for labeled[obj] in labeled:
        robj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].lenth()/2+5)
        if robj == labeled[obj] :
            newobj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2 +5) - 
            labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2) # bordering neurite
            nl36, nr_nobjects36 = scipy.ndimage.label(newobj) # segment the neurite
    
    mito_pred237 = mito_pred237.astype(np.uint8)
    labeled, nr_objects = scipy.ndimage.label(mito_pred237)
    labeled = labeled.astype(np.uint8)
    obj = 0
    for labeled[obj] in labeled:
        robj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].lenth()/2+5)
        if robj == labeled[obj] :
            newobj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2 +5) - 
            labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2) # bordering neurite
            nl37, nr_nobjects37 = scipy.ndimage.label(newobj) # segment the neurite
            
    mito_pred238 = mito_pred238.astype(np.uint8)
    labeled, nr_objects = scipy.ndimage.label(mito_pred238)
    labeled = labeled.astype(np.uint8)
    obj = 0
    for labeled[obj] in labeled:
        robj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].lenth()/2+5)
        if robj == labeled[obj] :
            newobj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2 +5) - 
            labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2) # bordering neurite
            nl38, nr_nobjects38 = scipy.ndimage.label(newobj) # segment the neurite
            
    mito_pred239 = mito_pred239.astype(np.uint8)
    labeled, nr_objects = scipy.ndimage.label(mito_pred239)
    labeled = labeled.astype(np.uint8)
    obj = 0
    for labeled[obj] in labeled:
        robj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].lenth()/2+5)
        if robj == labeled[obj] :
            newobj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2 +5) - 
            labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2) # bordering neurite
            nl39, nr_nobjects39 = scipy.ndimage.label(newobj) # segment the neurite
            
    mito_pred240 = mito_pred240.astype(np.uint8)
    labeled, nr_objects = scipy.ndimage.label(mito_pred240)
    labeled = labeled.astype(np.uint8)
    obj = 0
    for labeled[obj] in labeled:
        robj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].lenth()/2+5)
        if robj == labeled[obj] :
            newobj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2 +5) - 
            labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2) # bordering neurite
            nl40, nr_nobjects40 = scipy.ndimage.label(newobj) # segment the neurite
            
    mito_pred241 = mito_pred241.astype(np.uint8)
    labeled, nr_objects = scipy.ndimage.label(mito_pred241)
    labeled = labeled.astype(np.uint8)
    obj = 0
    for labeled[obj] in labeled:
        robj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].lenth()/2+5)
        if robj == labeled[obj] :
            newobj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2 +5) - 
            labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2) # bordering neurite
            nl41, nr_nobjects41 = scipy.ndimage.label(newobj) # segment the neurite
            
    mito_pred242 = mito_pred242.astype(np.uint8)
    labeled, nr_objects = scipy.ndimage.label(mito_pred242)
    labeled = labeled.astype(np.uint8)
    obj = 0
    for labeled[obj] in labeled:
        robj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].lenth()/2+5)
        if robj == labeled[obj] :
            newobj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2 +5) - 
            labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2) # bordering neurite
            nl42, nr_nobjects42 = scipy.ndimage.label(newobj) # segment the neurite
            
    mito_pred243 = mito_pred243.astype(np.uint8)
    labeled, nr_objects = scipy.ndimage.label(mito_pred243)
    labeled = labeled.astype(np.uint8)
    obj = 0
    for labeled[obj] in labeled:
        robj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].lenth()/2+5)
        if robj == labeled[obj] :
            newobj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2 +5) - 
            labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2) # bordering neurite
            nl43, nr_nobjects43 = scipy.ndimage.label(newobj) # segment the neurite
            
    mito_pred244 = mito_pred244.astype(np.uint8)
    labeled, nr_objects = scipy.ndimage.label(mito_pred244)
    labeled = labeled.astype(np.uint8)
    obj = 0
    for labeled[obj] in labeled:
        robj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].lenth()/2+5)
        if robj == labeled[obj] :
            newobj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2 +5) - 
            labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2) # bordering neurite
            nl44, nr_nobjects44 = scipy.ndimage.label(newobj) # segment the neurite
            
    mito_pred245 = mito_pred245.astype(np.uint8)
    labeled, nr_objects = scipy.ndimage.label(mito_pred245)
    labeled = labeled.astype(np.uint8)
    obj = 0
    for labeled[obj] in labeled:
        robj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].lenth()/2+5)
        if robj == labeled[obj] :
            newobj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2 +5) - 
            labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2) # bordering neurite
            nl45, nr_nobjects45 = scipy.ndimage.label(newobj) # segment the neurite
            
    mito_pred246 = mito_pred246.astype(np.uint8)
    labeled, nr_objects = scipy.ndimage.label(mito_pred246)
    labeled = labeled.astype(np.uint8)
    obj = 0
    for labeled[obj] in labeled:
        robj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].lenth()/2+5)
        if robj == labeled[obj] :
            newobj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2 +5) - 
            labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2) # bordering neurite
            nl46, nr_nobjects46 = scipy.ndimage.label(newobj) # segment the neurite
            
    mito_pred247 = mito_pred247.astype(np.uint8)
    labeled, nr_objects = scipy.ndimage.label(mito_pred247)
    labeled = labeled.astype(np.uint8)
    obj = 0
    for labeled[obj] in labeled:
        robj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].lenth()/2+5)
        if robj == labeled[obj] :
            newobj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2 +5) - 
            labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2) # bordering neurite
            nl47, nr_nobjects47 = scipy.ndimage.label(newobj) # segment the neurite
            
    mito_pred248 = mito_pred248.astype(np.uint8)
    labeled, nr_objects = scipy.ndimage.label(mito_pred248)
    labeled = labeled.astype(np.uint8)
    obj = 0
    for labeled[obj] in labeled:
        robj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].lenth()/2+5)
        if robj == labeled[obj] :
            newobj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2 +5) - 
            labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2) # bordering neurite
            nl48, nr_nobjects48 = scipy.ndimage.label(newobj) # segment the neurite
    
    mito_pred249 = mito_pred249.astype(np.uint8)
    labeled, nr_objects = scipy.ndimage.label(mito_pred249)
    labeled = labeled.astype(np.uint8)
    obj = 0
    for labeled[obj] in labeled:
        robj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].lenth()/2+5)
        if robj == labeled[obj] :
            newobj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2 +5) - 
            labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2) # bordering neurite
            nl49, nr_nobjects49 = scipy.ndimage.label(newobj) # segment the neurite
            
    mito_pred250 = mito_pred250.astype(np.uint8)
    labeled, nr_objects = scipy.ndimage.label(mito_pred250)
    labeled = labeled.astype(np.uint8)
    obj = 0
    for labeled[obj] in labeled:
        robj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].lenth()/2+5)
        if robj == labeled[obj] :
            newobj = labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2 +5) - 
            labeled[obj].scipy.ndimage.at(labeled[obj].ypos()+labeled[obj].length()/2) # bordering neurite
            nl50, nr_nobjects50 = scipy.ndimage.label(newobj) # segment the neurite
            
    
    
    mlab.clf()
    values = nl+nl2+nl3+nl4+nl5+nl6+nl7+nl8+nl9+nl10+nl11+nl12+nl13+nl14+nl15+nl16+nl17+nl18+nl19+nl20+nl21+nl22+
    nl23+nl24+nl25+nl26+nl27+nl28+nl29+nl30+nl31+nl32+nl33+nl34+nl35+nl36+nl37+nl38+nl39+nl40+nl41+nl42+nl43+nl44+nl45
    +nl46+nl47+nl48+nl49+nl50
    mlab.contour3d(values) # 3D reconstruction of the neurites, using all segmentations of the neurites
    mlab.show()
    

    



    
    
    
    
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
    # 
    # Updating and optimization of the code is underway
    # From here we post-process the segmentation with Rhoana's original results and measure our new results against 
    # Rhoana's original results to gage improvement in segmentation
    #################################
    
   

    
 print 'Done.'
