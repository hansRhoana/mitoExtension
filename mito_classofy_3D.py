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
    mito_pred25 = mahotas.erode(mito_pred25, disc)

    
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
