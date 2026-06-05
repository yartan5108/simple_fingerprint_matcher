
import os, sys
import cv2
from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
import time

for k, v in os.environ.items():
    if k.startswith("QT_") and "cv2" in v:
        del os.environ[k]

global similarity_thresh
similarity_thresh = 0.05

# Radius of circle
radius = 20
# Blue color in BGR
color = (255, 0, 0)
# Line thickness of 2 px
thickness = 2
# Using cv2.circle() method
# Draw a circle with blue line borders of thickness of 2 px

global param
param = {}
# param['muP']  = 20
# param['taoP']  = 2 / 5
param['minNp']  = 4
param['maxNp']  = 15  # 12
# param['mu1P'] = 5
# param['tao1P'] = -8 / 5
# param['mu2P'] = np.pi / 12
# param['tao2P'] = -30
# param['mu3P'] = np.pi / 12
# param['tao3P'] = -30

param['muP']  = 32
param['taoP']  = 0.25
param['mu1P'] = 0.041666666666666664
param['tao1P'] = -50 * 0.6
param['mu2P'] = 0.78539816339744828
param['tao2P'] =  -15 * 0.6
param['mu3P'] = 0.20943951023931953
param['tao3P'] = -28 * 0.6

# path = '/home/yartan/Documents/AFIS_2/server/AFIS/src/comparison3/FMLib/cmake-build-debug/'
#
# # filename = 'log_original_5566_one_Thread8_fmtype3.log'
# # path_embedding = '/home/yartan/PycharmProjects/Patch_Matcher/5566_one_embedding/'
# # path_xyt = '/media/yartan/DISK1/Sensor_FingerNet_Final/egm_img/egmlatent_5566_one/'
#
# filename = 'log_original_egmlatent_2000_images_Thread8_fmtype3.log'
# path_xyt = '/media/yartan/DISK1/Sensor_FingerNet_Final/egm_img/ekim_2021_one/'
# path_embedding = '/home/yartan/PycharmProjects/Patch_Matcher/2000_embedding/'

work_on = 1

if(work_on):
    path = '/home/yartan/Documents/AFIS_2/server/AFIS/src/comparison2/FMLib/cmake-build-debug/'
    filename =  'log_original_egmlatent_some_images_Thread8_fmtype3.log'
    path_xyt = '/media/yartan/DISK1/Sensor_FingerNet_Final/egm_img/some_images/'
    path_embedding = '/home/yartan/PycharmProjects/Patch_Matcher/5566_one_embedding/'
else:
    path = os.getcwd()
    filename = 'log_original_egmlatent_some_images_Thread8_fmtype3.log'
    path_xyt = 'C:/Users/asus/PycharmProjects/CylinderCommentli/some_images/'
    path_embedding = 'C:/Users/asus/PycharmProjects/CylinderCommentli/some_images_embedding/'

def get_files_2_compare(path_txt, filename):
    fid = open( os.path.join(path_txt,filename), 'r')

    flag=True
    image_files={}
    counter = 0
    read_flag = False

    while(flag):

        line=fid.readline()[:-1].split()
        print(line)

        if line.__len__()>1:

            if( line[0]=='Progress' ):
                read_flag = True
                continue

            if( line[0]=='Table'):
                break

            if(read_flag):
                line_matcher = line[-1]

                RankVal = int( line[3] )
                skore = float( line[4] )

                if( RankVal != -1 ):

                    file1, file2 = line[11], line[12]

                    # inds = []
                    # m_locs = line_matcher.split('[')[13][:-2].split(',')
                    # for vals in range(0, m_locs.__len__(), 3):
                    #     ind1, ind2 = int( m_locs[vals] ), int( m_locs[vals+1] )
                    #     inds.append( (ind1, ind2) )

                    image_files[counter] = [ file1, file2, RankVal, skore, ]
                    counter = counter + 1
                    print(counter)

    return image_files

# filename dosyasında bulunan minutia bilgileri okunuyor.
# (minutia_x, minutia_y, angle) bilgileri okunuyor. xyt dosyaları zaten kaliteye göre sıralandığı için onları almadım.
def xyt_file_reader(path_txt, filename):
    fid = open( os.path.join(path_txt,filename), 'r')
    flag=True
    coords=[]
    while(flag):
        line=fid.readline()[:-1].split()
        if( line==[] ):
            flag=False
        if (line.__len__() > 1):
            coords.append( [ int(line[0]),int(line[1]),int(line[2]) ] )
    return coords

# cyb file reader ile filename dosyasında bulunan silindir bilgleri okunmaktadır.
# cyb dosyasından örnek bir kesit aşağıda sunulmaktadır. Burada ikinci index, 4, xyt dosyasındaki
# belirtilen minutiaya karşılık geldiğini göstermektedir. İlk satırdan sonra belirtilen 8 adet satırda
# 64 bitlik silindir ve validity bilgileri bulunmaktadır.
# İlk 4 adet 64'lik veriden (1x256)'lık silindir vektörüne karşılık gelmektedir.
# Sonraki 4 adet 64'lik veriden (1x256)'lık validity vektörü oluşturulmakatdır.
# Bunlar aşagıdaki ana fonksiyonda similarity değerlerini hesaplarken kullanılmaktadır.
# 0 4 325 158 251 127 277 99 -1
# 0000000000000000000000000000000000000000011001100000000000000000
# 000000000000000000000000000    # 0010100105638033\17718826.s
# 000000000000000000000000000000000000000610320028190111
# 0000000010001000101110110000000000000000110111010000000000000000
# 0000000000000000000000000000000000001100110000000000000000110000
# 0000111111111111111111111111000000000000111111111111111100000000
# 1111111111111111111111111111111111111111111111111111111111111111
# 1111111111111111111111111111111111111111111111111111111111111111
# 00000000111111111111111100000000# Eq. 24 de belirtilen (nP) değeri burada te        # cv.circle(im_sensor, tuple(i for i in coords_sensor), 10, (255, 0, 0), 3)

def cyb_file_reader(path_txt, filename):
    fid = open( os.path.join(path_txt,filename), 'r')
    flag=True
    index, csMat, validMat = [], [], []

    while(flag):
        line=fid.readline().split(' ')
        if (line[0]==""):
            break
        line_csMat = fid.readline()[:64] + fid.readline()[:64] + fid.readline()[:64] + fid.readline()[:64]
        line_validMat = fid.readline()[:64] + fid.readline()[:64] + fid.readline()[:64] + fid.readline()[:64]

        index.append( int(line[1]) )
        csMat.append( line_csMat )
        validMat.append( line_validMat )

    return index, csMat, validMat


def embedding_file_reader(path_txt, filename):
    fid = open( os.path.join(path_txt,filename), 'r')
    flag=True
    embedding_file = []
    while(flag):
        line = fid.readline()
        line = line.split(' ')
        if (line[0]==""):
            break
        line_float = []
        for iter in range(256):
            line_float.append( float(line[iter]) )
        embedding_file.append( line_float )
    return np.asarray(embedding_file), embedding_file.__len__()

def cvec_ab( a, a_2, b, b_2, Ns, Nd ):
    Nc = Ns * Ns * Nd
    c_vec_ab = np.zeros( Nc )
    c_vec_ba = np.zeros( Nc )

    indis_overlap = np.where( np.multiply( np.array(a_2), np.array(b_2) )> 0 )
    c_vec_ab[indis_overlap] = a[indis_overlap]
    c_vec_ba[indis_overlap] = b[indis_overlap]

    return c_vec_ab, c_vec_ba

# Eq. 17 'de berlitilen silindir_a ve silindir_b için benzerlik hesaplaması gerçekleştiriliyor.
def pairwise_similarity( cy1, counter_cy1, cy2, counter_cy2 ):
    global param    # 0010100105638033\17718826.s

    score_mat = np.zeros( (counter_cy1, counter_cy2) )
    for iter_m1 in np.arange( 0, counter_cy1 ):
        for iter_m2 in np.arange( 0, counter_cy2 ):
            a, a_2 = cy1[(iter_m1, 0)], cy1[(iter_m1, 1)]
            b, b_2 = cy2[(iter_m2, 0)], cy2[(iter_m2, 1)]
            c_vec_ab, c_vec_ba = cvec_ab( a, a_2, b, b_2, param['Ns'], param['Nd'] )
            tmp = (c_vec_ab - c_vec_ba)
            tmp2 = np.linalg.norm(tmp) / ( np.linalg.norm(c_vec_ab) + np.linalg.norm(c_vec_ba) )
            score_mat[iter_m1, iter_m2] = 1 - tmp2
    return score_mat

# Embedding vectoru için Eq. 17 'de berlitilen silindir_a ve silindir_b için benzerlik hesaplaması gerçekleştiriliyor.
def pairwise_similarity_embedding( cy1, xyt1, cy2, xyt2 ):
    global param
    score_mat = np.zeros( (xyt1.shape[0], xyt2.shape[0]) )
    for iter_m1 in np.arange( 0, xyt1.shape[0] ):
        for iter_m2 in np.arange( 0, xyt2.shape[0] ):
            if( abs( directional_difference( xyt1[iter_m1][2] , xyt2[iter_m2][2], 0 ) ) <= 80 ):
                a = cy1[iter_m1]
                b = cy2[iter_m2]
                tmp = (a - b)
                tmp2 = np.linalg.norm(tmp) / ( np.linalg.norm(a) + np.linalg.norm(b) )
                score_mat[iter_m1, iter_m2] = 1 - tmp2
    return score_mat


def SimilarityScore( score, numSeeds ):
    indis_pair = []
    score_vals = []
    global similarity_thresh
    score[np.isnan(score)] = 0

    for iter in np.arange(1,numSeeds):
       max_elem = np.max( score[:] )
       if max_elem>similarity_thresh:
           [row, col] = np.where( score == max_elem )
           score[ :, col[0] ] = 0
           score[ row[0], : ] = 0
           indis_pair.append( [ row[0], col[0] ] )
           score_vals.append( max_elem )
       else:
           continue
    return indis_pair, score_vals

def compute_MinNo( Na, Nb ):
    global param
    min_Nab = np.min( (Na, Nb) )
    out = 1./( 1 + np.exp( -1*param['muP']*( min_Nab - param['taoP'] ) ) )
    difference = ( param['maxNp'] - param['minNp'] )
    out = param['minNp'] + round( out*difference )
    return out

def directional_difference( term1, term2, deg_flag_on):

    if deg_flag_on:
        term1 = term1*(180/np.iterpi)
        term2 = term2*(180/np.pi)

    if abs(term1 - term2) < 180:
        d_phi = term1 - term2
    elif (term1 - term2) <= -180:
        d_phi = 360 + (term1 - term2)
    elif (term1 - term2) >= 180:
        d_phi = -360 + (term1 - term2)
    return d_phi

# LSS_Relaxation adımında (Eq.25) kullanılan Z(.) fonksyonu burada uygulanmaktadır.
def compute_LLS_Phi( d1, d2, d3 ):
    global param
    out1 = 1./(1+np.exp(-1*param['tao1P'] *( d1 - param['mu1P'] )))
    out2 = 1./(1+np.exp(-1*param['tao2P'] *( d2 - param['mu2P'] )))
    out3 = 1./(1+np.exp(-1*param['tao3P'] *( d3 - param['mu3P'] )))
    out = out1*out2*out3
    return out

# Eq. 25-28 'e kadar olan denklemler LSS Relaxation adımıdır. Yüksek skorlar        # cv.circle(im_sensor, tuple(i for i in coords_sensor), 10, (255, 0, 0), 3)
# birbirlerine olan uzamsal veiter açısal bilgileri kullanılarak skor değerleri refine ediliyor.
def LSS_relaxation( cylinder_m1, cylinder_m2, indis_pair, numSeeds, score_vals, out ):

    template_A = []
    template_B = []
    numSeeds = min(numSeeds,score_vals.__len__())
    for iterator in np.arange(0,numSeeds):
        tmp = cylinder_m1[ (indis_pair[iterator][0], 2) ]
        template_A.append( tmp )
        tmp = cylinder_m2[ (indis_pair[iterator][1], 2) ]
        template_B.append( tmp )

    # Eq. 26 'da belirtilen p_tk matrisini oluşturuyoruz.
    # p_tk matrisinin içindeki herbir eşeleşn minutia pair için
    # Eq. 26 da belirtilen d1 (spatial terim), d2(angular terim) ve d3(radial terim) değerleri hesaplanıyor.
    p_tk = np.zeros( ( template_A.__len__(), template_A.__len__() ) )

    for iterx in np.arange( 0, template_A.__len__() ):
        for itery in np.arange( 0, template_A.__len__() ):
            if iterx != itery:
                tmp_t1 = template_A[iterx]
                tmp_t2 = template_A[itery]

                tmp_k1 = template_B[iterx]
                tmp_k2 = template_B[itery]

                # Spatial Term (d1)..
                ds_Atk = cdist( [np.array(tmp_t1[0:2])], [np.array(tmp_t2[0:2])] )
                ds_Btk = cdist( [np.array(tmp_k1[0:2])], [np.array(tmp_k2[0:2])] )
                d1_term = np.abs(ds_Atk - ds_Btk)

                # Angular Term (d2)...
                d_theta_A = directional_difference( tmp_t1[2], tmp_t2[2], 1 )
                d_theta_B = directional_difference( tmp_k1[2], tmp_k2[2], 1 )
                d2_term = np.abs(directional_difference(d_theta_A, d_theta_B, 0))

                # Radial Term (d3)...
                theta_m1 = tmp_t1[2]
                dR_m1 = np.arctan2( tmp_t1[0] - tmp_t2[0], tmp_t2[1] - tmp_t1[1] )
                dR_m1_term = directional_difference(theta_m1, dR_m1, 1)
                theta_m2 = tmp_k1[2]
                dR_m2 = np.arctan2( tmp_k1[0] - tmp_k2[0], tmp_k2[1] - tmp_k1[1] )
                dR_m2_term = directional_difference(theta_m2, dR_m2, 1)
                d3_term = abs(directional_difference(dR_m1_term, dR_m2_term, 0))

                # d1, d2, ve d3 terimleri kullanılarak Eq. 26 daki Z(.) fonskiyonu hesaplanıyor.
                p_tk[iterx, itery] = compute_LLS_Phi( d1_term, np.pi * (d2_term / 180), np.pi * (d3_term / 180) )


    # p_tk matrisi shesaplandıktan sonra Eq.(25) de belirtilen iterasyon burada başlıyor. Aşagıda berlitilen Lamba_vec terimi
    # Eq. 25 deki lambda y akaşılık gelmektedir.
    w_R = 0.5
    Lambda_vec = np.array( score_vals )
    Lambda_vec_original = np.array( score_vals )
    Number_iter = 2 # Eq. 25-28 'e kadar olan denklemler LSS Relaxation adımıdır. Yüksek skorlar ile elde edilen pair'ların
    # birbirlerine olan uzamsal ve açısal bilgileri kullanılarak skor değerleri refine ediliyor
    for out_iter in np.arange(0, Number_iter):
       for iterxx in np.arange(0,template_A.__len__()):
           Lambda_k = np.array( Lambda_vec )
           Lambda_k[iterxx] = 0
           term1 = w_R * Lambda_vec[iterxx]
           term2 = (1 / (numSeeds - 1)) * np.matmul(p_tk[iterxx],Lambda_k)
           Lambda_vec[iterxx] = term1 + (1-w_R)*term2

    # Eq. 25 iterasyonunda sonra Eq. 28 de belirtilen efficieny değerleiri tespit ediliyor.
    Lambda_Final_Values = []
    Lambda_Final_Indices = []
    Lambda_vec_tmp = np.divide(Lambda_vec, Lambda_vec_original)
    for iter in np.arange(0,out):
        max_ind = np.argmax( Lambda_vec_tmp, axis=0 )
        if Lambda_vec_tmp[max_ind] > 0.1:
            Lambda_Final_Values.append( Lambda_vec[max_ind] )
            Lambda_Final_Indices = [Lambda_Final_Indices, max_ind]
            Lambda_vec_tmp[max_ind] = 0
        else:
            Lambda_vec_tmp[max_ind] = 0
            continue

    # Efficiency değerine göre elde edilen skor değerleir toplanarak silindir benzeşme skoru elde ediliyor.
    if Lambda_Final_Indices:
        scoresum = sum(Lambda_Final_Values)
    else:
        scoresum = 0

    return scoresum

# Eq. 25-28 'e kadar olan denklemler LSS Relaxation adımıdır. Yüksek skorlar ile elde edilen pair'ların
# birbirlerine olan uzamsal ve açısal bilgileri kullanılarak skor değerleri refine ediliyor.
def LSS_relaxation_embedding( cylinder_m1, cylinder_m2, indis_pair, numSeeds, score_vals, out ):

    numSeeds = min( out, min( numSeeds, score_vals.__len__()) )

    score_vals = score_vals[0:numSeeds]
    # for iterator in np.arange(0,numSeeds):
    #     tmp = cylinder_m1[ indis_pair[iterator][0] ]
    #     template_A.append( tmp )
    #     tmp = cylinder_m2[ indis_pair[iterator][1] ]
    #     template_B.append( tmp )

    template_A = cylinder_m1[ indis_pair[:numSeeds,0] ]
    template_B = cylinder_m2[ indis_pair[:numSeeds,1] ]

    # Eq. 26 'da belirtilen p_tk matrisini oluşturuyoruz.
    # p_tk matrisinin içindeki herbir eşeleşn minutia pair için
    # Eq. 26 da belirtilen d1 (spatial terim), d2(angular terim) ve d3(radial terim) değerleri hesaplanıyor.
    p_tk = np.zeros( ( template_A.__len__(), template_A.__len__() ) )

    t1 = np.repeat( template_A, template_A.shape[0], axis=0)
    t2 = np.tile( template_A, (template_A.shape[0],1) )

    k1 = np.repeat( template_B, template_B.shape[0], axis=0)
    k2 = np.tile( template_B, (template_B.shape[0], 1))

    ds_Atk = np.linalg.norm( t1[:,:2] - t2[:,:2] , axis=1 )
    ds_Btk = np.linalg.norm( k1[:,:2] - k2[:,:2] , axis=1 )
    d1_term = np.divide( np.abs(ds_Atk - ds_Btk) , (ds_Atk + ds_Btk) )

    tmp = np.subtract(t1[:,2], t2[:,2])
    tmp2, tmp3 = 360 + tmp, -360 + tmp
    indis = np.argmin( np.array([ abs(tmp), abs(tmp2), abs(tmp3) ]).transpose() , axis=1 )
    vals = 0*np.array( tmp )
    vals[np.where(indis==0)] = tmp[np.where(indis==0)]
    vals[np.where(indis == 1)] = tmp2[np.where(indis == 1)]
    vals[np.where(indis == 2)] = tmp3[np.where(indis == 2)]
    d_theta_A = vals

    tmp = np.subtract(k1[:, 2], k2[:, 2])
    tmp2, tmp3 = 360 + tmp, -360 + tmp
    indis = np.argmin( np.array([ abs(tmp), abs(tmp2), abs(tmp3) ]).transpose() , axis=1 )
    vals = 0 * np.array(tmp)
    vals[np.where(indis == 0)] = tmp[np.where(indis == 0)]
    vals[np.where(indis == 1)] = tmp2[np.where(indis == 1)]
    vals[np.where(indis == 2)] = tmp3[np.where(indis == 2)]
    d_theta_B = vals

    tmp = np.subtract(d_theta_A, d_theta_B)
    tmp2, tmp3 = 360 + tmp, -360 + tmp
    indis = np.argmin( np.array([ abs(tmp), abs(tmp2), abs(tmp3) ]).transpose() , axis=1 )
    vals = 0 * np.array(tmp)
    vals[np.where(indis == 0)] = tmp[np.where(indis == 0)]
    vals[np.where(indis == 1)] = tmp2[np.where(indis == 1)]
    vals[np.where(indis == 2)] = tmp3[np.where(indis == 2)]
    d2_term = vals*(np.pi/180)
    # d2_term = np.abs(directional_difference(d_theta_A, d_theta_B, 0))


    theta_m1 = t1[:,2]
    dR_m1 = ( ( np.arctan2(t1[:,0] - t2[:,0], t2[:,1] - t1[:,1]) )/np.pi) * 180
    # dR_m1_term = directional_difference(theta_m1, (dR_m1 / np.pi) * 180, 0)
    tmp = np.subtract( theta_m1, dR_m1 )
    tmp2, tmp3 = 360 + tmp, -360 + tmp
    indis = np.argmin( np.array([ abs(tmp), abs(tmp2), abs(tmp3) ]).transpose() , axis=1 )
    vals = 0 * np.array(tmp)
    vals[np.where(indis == 0)] = tmp[np.where(indis == 0)]
    vals[np.where(indis == 1)] = tmp2[np.where(indis == 1)]
    vals[np.where(indis == 2)] = tmp3[np.where(indis == 2)]
    dR_m1_term = vals # np.minimum(np.minimum( abs(tmp), abs(tmp2) ), abs(tmp3) )

    theta_m2 = k1[:,2]
    dR_m2 = ( ( np.arctan2(k1[:,0] - k2[:,0], k2[:,1] - k1[:,1]) )/np.pi) * 180
    # dR_m2_term = directional_difference(theta_m2, (dR_m2 / np.pi) * 180, 0)
    tmp = np.subtract(theta_m2, dR_m2)
    tmp2, tmp3 = 360 + tmp, -360 + tmp
    indis = np.argmin( np.array([ abs(tmp), abs(tmp2), abs(tmp3) ]).transpose() , axis=1 )
    vals = 0 * np.array(tmp)
    vals[np.where(indis == 0)] = tmp[np.where(indis == 0)]
    vals[np.where(indis == 1)] = tmp2[np.where(indis == 1)]
    vals[np.where(indis == 2)] = tmp3[np.where(indis == 2)]
    dR_m2_term = vals # np.minimum(np.minimum(tmp, tmp2), tmp3)

    # d3_term = abs(directional_difference(dR_m1_term, dR_m2_term, 0))
    tmp = np.subtract(dR_m1_term, dR_m2_term)
    tmp2, tmp3 = 360 + tmp, -360 + tmp
    indis = np.argmin( np.array([ abs(tmp), abs(tmp2), abs(tmp3) ]).transpose() , axis=1 )
    vals = 0 * np.array(tmp)
    vals[np.where(indis == 0)] = tmp[np.where(indis == 0)]
    vals[np.where(indis == 1)] = tmp2[np.where(indis == 1)]
    vals[np.where(indis == 2)] = tmp3[np.where(indis == 2)]
    d3_term = abs(vals)*(np.pi/180) # np.abs(np.minimum(np.minimum(tmp, tmp2), tmp3))

    out1 = 1. / (1 + np.exp(-1 * param['tao1P'] * (d1_term - param['mu1P'])))
    out2 = 1. / (1 + np.exp(-1 * param['tao2P'] * (d2_term - param['mu2P'])))
    out3 = 1. / (1 + np.exp(-1 * param['tao3P'] * (d3_term - param['mu3P'])))

    out = out1*out2*out3
    out[ np.where( np.isnan(out) )[0] ] = 0
    p_tk = np.reshape( out, ( template_A.__len__(), template_A.__len__() )  )

    # for iterx in np.arange( 0, template_A.__len__() ):
    #     for itery in np.arange( 0, template_A.__len__() ):
    #         if iterx != itery:
    #             tmp_t1 = template_A[iterx]
    #             tmp_t2 = template_A[itery]
    #
    #             tmp_k1 = template_B[iterx]
    #             tmp_k2 = template_B[itery]
    #
    #             # Spatial Term (d1).
    #             ds_Atk = cdist( [np.array(tmp_t1[0:2])], [np.array(tmp_t2[0:2])] )
    #             ds_Btk = cdist( [np.array(tmp_k1[0:2])], [np.array(tmp_k2[0:2])] )
    #             d1_term = np.abs(ds_Atk - ds_Btk)/( ds_Atk + ds_Btk )
    #
    #             # Angular Term (d2)...
    #             d_theta_A = directional_difference( tmp_t1[2], tmp_t2[2], 0 )
    #             d_theta_B = directional_difference( tmp_k1[2], tmp_k2[2], 0 )
    #             d2_term = np.abs(directional_difference(d_theta_A, d_theta_B, 0))
    #
    #             # Radial Term (d3)...
    #             theta_m1 = tmp_t1[2]
    #             dR_m1 = np.arctan2( tmp_t1[0] - tmp_t2[0], tmp_t2[1] - tmp_t1[1] )
    #             dR_m1_term = directional_difference(theta_m1, (dR_m1/np.pi)*180, 0 )
    #             theta_m2 = tmp_k1[2]
    #             dR_m2 = np.arctan2( tmp_k1[0] - tmp_k2[0], tmp_k2[1] - tmp_k1[1] )
    #             dR_m2_term = directional_difference(theta_m2, (dR_m2/np.pi)*180, 0)
    #             d3_term = abs(directional_difference(dR_m1_term, dR_m2_term, 0))
    #
    #             # d1, d2, ve d3 terimleri kullanılarak Eq. 26 daki Z(.) fonskiyonu hesaplanıyor.
    #             p_tk[iterx, itery] = compute_LLS_Phi( d1_term, np.pi * (d2_term / 180), np.pi * (d3_term / 180) )
    #

    # p_tk matrisi shesaplandıktan sonra Eq.(25) de belirtilen iterasyon burada başlıyor. Aşagıda berlitilen Lamba_vec terimi
    # Eq. 25 deki lambda y akaşılık gelmektedir.
    w_R = 0.5
    Lambda_vec = np.array( score_vals )
    Number_iter = 2
    # Eq. 25-28 'e kadar olan denklemler LSS Relaxation adımıdır. Yüksek skorlar ile elde edilen pair'ların
    # birbirlerine olan uzamsal ve açısal bilgileri kullanılarak skor değerleri refine ediliyor
    for out_iter in np.arange(0, Number_iter):
       # Lambda_vec_original = Lambda_vec
       Lambda_vec_original = Lambda_vec
       for iterxx in np.arange(0,template_A.__len__()):
           Lambda_k = Lambda_vec# _original
           # Lambda_k = Lambda_vec_original
           term1 = w_R * Lambda_k[iterxx]
           Lambda_k[iterxx] = 0
           term2 = (1 / (numSeeds - 1)) * np.matmul( p_tk[iterxx],Lambda_k )
           Lambda_vec[iterxx] = term1 + (1-w_R)*term2

    # Eq. 25 iterasyonunda sonra Eq. 28 de belirtilen efficieny değerleiri tespit ediliyor.
    Lambda_Final_Values = []
    Lambda_Final_Indices = []
    Lambda_vec_tmp = np.divide(Lambda_vec, score_vals )
    for iter in np.arange( score_vals.__len__() ):
        max_ind = np.argmax( Lambda_vec_tmp, axis=0 )        # cv.circle(im_sensor, tuple(i for i in coords_sensor), 10, (255, 0, 0), 3)
        Lambda_Final_Values.append( Lambda_vec[max_ind] )
        Lambda_Final_Indices.append(max_ind)
        Lambda_vec_tmp[max_ind] = 0

    # Efficiency değerine göre elde edilen skor değerleir toplanarak silindir benzeşme skoru elde ediliyor.
    if Lambda_Final_Indices:
        scoresum = sum(Lambda_Final_Values)
    else:
        scoresum = 0

    return scoresum, indis_pair, Lambda_Final_Indices
        # cv.circle(im_sensor, tuple(i for i in coords_sensor), 10, (255, 0, 0), 3)

Plot_on = 0

def main( path, path_xyt, path_embedding, filename ):

    # path = 'C:/Users/asus/PycharmProjects/cylinderRead/'
    # filename = 'log_original_egmlatent_5566_one_Images_Thread8_fmtype3.log'

    image_files = get_files_2_compare( path, filename )

    dict_results={}
    counter_true = 0

    for iter1 in range(image_files.__len__()):

        start_time = time.time()
        score_tmp = np.zeros( (image_files.__len__(),1) )

        for iter2 in range( image_files.__len__() ):

            file1 =  image_files[iter1][0]
            file2 =  image_files[iter2][1]

            file1_xyts = xyt_file_reader(path_xyt, file1)
            file2_xyts = xyt_file_reader(path_xyt, file2)

            if(Plot_on):
                im1 = cv2.imread( path_xyt + file1[:-4]+'.jpg' )
                im2 = cv2.imread( path_xyt + file2[:-4]+'.jpg' )

            ntxt = file1.split('/')
            embedding_file1 = ntxt[-3] + '_' + ntxt[-2] + '_' + ntxt[-1][:-4] + '_embedding'

            ntxt = file2.split('/')
            embedding_file2 = ntxt[-3] + '_' + ntxt[-2] + '_' + ntxt[-1][:-4] + '_embedding'

            file1_embedding, file1_embedding_length = embedding_file_reader( path_embedding, embedding_file1 )
            file2_embedding, file2_embedding_length = embedding_file_reader( path_embedding, embedding_file2 )

            # assert (file1_embedding_length==file1_xyts.__len__())
            # assert (file2_embedding_length == file2_xyts.__len__())

            # # 1. ve 2. parmak izleri için elde edilen silindirler için Eq. (17)'de belirtildiği üzere birebir benzerlik hesaplanıyor.
            # # 1. parmakta N1, 2. parmakta N2 adet silindir var ise, score matrisimiz (N1xN2) dir.
            score = pairwise_similarity_embedding(file1_embedding, file1_embedding_length, file2_embedding, file2_embedding_length)

            # # En yüksek skor değerine sahip eşleşmeler min(N1i N2) sayıda eşleşme için score matrisinden elde ediliyor.
            # # Bu eşleşme sonuçları indis_pair ve score_vals (skor değerleri) olarak bulunuyor.
            numSeeds = np.min( (file1_embedding_length, file2_embedding_length) )
            indis_pair, score_vals = SimilarityScore(score, numSeeds)
            #
            # # Eq. 24 'ü kullanarak bu eşleşmelerden sadece out sayıda olanı eşleşme LSS relaxation aşamasında kullanılıyor.
            out = np.min( (12, numSeeds) )  # compute_MinNo( file1_embedding_length, file2_embedding_length )

            # # Eq. 25-28 'e kadar olan denklemler LSS Relaxation adımıdır. Yüksek skorlar ile elde edilen pair'ların
            # # birbirlerine olan uzamsal ve açısal bilgileri kullanılarak skor değerleri refine ediliyor ve benzerlik skoru hesaplanıyor.
            # # Bu benzerlik skor değeri karşıaştırılan parmak izlerinin benzerlik değerini göstermektedir.
            scoresum, indis_pair, Lambda_Final_Indices = LSS_relaxation_embedding( file1_xyts, file2_xyts, indis_pair, numSeeds, score_vals, out )

            score_tmp[ iter2 ] = scoresum

            if(Plot_on):
                for iter, it in enumerate(Lambda_Final_Indices):
                    im1_indis = file1_xyts[ indis_pair[it][0] ]
                    cv2.circle(im1, (im1_indis[0], 512-im1_indis[1]), radius, color, thickness)
                    cv2.putText(im1, str(iter), (im1_indis[0], 512-im1_indis[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    im2_indis = file2_xyts[ indis_pair[it][1] ]
                    cv2.circle(im2, (im2_indis[0], 512-im2_indis[1]), radius, color, thickness)
                    cv2.putText(im2, str(iter), (im2_indis[0], 512-im2_indis[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                end_time = time.time()
                print('Score Val: {} '.format(scoresum))
                final_img = cv2.hconcat((im1, im2))
                plt.imshow(final_img)
                plt.show()
                plt.close()

        print('Iteration {} --  time: {} seconds per iteration'.format( iter1, ( time.time() - start_time)))
        max_ind = np.argmax( score_tmp )
        dict_results[iter1] = [ score_tmp[max_ind], max_ind, score_tmp[iter1] ]
        if(max_ind==iter1):
            counter_true+=1

    rank1_ratio = counter_true/image_files.__len__()

    print('Rank-1 ratio: {}'.format(rank1_ratio))

    return dict_results, rank1_ratio


def main2( path, path_xyt, path_embedding, filename ):

    image_files = get_files_2_compare( path, filename )

    dict_results={}
    counter_true = 0
    embedding1, embedding2, embedding1_xyt, embedding2_xyt = {}, {}, {}, {}

    np_embedding2 = np.empty( (0,256) )
    np_embedding2_xyt = np.empty( (0,3) )

    No_Samples = np.empty( (0,1) )
    for iter1 in range(image_files.__len__()):

        # start_time = time.time()

        file1 =  image_files[iter1][0]
        file2 =  image_files[iter1][1]

        file1_xyts = xyt_file_reader(path_xyt, file1)
        file2_xyts = xyt_file_reader(path_xyt, file2)

        if(Plot_on):
             im1 = cv2.imread( path_xyt + file1[:-4]+'.jpg' )
             im2 = cv2.imread( path_xyt + file2[:-4]+'.jpg' )

        ntxt = file1.split('/')
        embedding_file1 = ntxt[-3] + '_' + ntxt[-2] + '_' + ntxt[-1][:-4] + '_embedding'

        ntxt = file2.split('/')
        embedding_file2 = ntxt[-3] + '_' + ntxt[-2] + '_' + ntxt[-1][:-4] + '_embedding'

        file1_embedding, file1_embedding_length = embedding_file_reader( path_embedding, embedding_file1 )
        file2_embedding, file2_embedding_length = embedding_file_reader( path_embedding, embedding_file2 )

        tmp1 = np.linalg.norm( np.asarray(file1_embedding), axis=1 )
        tmp2 = np.linalg.norm( np.asarray(file2_embedding), axis = 1 )

        np_array_2d = np.expand_dims( tmp1, axis=0)
        tmp11 = np.repeat(a=np_array_2d, repeats=256, axis=0).transpose()
        normalized_embbeding1 = np.multiply( np.asarray(file1_embedding), 1/tmp11 )
        check_normalized_embbeding = np.linalg.norm(np.asarray(normalized_embbeding1), axis=1)

        np_array_2d = np.expand_dims(tmp2, axis=0)
        tmp22 = np.repeat(a=np_array_2d, repeats=256, axis=0).transpose()
        normalized_embbeding2 = np.multiply(np.asarray(file2_embedding), 1/tmp22)
        check_normalized_embbeding2 = np.linalg.norm(np.asarray(normalized_embbeding2), axis=1)

        embedding1[iter1] = np.asarray(normalized_embbeding1)
        embedding2[iter1] = np.asarray(normalized_embbeding2)

        np_embedding2 = np.append( np_embedding2, normalized_embbeding2, axis=0  )

        embedding1_xyt[iter1] = np.asarray(file1_xyts)
        embedding2_xyt[iter1] = np.asarray(file2_xyts)

        np_embedding2_xyt = np.append( np_embedding2_xyt, np.asarray(file2_xyts) , axis= 0 )

        No_Samples = np.append( No_Samples, embedding2_xyt[iter1].__len__()  )

        # assert (file1_embedding_length==file1_xyts.__len__())
        # assert (file2_embedding_length == file2_xyts.__len__())

    Samples_CumSum = np.cumsum( No_Samples )

    counter_1, counter_5, counter_10, counter_20 = 0, 0, 0, 0

    for iter1 in range(image_files.__len__()) :

        startx_time = time.time()
        score_tmp = np.zeros( (image_files.__len__(),1) )

        # # 1. ve 2. parmak izleri için elde edilen silindirler için Eq. (17)'de belirtildiği üzere birebir benzerlik hesaplanıyor.
        # # 1. parmakta N1, 2. parmakta N2 adet silindir var ise, score matrisimiz (N1xN2) dir.

        A_B_prod = np.matmul( embedding1[iter1], np_embedding2.transpose() )
        # A_angle = np.repeat(  np.expand_dims(np.array(embedding1_xyt[iter1][:, 2]),axis=0), np_embedding2_xyt.__len__(), 0).transpose()
        # B_angle = np.repeat(  np.expand_dims( np_embedding2_xyt[:, 2] ,axis=0 ), embedding1_xyt[iter1].__len__(), 0)
        # tmp = np.subtract(A_angle, B_angle)
        # tmp2, tmp3 = 360 + tmp, -360 + tmp
        # diff_angle = np.minimum( np.minimum(tmp, tmp2), tmp3 )
        # diff_angle[diff_angle<=100] = 1
        # diff_angle[diff_angle>100] = 0

        for iter2 in range(image_files.__len__()):

            if(iter2==0):
                start_index = 0
                end_index = Samples_CumSum[iter2]
            else:
                start_index = Samples_CumSum[iter2-1]
                end_index = Samples_CumSum[iter2]

            score = A_B_prod[:, int(start_index) : int(end_index) ]

            # # En yüksek skor değerine sahip eşleşmeler min(N1i N2) sayıda eşleşme için score matrisinden elde ediliyor.
            # # Bu eşleşme sonuçları indis_pair ve score_vals (skor değerleri) olarak bulunuyor.
            numSeeds = np.min( ( embedding1[iter1].__len__(), int(end_index-start_index) ) )
            numSeeds = np.min( (30, numSeeds) )  # compute_MinNo( file1_embedding_length, file2_embedding_length )
            out = np.min( (10, numSeeds) )

            start1 = time.time()
            indis_pair, score_vals = SimilarityScore(score, numSeeds)
            # print('Similarity Score: {} seconds per iteration'.format( 1000*(time.time() - start1)))

            # # Eq. 24 'ü kullanarak bu eşleşmelerden sadece out sayıda olanı eşleşme LSS relaxation aşamasında kullanılıyor.
            # # Eq. 25-28 'e kadar olan denklemler LSS Relaxation adımıdır. Yüksek skorlar ile elde edilen pair'ların
            # # birbirlerine olan uzamsal ve açısal bilgileri kullanılarak skor değerleri refine ediliyor ve benzerlik skoru hesaplanıyor.
            # # Bu benzerlik skor değeri karşıaştırılan parmak izlerinin benzerlik değerini göstermektedir.

            start2 = time.time()
            indis_pair = np.array(indis_pair)
            scoresum, indis_pair, Lambda_Final_Indices = LSS_relaxation_embedding( embedding1_xyt[iter1], embedding2_xyt[iter2], indis_pair, numSeeds, score_vals, out )
            # print('LSS Similarity Score: {} seconds per iteration'.format( 1000*(time.time() - start2)))

            score_tmp[ iter2 ] = scoresum # np.sum(score_vals[:8]) # scoresum
            # score_tmp[iter2] = np.sum(score_vals[:10])

        print('Iteration {} --  time: {} seconds per iteration'.format( iter1, ( time.time() - startx_time) ) )
        max_ind = np.flipud( np.argsort( score_tmp, axis=0 ) )
        dict_results[iter1] = [ score_tmp[max_ind[0]], max_ind[0], score_tmp[iter1] ]
        if(max_ind[0]==iter1):
            counter_1+=1
        elif( iter1 in max_ind[:5] ):
            counter_5+=1
        elif( iter1 in max_ind[:10] ):
            counter_10+=1
        elif( iter1 in max_ind[:20] ):
            counter_20+=1

    rank1_ratio = counter_1/image_files.__len__()
    rank5_ratio = (counter_1+counter_5) / image_files.__len__()
    rank10_ratio = (counter_1 + counter_5+counter_10) / image_files.__len__()
    rank20_ratio = (counter_1 + counter_5 + counter_10 + counter_20) / image_files.__len__()

    return dict_results, rank1_ratio, rank5_ratio, rank10_ratio, rank20_ratio

if __name__ == '__main__':

    results, rank1, rank5, rank10, rank20 = main2( path, path_xyt, path_embedding, filename )
    print('Rank-1 ratio: {} \n'.format(rank1) )
    print('Rank-5 ratio: {} \n'.format(rank5) )
    print('Rank-10 ratio: {} \n'.format(rank10))
    print('Rank-20 ratio: {} \n'.format(rank20))

    print(results)
