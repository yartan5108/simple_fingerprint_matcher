
import os, sys
import cv2
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import norm
from skimage.morphology import convex_hull_image
from scipy.spatial.distance import cdist

from file_reader import fmr_reader
from params import ParamsAll

import time
import concurrent.futures

for k, v in os.environ.items():
    if k.startswith("QT_") and "cv2" in v:
        del os.environ[k]

global similarity_thresh
similarity_thresh = 0.3


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

                    file1, file2 = line[11], line[13]

                    # inds = []
                    # m_locs = line_matcher.split('[')[13][:-2].split(',')
                    # for vals in range(0, m_locs.__len__(), 3):
                    #     ind1, ind2 = int( m_locs[vals] ), int( m_locs[vals+1] )
                    #     inds.append( (ind1, ind2) )

                    image_files[counter] = [ file1, file2, RankVal, skore, ]
                    counter += 1
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
# 000000000000000000000000000000000000000000000000000000000000000000610320028190111
# 0000000010001000101110110000000000000000110111010000000000000000
# 0000000000000000000000000000000000001100110000000000000000110000
# 0000111111111111111111111111000000000000111111111111111100000000
# 1111111111111111111111111111111111111111111111111111111111111111
# 1111111111111111111111111111111111111111111111111111111111111111
# 0000000011111111111111110000000000001111111111111111111111110000
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
    global param
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
def pairwise_similarity_embedding( cy1, counter_cy1, cy2, counter_cy2 ):
    global param
    score_mat = np.zeros( (counter_cy1, counter_cy2) )
    for iter_m1 in np.arange( 0, counter_cy1 ):
        for iter_m2 in np.arange( 0, counter_cy2 ):
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

# Eq. 24 de belirtilen (nP) değeri burada tespit edilmektedir.
def compute_MinNo( Na, Nb ):
    global param
    min_Nab = np.min( (Na, Nb) )
    out = 1./( 1 + np.exp( -1*param['muP']*( min_Nab - param['taoP'] ) ) )
    difference = ( param['maxNp'] - param['minNp'] )
    out = param['minNp'] + round( out*difference )
    return out

def directional_difference( term1, term2, deg_flag_on):
    if deg_flag_on:
        term1 = term1*(180/np.pi)
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

# Eq. 25-28 'e kadar olan denklemler LSS Relaxation adımıdır. Yüksek skorlar ile elde edilen pair'ların
# birbirlerine olan uzamsal ve açısal bilgileri kullanılarak skor değerleri refine ediliyor.
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



def main():

    path = 'C:/Users/asus/PycharmProjects/cylinderRead/'
    filename = 'log_original_egmlatent_5566_one_Images_Thread8_fmtype3.log'

    image_files = get_files_2_compare(path, filename)

    # 0010100105638033\17718826.s
    iter = 0
    file1 = image_files[iter][0]
    file2 = image_files[iter][1]

    start_time = time.time()

    path_xyt = os.getcwd()
    file1_xyt = xyt_file_reader(path_xyt, file1)
    file2_xyt = xyt_file_reader(path_xyt, file2)


    # # 1. ve 2. parmak izleri için elde edilen silindirler için Eq. (17)'de belirtildiği üzere birebir benzerlik hesaplanıyor.
    # # 1. parmakta N1, 2. parmakta N2 adet silindir var ise, score matrisimiz (N1xN2) dir.
    # score = pairwise_similarity_embedding(cylinderV1, counterV1, cylinder_testV1, counter_testV1)
    #
    # # En yüksek skor değerine sahip eşleşmeler min(N1i N2) sayıda eşleşme için score matrisinden elde ediliyor.
    # # Bu eşleşme sonuçları indis_pair ve score_vals (skor değerleri) olarak bulunuyor.
    # numSeeds = np.min((cylinder.data.shape[0], cylinder_test.data.shape[0]))
    # indis_pair, score_vals = SimilarityScore(score, numSeeds)
    #
    # # Eq. 24 'ü kullanarak bu eşleşmelerden sadece out sayıda olanı eşleşme LSS relaxation aşamasında kullanılıyor.
    # out = compute_MinNo(cylinder.data.shape[0], cylinder_test.data.shape[0])
    #
    # # Eq. 25-28 'e kadar olan denklemler LSS Relaxation adımıdır. Yüksek skorlar ile elde edilen pair'ların
    # # birbirlerine olan uzamsal ve açısal bilgileri kullanılarak skor değerleri refine ediliyor ve benzerlik skoru hesaplanıyor.
    # # Bu benzerlik skor değeri karşıaştırılan parmak izlerinin benzerlik değerini göstermektedir.
    # scoresum = LSS_relaxation(cylinderV1, cylinder_testV1, indis_pair, numSeeds, score_vals, out)

    end_time = time.time()
    print('Total Time: {} seconds'.format((end_time - start_time)))
    print('Score Val: {} '.format(scoresum))
    print('Cylinder {} x {} x {}'.format(cylinder.Nd, cylinder.Ns, cylinder.Ns))


if __name__ == '__main__':

    main()
