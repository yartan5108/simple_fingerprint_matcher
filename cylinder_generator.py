
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

class Cylinder_Generator:

    def  __init__(self, *args):

        if args.__len__() != 0:
            # img : parmak izi görüntüsüdür.
            self.img = args[0]

            # data değerleri (x,y,theta) minutia bilgilerini içermektedir.
            self.data = np.array( args[1] )
        else:
            self.img = []
            self.data = []

        self.Plot_on = 0

        # Tablo 2 de yer alan Parametre değerleri. Bazı parametreler makalede belirtilenden farklı kullanılmıştır.
        self.Nd = 6
        self.Ns = 16
        self.R = 70
        self.sigma_S = 10
        self.sigmaD = 2 * np.pi / 9
        self.mu = 0.005
        self.tao = 1000
        self.muP = 20
        self.taoP = 2 / 5
        self.minNp = 4
        self.maxNp = 15 # 12
        self.mu1P = 5
        self.tao1P = -8 / 5
        self.mu2P = np.pi / 12
        self.tao2P = -30
        self.mu3P = np.pi / 12
        self.tao3P = -30
        self.delta_S = 2 * self.R / self.Ns
        self.delta_D = (1 / self.Nd) * 2 * np.pi

        # Parmak izine ait minutia'lardan  Figure 5'de gösterilen convex_hull oluşturulmaktadır.
        if self.data.__len__() != 0:
            binary_image = 0 * self.img
            for iter in np.arange(0, self.data.shape[0]):
                x_m, y_m, angle = self.data[iter, :]
                binary_image[ y_m, x_m ] = 1
            self.convex_hull = convex_hull_image( binary_image )
            kernel = np.ones( (10,10), np.uint8 )
            tmp_convex = cv2.dilate( np.uint8(self.convex_hull), kernel, iterations=1 )
            self.convex_hull = tmp_convex

        # Eq.1 de gosterilen delta_phi terimi.
        k = np.arange(1, self.Nd + 1) - 0.5
        self.delta_phi = 180 * (-np.pi + k*self.delta_D ) * (1/np.pi)

    # Eq. (4)'de  belirtilen silindir fonksiyonun hesaplanmasına burada başlanıyor.
    def compute_phi(self):

        data = self.data
        data[:, 2] = data[:,2] * 1.4117  # iso-fmr dosyasından elde edilen açı değerini 0-255 ( açı değerini 1 byte olarak saklamasındna dolayı) aralığından 0-360 a map et.

        cylinder = {}

        if self.Plot_on:
            plt.figure(1)
            plt.imshow(self.img, cmap='gray')

        counter = 0

        # Minutia listemizde yer alan her bir minutia elemanı için  bir silindir (yani C(i,j,k)^m ) hesaplanmaktadır.
        for iter in np.arange(0,data.shape[0]):

            y_m, x_m, angle = data[ iter, : ]

            angle = np.pi * (angle/180)

            # (x_m, y_m) minutia sının etrafında Fig 1(b) belirtilen cuboid i oluşturmaya başlıyoruz ( yani p(i,j)^(m) 3-boyutlu matrisinin hesaplanması )..
            tmp_mat = np.array( [ [np.cos(angle), -1*np.sin(angle)], [np.sin(angle), np.cos(angle)] ] )
            pMat = np.zeros((self.Ns, self.Ns, 2))
            if self.Plot_on:
                plt.plot( x_m, y_m, 'g.')
            for i in np.arange(0,self.Ns):
                for j in np.arange(0,self.Ns):
                    tmp = np.array( [ x_m, y_m ] ).T
                    tmp = tmp.astype(float)

                    tmp2 = self.delta_S * tmp_mat
                    tmp3 = np.array( [ [ (i+1)-(0.5*(self.Ns+1))], [ (j+1)-(0.5*(self.Ns+1))] ] )
                    tmp = tmp + np.matmul( tmp2, tmp3 ).T

                    pMat[i, j, :] = [ np.max( (tmp[0,0], 1) ), np.max( (tmp[0,1], 1) ) ]

                    if self.Plot_on:
                        if np.mod(i+j,5)==2:
                            plt.plot( pMat[i, j, 0], pMat[i, j, 1], 'r.' )

            # Bu aşamada elimizde Fig.1(a) da gösterilen gösterilen cuboid discretized edilmiş şekilde (pMat variablenında) bulunmaktadır.
            # p(i, j) ^ (m) matrisinde bulunan elemanlardan R yarıçapı dışında kalan ve convex_hull dışında
            # kalan alanların çıkarılması.
            NeighborMat, indis_list, Pmat, validity = self.refine_pMat( pMat, [x_m, y_m] )
            indis_lists =  np.array( indis_list )

            # Tablo 2 'de belirtilen minimum number of valid cells (minVC) for cylinder to be valid.
            check_cell_count = np.sum( validity[:] ) <= (self.Ns * self.Ns * 0.75)
            check_minutiae_count = np.setdiff1d( np.unique( indis_lists ), iter ).shape[0] <= 1

            if (check_cell_count or check_minutiae_count):
                print(' {}, {}, {} :  Skip this minutiae, too few minutiae detected.\n'.format( x_m, y_m, angle) )
                continue

            # Eq. (4) de belirtilen C_ijk silindir terimi burada hesaplanıyor.
            Cmat = self.compute_Cmat( iter, data, angle, NeighborMat, Pmat )
            c_v, v_v = self.vectorize_Cmat( Cmat, validity)

            cylinder[ (counter,0) ] = np.array(c_v).ravel()
            cylinder[ (counter,1) ] = np.array(v_v).ravel()
            cylinder[ (counter,2) ] = [ x_m, y_m, angle ]
            counter+=1

        print('Cylinder Generator:..')
        return cylinder, counter


    # refine_pMat fonksiyonu pMat matrisini oluşturuken R yarıçapının dışında kalan ve minuşalardan oluşan convex_hull'ın
    # dışından bulunan pMat(,i,j,k) elemanlarını sıfırlamakta, yani çıkarmaktadır.
    def refine_pMat(self, pMat, xtest ):

        Pmat = pMat
        Validity = np.ones( (self.Ns, self.Ns) )

        for i in np.arange(0, self.Ns):
            for j in np.arange(0, self.Ns):

                tmp1 = np.round( Pmat[i, j, 0] ).astype(int)
                tmp2 = np.round( Pmat[i, j, 1] ).astype(int)

                if tmp1<0 or tmp2<0:
                    Pmat[i, j, :] = 0
                    Validity[i, j] = 0

                try:
                    if  tmp1< self.convex_hull.shape[0] and tmp2 < self.convex_hull.shape[1]:
                        if self.convex_hull[tmp1, tmp2 ] == 0:
                            Validity[i,j] = 0

                        v1 = Pmat[i,j,:]
                        v2 = xtest

                        if np.sqrt( (v1[0]-v2[0])**2 + (v1[1]-v2[1])**2 ) >= self.R :
                            Pmat[i,j,:] = 0
                            Validity[i,j] = 0
                except:
                    print('Problem var..')


        NeighborMat, indis_list = self.Compute_NeighborHood( Pmat, Validity )
        return NeighborMat, indis_list, Pmat, Validity


    # (x_m, y_m) minutia sı için Cuboid oluşturulurken p(i,j) noktasının yakınındaki terimler burada tespit ediliyor.
    # Yani  Eq. 4'deki N_p(i,j)^m termini her bir i ve j terimi için hesaplıyoruz.
    def Compute_NeighborHood( self, Pmat, validity ):

         indis_list = []
         indis_list = np.asarray( indis_list )
         NeighborMat = {}
         tmp_data = self.data[:, 0:2]
         for i in np.arange(0,Pmat.shape[0]):
            for j in np.arange(0,Pmat.shape[1]):
                if ( Pmat[i,j,0]!=0 and Pmat[i,j,1]!=0 ):
                    if validity[i, j] != 0:
                        v1 = Pmat[i, j, :]
                        ind =  np.where( cdist( tmp_data, np.fliplr([v1]) ) <= 3*self.sigma_S )
                        NeighborMat[(i, j)] = np.asarray( ind[0] )
                        indis_list = np.append(indis_list,ind[0])
         return NeighborMat, indis_list


    # Eq. 4 'de berlitilen Cmat matrisinin değerleri hesaplanmakatadır.
    def compute_Cmat( self, iter, data, angle, NeighborMat, Pmat ):

        Cmat = np.zeros( (self.Ns, self.Ns, self.Nd) )
        if self.Plot_on:
           plt.plot( data(iter, 1), data(iter, 2), 'r.' )

        for i in np.arange(0, self.Ns):
            for j in np.arange(0, self.Ns):
               tmp_cell_loc = Pmat[i, j,: ]

               try:
                  tmp_neigh = np.setdiff1d( NeighborMat[(i, j)], iter )
               except:
                  continue

               if tmp_neigh.any():
                   for k in np.arange(0, self.delta_phi.shape[0]):
                       sum_tmp = 0
                       for vals in tmp_neigh:
                           if vals:
                               # Eq. (6) belirtilen uzamsal terim hesaplanıyor...
                               tmp_spatial = self.compute_spatial_dist( data[vals, 0:2], Pmat[i,j,:] )
                               # Eq. (8) belirtilen açısal terim hesaplanıyor...
                               tmp_directional = self.compute_directional_dist( angle, self.delta_phi[k], data[vals, 2] )

                           if vals:
                               # açısla terimle uzamsal terimin çarpımı eq.(4)'de kullanılmaktadır.
                               sum_tmp = sum_tmp + tmp_spatial * tmp_directional

                        # sum_tmp terimi Eq. 4^de bulunan Phi fonsiyonuna geçirilmektedir ve Cmat(i,j,k) hesaplanmakatadır.
                       Cmat[i,j,k] = self.compute_Phi2(sum_tmp)
        return Cmat

    # Eq. (6) belirtilen uzamsal terim hesaplanıyor...
    def compute_spatial_dist(self,a,b):
        dist_val = cdist( np.fliplr([a]), np.array([b]) )
        # Eq. (7) de berlitilen terim hesaplanmaktadır.
        out = (1/(self.sigma_S * np.sqrt(2 * np.pi))) * np.exp( -0.5 * (dist_val / self.sigma_S)**2 )
        return out

    # Eq. (8) belirtilen açısal terim hesaplanıyor. Bunu hesaplarken Eq. 9, 10 ve Eq. 11 kullanılmaktadır.
    def compute_directional_dist(self, minutia_angle, delta_phi, test_point ):
        # Eq. 9 ile directional difference tespit ediliyor. Directional difference (d_phi) Eq. 11'e veriliyor.
        d_theta = self.directional_difference( minutia_angle*(180/np.pi), test_point )
        d_phi = self.directional_difference( delta_phi, d_theta )

        # Eq 11 ile g_d hesaplanıyor.
        g_d = self.compute_gaussian( d_phi )
        return g_d

    # Eq. 9'da belirtilen terim. Bu denklem 10 da kullanılıyor.
    def directional_difference(self, term1, term2):
        if abs( term1 - term2) < 180:
            d_phi = term1 - term2
        elif (term1 - term2) <= -180:
            d_phi = 360 + (term1 - term2)
        elif(term1 - term2) >= 180:
            d_phi = -360 + (term1 - term2)
        return d_phi

    # Eq. 11 'de berlitilen terim..
    def compute_gaussian(self, alpha):
        delta_subtract = self.delta_D*57 # change radian to degrees
        alpha_int = (1/180)*np.array([ np.array(alpha) - np.array(delta_subtract)/2, np.array(alpha) + np.array(delta_subtract)/2]) # convert back to radians.
        p = norm.cdf( alpha_int, loc = 0, scale = self.sigmaD )
        out = p[1] - p[0]
        return out

    # Eq.(5) 'de belirtilen  Z(v, m, tao)
    def compute_Phi2(self,v):
        out = 1. / (1 + np.exp( -1* self.tao  * (v - self.mu)) )
        return out

    def vectorize_Cmat(self, Cmat, validity ):
        c_vec, v_vec = [], []
        for iterx in np.arange(0, Cmat.shape[2] ):
            tmp = Cmat[ :, :, iterx ]
            tmp2 = validity
            c_vec.append( np.ravel(tmp) )
            v_vec.append( np.ravel(tmp2) )
        return c_vec, v_vec


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

def cvec_ab( a, a_2, b, b_2, Ns, Nd ):

    Nc = Ns * Ns * Nd
    c_vec_ab = np.zeros( Nc )
    c_vec_ba = np.zeros( Nc )

    indis_overlap = np.where( np.multiply( np.array(a_2), np.array(b_2) )> 0 )
    c_vec_ab[indis_overlap] = a[indis_overlap]
    c_vec_ba[indis_overlap] = b[indis_overlap]

    return c_vec_ab, c_vec_ba

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

def main():

    # Parmak izi görüntümülerimizin ./Db1_a/,
    # Iso-fmr dosyalarının ( iso-fmr dosyası parmak izinde bulunan minutia ların x, y, theta bilgilerini içerkemektedir)
    # ./FM3_FVC2002DB1A/ dizininde oldugunu varsayıyoruz.
    Image_Path = os.getcwd() + '/' # './Db1_a/'
    FMR_path = os.getcwd() + '/' # './FM3_FVC2002DB1A/'

    # Cylinder_generator initialize ediliyor, burası önemli değil, atlanabilir.
    global param
    param_tmp = Cylinder_Generator()
    param = ParamsAll( param_tmp )

    start_time = time.time()

    # Birinci parmak izi için (1_1) ilk once iso-fmr dosyasını okuyoruz  (fmr reader ile) ve her bir minutia için
    # silindir dosyasını oluşturuyoruz. FMR reader'in içeriği önemli degil, döndürdüğü data (yani minutia lar) ve im (yani parmak izi görüntüsüdür)
    # kullanılmaktadır. Data'nin içinde her bir minutia için (x, y, theta, quality) değerleri bulunmakatdır.
    imagename = '1_1'
    data, im = fmr_reader( Image_Path, FMR_path, imagename)
    # Birinci parmak izi (im) ve minutia bilgileri kullanılarak cylinder_generator initialize ediliyor.
    cylinder = Cylinder_Generator( im, data )
    # Compute_Phi fonksiyonu ile her birminutia için silindir üretiliyor ve cylinderV1 içinde tutuluyor.
    # Her bir minutia için elde edilen silindirlerin boyutu 1x(Ns*Ns*Nd) olup, counterV1 adet bulunmaktadır.
    cylinderV1, counterV1 = cylinder.compute_phi()

    # Birinci parmak izi için yapılan işlemler 2. parmak izi içinde benzer şekilde yapılıyor.
    imagename_test = '1_2'
    data_test, im_test = fmr_reader(Image_Path, FMR_path, imagename_test)
    cylinder_test = Cylinder_Generator( im_test, data_test )
    cylinder_testV1, counter_testV1  = cylinder_test.compute_phi()

    # 1. ve 2. parmak izleri için elde edilen silindirler için Eq. (17)'de belirtildiği üzere birebir benzerlik hesaplanıyor.
    # 1. parmakta N1, 2. parmakta N2 adet silindir var ise, score matrisimiz (N1xN2) dir.
    score = pairwise_similarity( cylinderV1, counterV1, cylinder_testV1, counter_testV1 )

    # En yüksek skor değerine sahip eşleşmeler min(N1i N2) sayıda eşleşme için score matrisinden elde ediliyor.
    # Bu eşleşme sonuçları indis_pair ve score_vals (skor değerleri) olarak bulunuyor.
    numSeeds = np.min( (cylinder.data.shape[0], cylinder_test.data.shape[0]) )
    indis_pair, score_vals = SimilarityScore(score, numSeeds)

    # Eq. 24 'ü kullanarak bu eşleşmelerden sadece out sayıda olanı eşleşme LSS relaxation aşamasında kullanılıyor.
    out = compute_MinNo( cylinder.data.shape[0],  cylinder_test.data.shape[0] )

    # Eq. 25-28 'e kadar olan denklemler LSS Relaxation adımıdır. Yüksek skorlar ile elde edilen pair'ların
    # birbirlerine olan uzamsal ve açısal bilgileri kullanılarak skor değerleri refine ediliyor ve benzerlik skoru hesaplanıyor.
    # Bu benzerlik skor değeri karşıaştırılan parmak izlerinin benzerlik değerini göstermektedir.
    scoresum = LSS_relaxation( cylinderV1, cylinder_testV1, indis_pair, numSeeds, score_vals, out )

    end_time = time.time()
    print('Total Time: {} seconds'.format(  (end_time-start_time) ) )
    print('Score Val: {} '.format( scoresum ) )
    print( 'Cylinder {} x {} x {}'.format( cylinder.Nd, cylinder.Ns, cylinder.Ns ) )

if __name__ == '__main__':

    main()
