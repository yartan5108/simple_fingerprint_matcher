
import os, sys
import cv2
import numpy as np
from matplotlib import pyplot as plt

Plot_on = 0

def fmr_reader( Image_Path, FMR_path, imagename):

    im = plt.imread( Image_Path + imagename + '.tif' )
    img = im

    imx, imy = im.shape

    fid = open(  FMR_path + imagename + ".iso-fmr" , 'rb'   )   #"r" , encoding='cp850' )

    # Index-1 : Format Identifier
    c1 = []
    num1 = 4
    for iter in range(num1):
        cc = fid.read(1)
        c1.append( chr(ord(cc)) )
    #print(' Format Identifier: %s \n',  c1 )

    # Index-2 : Version
    c2 = []
    num2 = num1+4
    for iter in range(num1,num2,1):
        cc = fid.read(1)
        c2.append( chr(ord(cc)) )
    #print(' Version: %s \n',  c2 )

    # Index-3 : Total Length
    c3 = []
    num3 = num2+4
    for iter in range(num2,num3,1):
        cc = fid.read(1)
        c3.append( ord(cc) )
    c3 = list( reversed(c3) )
    total_len = c3[0]*1+c3[1]*2**8+c3[2]*2**16+c3[3]*2**24
    #print(' Total Length: %2.2f \n',  total_len  )

    # Index-4-5 : Dummy 4 ve 5 e kar��l�k gelen dummmy
    c4_5 = []
    num4 = num3 + 2
    for iter in range(num3,num4,1):
        cc = fid.read(1)
        c4_5.append( ord(cc) )

    # Index-4-5 : Dummy
    c6 = []
    num5 = num4 + 2
    for iter in range(num4,num5,1):
        cc = fid.read(1)
        c6.append( ord(cc) )
    c6 = list( reversed(c6) )
    X_ = c6[0]*1+c6[1]*2**8

    #print(' Image size X: %2.2f \n',  X_  )

    # Image size ...
    c7 = []
    num6 = num5 + 2
    for iter in range(num5,num6,1):
        cc = fid.read(1)
        c7.append( ord(cc) )
    c7 = list( reversed(c7) )
    Y_ = c7[0]*1+c7[1]*2**8
    #print(' Image size Y: %2.2f \n',  Y_  )
    #print(' Orginal Image size  %2.2f x %2.2f \n',  imx, imy )

    # Image size xxxx
    c8 = []
    num7 = num6 + 2
    for iter in range(num6,num7,1):
        cc = fid.read(1)
        c8.append(ord(cc))
    c8 = list( reversed(c8) )
    X_res = c8[0]*1+c8[1]*2**8
    #print(' Image Resolution X: %2.2f \n',  X_res  )


    c9 = []
    num8 = num7 + 2
    for iter in range(num7,num8,1):
        cc = fid.read(1)
        c9.append(ord(cc))
    c9=  list( reversed( c9 ) )
    Y_res = c9[0]*1+c9[1]*2**8
    #print(' Image Resolution Y: %2.2f \n',  Y_res  )


    c10 = []
    num9 = num8 + 1
    for iter in range(num8,num9,1):
        cc = fid.read(1)
        c10.append(ord(cc))
    c10 =  list( reversed( c10 ) )
    No_fingers = c10[0]
    #print(' No Finger Views: %2.2f \n',  No_fingers  )


    c11 = []
    num10 = num9 + 1
    for iter in range(num9,num10,1):
        cc = fid.read(1)
        c11.append(ord(cc))
    c11 = list( reversed( c11 ) )
    Reserved = c11[0]


    c12 = []
    num11 = num10 + 1
    for iter in range( num10,num11,1):
        cc = fid.read(1)
        c12.append(ord(cc))
    c12 = list( reversed( c12 ) )
    FG_pos = c12[0]
    #print(' No Finger Pos: %2.2f \n',  FG_pos  )


    c13_14 = []
    num12 = num11 + 1
    for iter in range(num11,num12,1):
        cc = fid.read(1)
        c13_14.append(ord(cc))
    c13_14 = list( reversed( c13_14 ) )
    View_no = c13_14[0]


    c15 = []
    num13 = num12 + 1
    for iter in range(num12,num13,1):
        cc = fid.read(1)
        c15.append(ord(cc))
    c15 = list( reversed( c15 ) )
    F_Quality = c15[0]
    #print(' Finger Quality: %2.2f \n',  F_Quality  )


    # Index-16 : Number of Minutaes
    c16= []
    num14 = num13 + 1
    for iter in range(num13,num14,1):
        cc = fid.read(1)
        c16.append(ord(cc))
    c16 = list( reversed( c16 ) )
    No_Mins = c16[0]
    #print(' No Minutiae: %2.2f \n',  No_Mins  )

    if Plot_on:
        plt.imshow( img, cmap='gray' )

    # Index-17-XX: Minutia locs
    data = []
    for iter_out in range(0,No_Mins):

        # Minuatia X...
        cc = fid.read(1)
        tmp = bin( ord(cc) )[2:].zfill(8)
        a = int( tmp, 2)
        b = int( '00111111',2 )
        tmp2 = int( bin(a & b)[2:].zfill(8) ,2 )
        tmp3 = ord( fid.read(1) )
        X = tmp3 * 1 + tmp2 * (2**8)

        if tmp[0:2]=='10':
            classname = 'Bifurcation'
        elif tmp[0:2]=='01':
            classname = 'Ending'
        elif tmp[0:2]=='00':
            classname = 'Other'
        else:
            classname = 'Unknown'

        # Minuatia Y...
        cc = fid.read(1)
        tmp = bin(ord(cc))[2:].zfill(8)
        a = int(tmp, 2)
        b = int('00111111', 2)
        tmp2 = int(bin(a & b)[2:].zfill(8), 2)
        tmp3 = ord(fid.read(1))
        Y = tmp3 * 1 + tmp2 * (2 ** 8)

        c19 = []
        num15 = num14 + 2
        num16 = num15 + 1
        for iter in range(num15,num16,1):
            cc = fid.read(1)
            c19.append(ord(cc))
        c19 = list( reversed( c19 ) )
        Mins_Theta = c19[0]

        c20 = []
        num17 = num16 + 1
        for iter in range(num16,num17,1):
            cc = fid.read(1)
            c20.append(ord(cc))
        c20 = list(reversed(c20))
        Mins_Q = c20[0]

        theta = Mins_Theta*1.4
        theta = theta/360
        X_delta, Y_delta = 20, 0
        r = 20
        o = 2*np.pi*theta

        data.append( [ X, Y, Mins_Theta ])

        X_prime = X_delta * np.cos(o) - Y_delta * np.sin(o)
        Y_prime = X_delta * np.sin(o) + Y_delta * np.cos(o)

        if Plot_on:
            plt.quiver( X, Y, X_prime, Y_prime)
            plt.plot([X, X + r * np.cos(o)], [Y, Y - r * np.sin(o)], 'r-')

            plt.text(X,Y,classname[0:3])
            #print(' Min. No: {},  X: {}, Y: {}, Theta: {}, Q: {} \n'.format( iter_out, X, Y, Mins_Theta, Mins_Q) )

    if Plot_on:
        plt.show()

    # Index-21-XX: Minutia locs
    c21= []
    num18 = num17 + 2
    for iter in range(num17,num18,1):
        cc = fid.read(1)
        c21.append(ord(cc))

    c21 = list(reversed(c21))
    Tmp = c21

    fid.close()

    return data, im
