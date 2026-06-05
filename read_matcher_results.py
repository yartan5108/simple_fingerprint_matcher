import os
import glob as glob
import numpy as np
import shutil

def txt_file_reader(path_txt, filename):
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

            if(read_flag):
                line_matcher = line[-1]

                RankVal = int( line[3] )
                skore = float( line[4] )
                file1, file2 = line[11], line[13]

                inds = []
                m_locs = line_matcher.split('[')[13][:-2].split(',')
                for vals in range(0, m_locs.__len__(), 3):
                    ind1, ind2 = int( m_locs[vals] ), int( m_locs[vals+1] )
                    inds.append( (ind1, ind2) )

                image_files[counter] = [ file1, file2, RankVal, skore, ]
                counter += 1
    return line


def main( path, filename ):
    txt_file_reader( path, filename )

import os

if __name__=="__main__":

    # path = '/home/yartan/Documents/AFIS_2/server/AFIS/src/comparison2/FMLib/cmake-build-debug/'
    path = 'C:/Users/asus/PycharmProjects/cylinderRead/'
    filename = 'log_original_egmlatent_5566_one_Images_Thread8_fmtype3.log'
    main(path, filename)
