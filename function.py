import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from sklearn.decomposition import PCA
import os

def PCA_scratch(X , num_components):

    #Step-1
    X_meaned = X - np.mean(X , axis = 0)

    #Step-2
    cov_mat = np.cov(X_meaned , rowvar = False)

    #Step-3
    eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)

    #Step-4
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:,sorted_index]

    #Step-5
    eigenvector_subset = sorted_eigenvectors[:,0:num_components]

    #Step-6
    X_reduced = np.dot(eigenvector_subset.transpose() , X_meaned.transpose() ).transpose()

    return X_reduced,sorted_eigenvalue

def create3DArrayPh():
  lstname = glob.glob('*/water/*')
  lstname.sort()
  first = int(lstname[0].replace('.', '/').split('/')[-2])
  last = int(lstname[-1].replace('.', '/').split('/')[-2])
  # create zeros array to store all of pictures to be 3d array.
  abc = np.zeros((512,512,last-first+1))
  for name in lstname:
    number = int(name.replace('.', '/').split('/')[-2])
    img = cv2.imread(name,0)
    abc[:,:,number-first] = img
  return abc>=255

def edges_piece(labels_outPh,num):
  slides_ = labels_outPh.shape[2]
  canny_labels = np.zeros((512, 512, slides_))
  for i in range(slides_):
    img1 = (labels_outPh==num)[:,:,i]
    edges = cv2.Canny(img1.astype('uint8')*255,100,200)
    canny_labels[:,:,i] = edges
  dx,dy,dz = np.where(canny_labels == 255)
  dfarray = np.array([dx/0.447265625,dy/0.447265625,dz/1.5]).T
  X_reduced,eigen_values = PCA_scratch(dfarray,3)

  return canny_labels,X_reduced,eigen_values,dfarray

def res_vol_disk(labels_outPh):
  result_list = []
  for i in range(1,labels_outPh.max()+1):
    r_res = (labels_outPh==i).sum()*0.447265625*0.447265625*1.5/1000
    result_list.append([i,r_res])
  return result_list

def pca3_piece(labels_outPh,piece):
  dx,dy,dz = np.where(labels_outPh == piece)
  dfarray = np.array([dx*0.447265625,dy*0.447265625,dz*1.5]).T
  if dfarray.shape[0] <= 1:
    return 0
  mat_reduced,eigen_values = PCA_scratch(dfarray,3)
  principal_df = pd.DataFrame(abs(mat_reduced) , columns = ['PC1','PC2','PC3'])
  a_s = 2*principal_df['PC1'].max()
  b_s = 2*principal_df['PC2'].max()
  c_s = 2*principal_df['PC3'].max()
  return a_s*b_s*c_s/2000

def res_vol_pca3(labels_outPh):
  result_list = []
  for i in range(1,labels_outPh.max()+1):
    try:
      result_list.append([i,float('%.6f'%pca3_piece(labels_outPh,i))])
    except:
      pass
  return result_list

def calallvalues(labels_outPh):
  
  #find volume disk method
  lst_v_disk = res_vol_disk(labels_outPh)
  df2 = pd.DataFrame(lst_v_disk, columns =['Num', 'DiskMethod'])

  #find volmue PCA3
  lst_v_pca3 = res_vol_pca3(labels_outPh)
  df4 = pd.DataFrame(lst_v_pca3, columns =['Num', 'PCA_3'])

  #Merge df
  df = pd.merge(df2, df4, on="Num")
  return df

def findminmax(x):
  # minimize volume for less time consuming of visualizing
  avai1st = []
  for i in range(x.shape[0]):
    if x[i,:,:].sum():
      avai1st.append(i)
  avai1st = np.array(avai1st)
  min1st = avai1st.min()
  max1st = avai1st.max()
  avai2nd = []
  for i in range(x.shape[1]):
    if x[:,i,:].sum():
      avai2nd.append(i)
  avai2nd = np.array(avai2nd)
  min2nd = avai2nd.min()
  max2nd = avai2nd.max()
  avai3rd = []
  for i in range(x.shape[2]):
    if x[:,:,i].sum():
      avai3rd.append(i)
  avai3rd = np.array(avai3rd)
  min3rd = avai3rd.min()
  max3rd = avai3rd.max()
  return min1st,max1st,min2nd,max2nd,min3rd,max3rd

def show_vol_Ph(x):
  min1st,max1st,min2nd,max2nd,min3rd,max3rd = findminmax(x)
  # visualize a specific component
  fig = plt.figure(figsize=(10, 10))
  ax = fig.add_subplot(projection='3d')
  ax.grid(True)
  ax.voxels(x[min1st:max1st+1,min2nd:max2nd+1,min3rd:max3rd+1], facecolors='#7A88CC60', edgecolors='#7D84A6')
  ax.view_init(15, 15)
  # Get the current directory
  dir = os.path.dirname(os.path.abspath(__file__))
  result_filename = os.path.join(dir, "result.jpeg")
  plt.savefig(result_filename)
  plt.close()

if __name__ == '__main__':
  print(os.path.dirname(os.path.abspath(__file__)))
