import os
from model import GAN
import scipy.misc
from glob import glob
def __main__():     
     if not os.path.exists('./images'):
          os.makedirs('./images')

     # path =  "./dataset\\dogs\\*.*"
     # save = "dogs\\"
     # image = glob(os.path.join(path))
     # test = []
     # for name in image:
     #     test.append(name.split('\\')[-1].split('_')[0])
     # n = list(set(test))

     # for name in image:
     #      img = scipy.misc.imread(name)
     #      cls_num = str(n.index(name.split('\\')[-1].split('_')[0]))
     #      im = save + cls_num + '_' + name.split('\\')[-1].split('_')[1]
     #      scipy.misc.imsave(im, img)

     gan = GAN()
     gan()

if __name__ == "__main__":
     __main__()