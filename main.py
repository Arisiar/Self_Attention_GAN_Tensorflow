import os
from model import GAN
import scipy.misc
from glob import glob
def __main__():     
     if not os.path.exists('./images'):
          os.makedirs('./images')
     if not os.path.exists('./dataset'):
          os.makedirs('./dataset')
          
     gan = GAN()
     gan()

if __name__ == "__main__":
     __main__()