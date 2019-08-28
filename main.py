import os
from model import GAN

def __main__():     
     if not os.path.exists('./images'):
          os.makedirs('./images')
     if not os.path.exists('./dataset'):
          os.makedirs('./dataset')
   
     gan = GAN(image_path = "./dataset/train.tfrecords", image_num = 17720)
     gan()

if __name__ == "__main__":
     __main__()