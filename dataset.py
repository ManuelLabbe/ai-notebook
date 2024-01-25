import os
import gzip
import numpy as np

class dataset:
    def __init__(self, image_path, label_path):
        self.image_path = image_path
        self.label_path = label_path

    def load_dataset(self):
            X = dataset.cargar_imagen(self.image_path)
            y = dataset.cargar_etiquetas(self.label_path)    
            X = X.reshape(-1,28,28)
            return X, y
        
    def cargar_imagen(image_path):
        with gzip.open(image_path, 'rb') as f:
            content = f.read()
            imagenes = np.frombuffer(content, dtype=np.uint8, offset=16)
        return imagenes

    def cargar_etiquetas(label_path):
        with gzip.open(label_path, 'rb') as f:
            content = f.read()
            etiquetas = np.frombuffer(content, dtype=np.uint8, offset=8)
        return etiquetas
    
    

if __name__ == "__main__":
    
    path_train_images = './mnist_dataset/train-images-idx3-ubyte.gz'
    path_train_labels = './mnist_dataset/train-labels-idx1-ubyte.gz'
    path_test_images = './mnist_dataset/t10k-images-idx3-ubyte.gz'
    path_test_labels = './mnist_dataset/train-labels-idx1-ubyte.gz'
    

    

