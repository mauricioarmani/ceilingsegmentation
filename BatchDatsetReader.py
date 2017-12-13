"""
Code ideas from https://github.com/Newmu/dcgan and tensorflow mnist dataset reader
"""
import numpy as np
import scipy.misc as misc


class BatchDatset:
    files = []
    images = []
    annotations = []
    image_options = {}
    batch_offset = 0
    epochs_completed = 0

    def __init__(self, records_list, image_options={}):
        """
        Intialize a generic file reader with batching for list of files
        :param records_list: list of file records to read -
        sample record: {'image': f, 'annotation': annotation_file, 'filename': filename}
        :param image_options: A dictionary of options for modifying the output image
        Available options:
        resize = True/ False
        resize_size = #size of output image - does bilinear resize
        color=True/False
        """
        print("Initializing Batch Dataset Reader...")
        print(image_options)
        self.files = records_list
        self.image_options = image_options
        self._read_images()

    def _read_images(self):
        self.__channels = True
        self.images = np.array([self._transform(filename['image']) for filename in self.files])
        self.__channels = False
        # self.annotations = np.array(
        #     [np.expand_dims(self._transform(filename['annotation']), axis=3) for filename in self.files])

        self.annotations = np.array(
            [self._transform(filename['annotation']) for filename in self.files])

        # 224, 192, 0

        #224 224 192 borda
        #0   0   0   preto
        #225 0   0   vermelho
        
        # unique_values =  np.unique(self.annotations[0][:,:,0])

        resize_size = 256
        class_matrix = np.zeros(shape=(len(self.annotations),resize_size, resize_size, 1), dtype='int32')

        for i, (image, filename) in enumerate(zip(self.annotations, self.files)):

            mask_bool = image[:,:,0] == 255
            if np.sum(mask_bool) == 0:
                print(i, filename, np.max(image), np.mean(image), np.min(image))
            mask_im = np.zeros(shape=(256, 256))          
            mask_im[mask_bool] = 1
            image = np.zeros(shape=(256,256,1))            
            image[:,:,0] = mask_im
            class_matrix[i] = image 

        self.annotations = class_matrix

        print('images: ',self.images.shape)
        print('annotations: ', self.annotations.shape)



    def _transform(self, filename):
        image = misc.imread(filename)
        if self.__channels and len(image.shape) < 3:  # copia as imagens gray para tres canais
            image = np.array([image for i in range(3)])

        if self.image_options.get("resize", False) and self.image_options["resize"]:
            resize_size = int(self.image_options["resize_size"])
            resize_image = misc.imresize(image,
                                         [resize_size, resize_size], interp='nearest')
        else:
            resize_image = image

        return np.array(resize_image)

    def get_records(self):
        return self.images, self.annotations


    def reset_batch_offset(self, offset=0):
        self.batch_offset = offset


    def next_batch(self, batch_size):
        start = self.batch_offset
        self.batch_offset += batch_size
        if self.batch_offset > self.images.shape[0]:
            # Finished epoch
            self.epochs_completed += 1
            print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            # Shuffle the data
            perm = np.arange(self.images.shape[0])
            np.random.shuffle(perm)
            self.images = self.images[perm]
            self.annotations = self.annotations[perm]
            # Start next epoch
            start = 0
            self.batch_offset = batch_size

        end = self.batch_offset
        return self.images[start:end], self.annotations[start:end]


    def get_random_batch(self, batch_size):
        indexes = np.random.randint(0, self.images.shape[0], size=[batch_size]).tolist()
        return self.images[indexes], self.annotations[indexes]
