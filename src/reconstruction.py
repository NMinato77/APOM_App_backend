import cupy as cp
import numpy as np
from skimage.io import imread
import os
from astropy.io import fits
from cupyx.scipy.ndimage import affine_transform
import tifffile

# set fixed parameters
THETA = np.arcsin(np.sin(0.62)/1.33) # angle of the light sheet (radian)
PIXEL_SIZE = 6.5 # pixel size of the image sensor (um)
MAG = 200/9 # magnification of the imaging system
F_INDEXMISMATCH = 1.412 # coefficient that corrects for the depth deformation due to refractive-index mismatch between the primary and secondary objectives
COEFF405 = 0.994 # coefficient that corrects for refractive-index dispersion between 405 and 488 channels
COEFF637 = 1.0035 # coefficient that corrects for refractive-index dispersion between 637 and 488 channels
SHIFT_405 = [ -5.51067196, -19.86035447, -12.61294245] # shift in x, y, z for 405 nm channel
SHIFT_637 = [12.27092666,  1.06970746, -4.33760026] # shift in x, y, z for 637 nm channel

class Reconstruction:
    def __init__(self, fps, v, theta=THETA, pixel_size=PIXEL_SIZE, mag=MAG, 
                 F_indexmismatch=F_INDEXMISMATCH, coeff405=COEFF405, coeff637=COEFF637, 
                 shift405=SHIFT_405, shift637=SHIFT_637,
                 polarity=0, wavelength=(405, 488, 637),
                 save_folder=None):
        # Initialize parameters
        self.theta = theta
        self.pixel_size = pixel_size
        self.mag = mag
        self.F_indexmismatch = F_indexmismatch
        self.coeff405 = coeff405
        self.coeff637 = coeff637
        self.shift405 = shift405
        self.shift637 = shift637
        self.fps = fps
        self.v = v
        self.polarity = polarity    
        self.wavelength = wavelength
        self.image = None
        self.transform_matrix = None
        self.profile = None
        self.background = None
        self.save_folder = save_folder
        self.result_image = None
        self.result_path = None

        if self.wavelength==405:
            self.F_indexmismatch = self.F_indexmismatch / self.coeff405
        elif self.wavelength==637:
            self.F_indexmismatch = self.F_indexmismatch / self.coeff637

    # def create_save_folder(self, image_path):
    #     # Create the result folder one level up from the image path
    #     self.save_folder = os.path.join(os.path.dirname(os.path.dirname(image_path)), 'results')
    #     if not os.path.exists(self.save_folder):
    #         os.makedirs(self.save_folder)

    def load_image(self, path):
        # Load image based on file extension
        if path.endswith('.fits'):
            image = fits.getdata(path).T
        elif path.endswith('.tif') or path.endswith('.tiff'):
            image = imread(path)
        else:
            raise ValueError('Unsupported file format')
                
        return cp.array(image).astype(cp.float32)

    def load_images(self, image_path, profile_path, background_path):
        # # Create the save folder
        # if self.save_folder is None:
        #     self.create_save_folder(image_path)
        
        # Load and flip the main image
        self.image = cp.flip(self.load_image(image_path), axis=1)
        # Load profile and background images
        self.profile = self.load_image(profile_path)
        self.background = self.load_image(background_path)

    def pre_process_images(self):
        # Pre-process images by normalizing with profile and subtracting background
        processed_profile = self.profile / cp.mean(self.profile)
        for i in range(self.image.shape[2]):
            self.image[:, :, i] = (self.image[:, :, i] - self.background.T) / processed_profile.T

            self.image = cp.clip(self.image, 0, None)
            self.image = 65535 * (self.image - cp.min(self.image)) / (cp.max(self.image) - cp.min(self.image))

    def get_3Dtransform_matrix(self):
        # Calculate the 3D transformation matrix
        scale_y = self.F_indexmismatch * np.cos(self.theta)
        scale_z = self.v / self.fps / (self.pixel_size / self.mag)

        M_zxscale = cp.array([[1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, scale_z, 0],
                              [0, 0, 0, 1]])

        M_yxscale = cp.array([[1, 0, 0, 0],
                              [0, scale_y, 0, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])

        M_yzshear = cp.array([[1, 0, 0, 0],
                              [0, 1, np.tan(self.theta), 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])

        self.transform_matrix = cp.dot(cp.dot(M_zxscale, M_yxscale), M_yzshear)

    def apply_affine_transform_3d(self):
        # Apply the affine transformation to the 3D image
        inverse_matrix_offset = cp.linalg.inv(self.transform_matrix.T)

        if self.polarity == 1:
            self.theta = -self.theta

        inverse_matrix = cp.linalg.inv(self.transform_matrix.T)
        output_shape = output_shape = [int(self.image.shape[i] * self.transform_matrix[i, i]) for i in range(3)] # Calculate size after affine transformation

        transformed_img = affine_transform(
            self.image, inverse_matrix[:3, :3], offset=-inverse_matrix_offset[:3, 3], 
            order=1,  # Interpolation method (1 is for linear interpolation)
            mode='constant',  # Value outside the image
            cval=0,  # Value for constant mode
            output_shape=tuple(output_shape)
        )

        return transformed_img
    
    def shift_image(self, transformed_img):
        # Shift the transformed image based on the wavelength
        if self.wavelength == 405:
            shift = self.shift405
        elif self.wavelength == 637:
            shift = self.shift637
        else:
            return transformed_img  # Do nothing and return the original transformed image

        matrix = cp.eye(4)
        matrix[:3, 3] = shift
        
        return affine_transform(transformed_img, matrix[:3, :3], offset=matrix[:3, 3])
        
    def image_3d_reconstraction(self, calibration=True, shift=True):
        # Perform 3D image reconstruction
        if calibration:
            self.pre_process_images()
 
        transformed_img = self.apply_affine_transform_3d()

        if shift:
            transformed_img = self.shift_image(transformed_img)

        self.result_image = transformed_img.get()  # Convert to NumPy array for saving
    
    def save_image(self, original_filename, save_tif=True):
        # Save the reconstructed image
        if self.save_folder is None:
            raise ValueError('Save folder is not set. Load an image first.')
        
        base_filename = os.path.splitext(os.path.basename(original_filename))[0]

        if save_tif:
            result_filename = f'{base_filename}_reconstructed.tif'
        else:
            result_filename = f'{base_filename}_reconstructed.npy'

        # Save the result
        self.result_path = os.path.join(self.save_folder, result_filename)
        if save_tif:
            tifffile.imsave(self.result_path, self.result_image)
        else: 
            np.save(self.result_path, self.result_image)

def reconstruction(fps, v, wavelength=(405, 488, 637), image_path=None, profile_path=None, background_path=None, 
                   polarity=0, calibration=True, save_tif=True, save_folder=None):
    print('Starting reconstruction process...')
    recon = Reconstruction(fps=fps, v=v, polarity=polarity, wavelength=wavelength, save_folder=save_folder)
    
    print('Loading images...')
    recon.load_images(image_path, profile_path, background_path)
    
    print('Calculating 3D transform matrix...')
    recon.get_3Dtransform_matrix()
    
    print('Applying 3D affine transformation...')
    recon.image_3d_reconstraction(calibration=calibration)
    
    print('Saving reconstructed image...')
    recon.save_image(image_path, save_tif)
    
    print('Reconstruction process completed.')
    return recon.result_path
