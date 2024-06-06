from math import log

import matplotlib.pyplot as plt
from tifffile import imread
from skimage.transform import pyramid_gaussian
from skimage.metrics import structural_similarity as ssim
import numpy as np
import math, sys, os

# import SimpleITK as sitk

# has_tigre = True
# try:
#     import tigre
#     print("Tigre detected")
# except:
#     has_tigre = False
#     print("Tigre not detected")

# has_cil = True
# try:
#     # from cil.utilities.jupyter import islicer
#     from cil.processors import TransmissionAbsorptionConverter
#     from cil.utilities.display import show_geometry, show2D
#     from cil.utilities.jupyter import islicer, link_islicer
#     from cil.recon import FBP, FDK
#     from cil.plugins.astra.processors.FDK_Flexible import FDK_Flexible

#     # GD_LS
#     from cil.optimisation.algorithms import GD
#     from cil.plugins.astra import ProjectionOperator
#     from cil.optimisation.functions import LeastSquares
#     from cil.processors import Slicer
#     from cil.plugins.ccpi_regularisation.functions import FGP_TV

#     # SPDHG
#     from cil.optimisation.algorithms import SPDHG
#     from cil.optimisation.operators import BlockOperator
#     from cil.optimisation.functions import BlockFunction, L2NormSquared

    
#     print("CIL detected")
# except:
#     has_cil = False
#     print("CIL not detected")
    
    
from gvxrPython3 import gvxr
# from gvxrPython3.JSON2gVXRDataReader import *


if os.name == 'nt':
    #base_dir = "C:/Users/snn23kfl/CylinderData/"
    base_dir ="C:/Users/snn23kfl/SimData"
    #base_dir = "C:/Users/snn23kfl/missing_angles_data/"
    base_dir = "C:/Users/snn23kfl/missing_angles_data/"
else:
    base_dir = "/run/media/fpvidal/DATA/CT/2024/Bangor"

voxel_size = [0.009996474264487798, 0.009996474264487798, 0.009996474264487798]

vmin=0.0
vmax=1.5




# # Y = -192 # mm

# # binning = 2

# # geometry_filename = data_path + '/geom.csv'


# # if binning == 1:
# #     radiographies_path = data_path + '/Proj'
# #     output_filename = data_path + '/electric_cable.tif'
# # else:
# #     old_radiographies_path = data_path + '/Proj'
# #     radiographies_path = data_path + '/Proj_bin' + str(binning)
# #     output_filename = data_path + '/electric_bin' + str(binning) + ".tif"


x_default = 6*[0.0]

elements = ["Ti", "Al", "V"];
default_weights = [
           0.9,
           0.06,
           0.04
]
default_weights = np.array(default_weights) / np.sum(default_weights)

default_density = 4.43

figsize = (15, 17)

# # default_cable_insulation_density = (1.1 + 1.35) / 2 # Flexible PVC


# # ref_image = None
# # selected_angles = None
# # indices = None
# # x_current = None
# # plot_directory = None
# # best_fitness = sys.float_info.max
# # fitness_set = []
# # counter = 1


# def getRuntime(start, stop):
#     return stop - start, "sec"

# # def rebin(img, new_shape):
# #     """Rebin 2D array arr to shape new_shape by averaging."""
# #     shape = (new_shape[0], img.shape[0] // new_shape[0],
# #              new_shape[1], img.shape[1] // new_shape[1])
# #     return img.reshape(shape).mean(-1).mean(1)


def plotSpectrum(k, f, fname=None, xlim=[0,200], figsize=(20,10)):

    plt.figure(figsize=figsize)

    plt.bar(k, f / f.sum()) # Plot the spectrum
    plt.xlabel('Energy in keV')
    plt.ylabel('Probability distribution of photons per keV')
    plt.title('Photon energy distribution')

    plt.xlim(xlim)

    plt.tight_layout()

    if fname is not None:
        plt.savefig(fname + '.pdf', bbox_inches = 'tight')
        plt.savefig(fname + '.png', bbox_inches = 'tight')


    
def getReference(experimental_img_paths, angles_in_deg, number_of_angles, white_level, downscale: int=1):

    images = []
    angles = []
    indices = []
    
    for i in range(number_of_angles):

        if number_of_angles == 1:
            index = 0
        else:
            # index = round((i + 1) / number_of_angles * (len(angles_in_deg) // 2))
            index = round((i + 1) / number_of_angles * (len(angles_in_deg) - 1))
        
        projection = imread(experimental_img_paths[index]).astype(np.single)
        
        if downscale <= 1:
            images.append(projection)
        else:
            pyramid = tuple(pyramid_gaussian(projection, downscale=downscale, channel_axis=None))
            images.append(pyramid[1])
            
        angles.append(angles_in_deg[index])
        indices.append(index)

    ncols = images[-1].shape[1]
    images = np.array(images)
    images /= white_level
    
    return -np.log(images).astype(np.single), np.array(angles), np.array(indices)

    #images *= 255
    #images[images>255]=255
    #return np.round(images).astype(np.uint8), np.array(angles), np.array(indices)

def getXrayImage(x, take_screenshot=False):

    global screenshot, selected_angles, x_current
    screenshot = []

    test_image = np.zeros((len(selected_angles), gvxr.getDetectorNumberOfPixels()[1], gvxr.getDetectorNumberOfPixels()[0]), dtype=np.single)
    total_incident_energy = gvxr.getTotalEnergyWithDetectorResponse()
    
    backup_density = gvxr.getDensity("steel-smoothed")
    
    if len(x) == 6:
        transformation = np.copy(x)
    elif len(x) == 4:
        transformation = [x[0], x[1], 0, x[2], x[3], 0]

    else:
        transformation = np.copy(x_current)

#     elif len(x) == len(default_weights) + 1:
#         transformation = np.copy(x_current)

#         backup_density = gvxr.getDensity("steel-smoothed")
        
#         temp_weight = []
#         for i in range(len(default_weights)):
#             temp_weight.append(x[i+1] * default_weights[i])
            
#         gvxr.setMixture("steel-smoothed",
#                         elements,
#                         np.array(temp_weight) / np.sum(temp_weight))
        
#         gvxr.setDensity("steel-smoothed", x[0] * default_density, "g/cm3")
        
        
        
        
    label = "root"
    for i, rot_angle in enumerate(selected_angles):
        matrix_backup = gvxr.getLocalTransformationMatrix(label)

        # Sample position on turntable
        gvxr.translateNode(label, transformation[0], transformation[1], transformation[2], "mm")
        
        # Turntable position
        gvxr.translateNode(label, transformation[3], transformation[4], transformation[5], "mm")
        
        # rotation
        #gvxr.rotateNode(label, rot_angle, 0, 0, 1)
        gvxr.rotateNode(label, float(rot_angle), 0.0, 0.0, 1.0) 
        gvxr.translateNode(label, -transformation[3], -transformation[4], -transformation[5], "mm")

        # Compute an X-ray image
        # We convert the array in a Numpy structure and store the data using single-precision floating-point numbers.
        xray_image = np.array(gvxr.computeXRayImage(), dtype=np.single) / total_incident_energy
        test_image[i] = -np.log(xray_image).astype(np.single)

        #test_image[i] = np.round(255 * xray_image).astype(np.uint8)
        
        

        # Update the visualisation window
        if take_screenshot:

            gvxr.displayScene()        
            screenshot.append(gvxr.takeScreenshot())
    
        gvxr.setLocalTransformationMatrix("root", matrix_backup)
        
    if len(x) == len(default_weights) + 1:
        gvxr.setMixture("sample-smoothed",
                        elements,
                        default_weights)

        gvxr.setDensity("sample-smoothed", backup_density, "g/cm3")
    
    return test_image

def compareMAE(ref, test):
    return np.abs(ref - test).mean()

def compareMSE(ref, test):
    return np.square(ref - test).mean()

def compareRMSE(ref, test):
    return math.sqrt(compareMSE(ref, test))

def compareZNCC(ref, test):
    if ref.std() < 1e-4 or test.std() < 1e-4:
        return 1e-4
    
    return np.mean(((ref - ref.mean()) / ref.std()) * ((test - test.mean()) / test.std()))

def compareSSIM(ref, test):
    
    channel_axis = None
    if len(ref.shape) == 3:
        channel_axis = 0
        
    return ssim(ref, test, channel_axis=channel_axis, data_range=(ref.max()-ref.min()))

def displayResult(x, figsize=(15, 4), fname=None, crop=False):
    global screenshot, bbox
    test_image = getXrayImage(x, True)

    roi_size = [ref_image.shape[1], ref_image.shape[2]]
    offset = [0, 0]
    
    if crop:

        roi_size[0] = round(0.9 * roi_size[0])
        roi_size[1] = round(0.7 * roi_size[1])

        offset = [
            (ref_image.shape[1] - roi_size[0]) // 2,
            (ref_image.shape[2] - roi_size[1]) // 2
        ]
    
    ref_tmp = np.array(ref_image, dtype=np.single)
    test_tmp = np.array(test_image, dtype=np.single)

    MAE = 0.0
    RMSE = 0.0
    SSIM = 0.0
    ZNCC = 0.0
    
    SSIM = 0.0
    
    for img1, img2 in zip(ref_tmp, test_tmp):
        
        tmp1 = img1[offset[0]:roi_size[0], offset[1]:roi_size[1]]
        tmp2 = img2[offset[0]:roi_size[0], offset[1]:roi_size[1]]
                    
        MAE += compareMAE(tmp1, tmp2);
        RMSE += math.sqrt(compareMSE(tmp1, tmp2));
        SSIM += ssim(tmp1, tmp2, data_range=(tmp1.max() - tmp1.min()));
        
        tmp1 -= tmp1.mean()
        tmp1 /= tmp1.std()

        tmp2 -= tmp2.mean()
        tmp2 /= tmp2.std()
        
        ZNCC += 100 * (tmp1 * tmp2).mean()
        
    MAE /= ref_image.shape[0]
    RMSE /= ref_image.shape[0]
    SSIM /= ref_image.shape[0]
    ZNCC /= ref_image.shape[0]
        
    fig, axs = plt.subplots(len(ref_image), 4, figsize=figsize, squeeze=False)
    plt.suptitle("Overall ZNCC=" + "{:.4f}".format(ZNCC) + "%\n" +
                "Overall MAE=" + "{:.4f}".format(MAE) + "\n" +
                "Overall RMSE=" + "{:.4f}".format(RMSE) + "\n" +
                "Overall SSIM=" + "{:.4f}".format(SSIM))

    for index in range(len(ref_image)):
        axs[index][0].imshow(screenshot[index])
        axs[index][1].imshow(ref_image[index], cmap="gray", vmin=vmin, vmax=vmax)
        axs[index][2].imshow(test_image[index], cmap="gray", vmin=vmin, vmax=vmax)
        im = axs[index][3].imshow((ref_image[index].astype(np.single) - test_image[index].astype(np.single)), cmap="gray", vmin=-vmax, vmax=vmax)
        axs[index][0].set_title("Image: " + str(indices[index]) + "\nRotation angle: "  + "{:.4f}".format(selected_angles[index]) + "$^\circ$")
        axs[index][1].set_title("Experimental radiograph")
        axs[index][2].set_title("Simulated radiograph")
        axs[index][3].set_title("Error map")
        
        axs[index][1].set_xlim((offset[1], offset[1] + roi_size[1]))
        axs[index][2].set_xlim((offset[1], offset[1] + roi_size[1]))
        axs[index][3].set_xlim((offset[1], offset[1] + roi_size[1]))

        axs[index][1].set_ylim((offset[0], offset[0] + roi_size[0]))
        axs[index][2].set_ylim((offset[0], offset[0] + roi_size[0]))
        axs[index][3].set_ylim((offset[0], offset[0] + roi_size[0]))

        
    plt.tight_layout()

    if fname is not None:
        plt.savefig(fname + '.pdf', bbox_inches = 'tight')
        plt.savefig(fname + '.png', bbox_inches = 'tight')
        
def fitnessRMSE(x):
    global ref_image, best_fitness, fitness_set, counter, plot_directory

    test_image = getXrayImage(x)
        
    fitness_value = 0.0
    
    for ref, test in zip(ref_image, test_image):
        fitness_value += compareRMSE(ref, test)

    fitness_value /= ref_image.shape[0]

    if best_fitness > fitness_value:
        fitness_set.append([counter, fitness_value])
        best_fitness = fitness_value
        displayResult(x, figsize)
        plt.savefig(plot_directory + "/plot_" + str(counter).zfill(4) + ".png")
        plt.close()

    counter += 1

    return fitness_value


# def fitnessRMSE(x):
#     global ref_image, best_fitness, fitness_set, counter, plot_directory

#     if 'best_fitness' not in globals():
#         best_fitness = sys.float_info.max
#     if 'fitness_set' not in globals():
#         fitness_set = []
#     if 'counter' not in globals():
#         counter = 1
#     if 'plot_directory' not in globals():
#         plot_directory = "."

#     test_image = getXrayImage(x)
        
#     fitness_value = 0.0
    
#     for ref, test in zip(ref_image, test_image):
#         fitness_value += compareRMSE(ref, test)

#     fitness_value /= ref_image.shape[0]

#     if best_fitness > fitness_value:
#         fitness_set.append([counter, fitness_value])
#         best_fitness = fitness_value
#         displayResult(x, figsize)
#         plt.savefig(plot_directory + "/plot_" + str(counter).zfill(4) + ".png")
#         plt.close()

#     counter += 1

#     return fitness_value


def fitnessMAE(x):
    global ref_image, best_fitness, fitness_set, counter, plot_directory

    test_image = getXrayImage(x)
        
    fitness_value = 0.0
    
    for ref, test in zip(ref_image, test_image):
        fitness_value += compareMAE(ref, test)
    
    fitness_value /= ref_image.shape[0]

    if best_fitness > fitness_value:
        fitness_set.append([counter, fitness_value])
        best_fitness = fitness_value
        displayResult(x, figsize)
        plt.savefig(plot_directory + "/plot_" + str(counter).zfill(4) + ".png")
        plt.close()

    counter += 1

    return fitness_value

def fitnessZNCC(x):
    global ref_image, best_fitness, fitness_set, counter, plot_directory

    test_image = getXrayImage(x)
        
    fitness_value = 0.0
    
    for ref, test in zip(ref_image, test_image):
        fitness_value += 1.0 / compareZNCC(ref, test)
    
    fitness_value /= ref_image.shape[0]

    if best_fitness > fitness_value:
        fitness_set.append([counter, fitness_value])
        best_fitness = fitness_value
        displayResult(x, figsize)
        plt.savefig(plot_directory + "/plot_" + str(counter).zfill(4) + ".png")
        plt.close()
    counter += 1

    return fitness_value




# # # Perform reconstruction using a gVirtualXRay configuration file

# # def reconstructFBPWithCIL(data, ig, verbose):
# #     if verbose > 0: print("Parallel beam detected")

# #     if has_tigre:
# #         if verbose > 0: print("Backend: Tigre")
# #         reconstruction:ImageData | None = FBP(data, ig).run()
# #     else:
# #         if verbose > 0: print("Backend: Astra-Toolbox")
# #         reconstruction:ImageData | None = FBP(data, ig, backend="astra").run()

# #     return reconstruction

# # def reconstructFDKWithCIL(data, ig, verbose):
# #     if verbose > 0: print("Cone beam detected")

# #     if has_tigre:
# #         if verbose > 0: print("Backend: Tigre")
# #         data.geometry.set_angles(data.geometry.angles)
# #         reconstruction:ImageData | None = FDK(data, ig).run()
# #     else:
# #         if verbose > 0: print("Backend: Astra-Toolbox")
# #         fbk = FDK_Flexible(ig, data.geometry)
# #         fbk.set_input(data)
# #         reconstruction:ImageData | None = fbk.get_output()
    
# #     return reconstruction

# # def reconstruct(JSON_fname, ref_size, voxel_size, verbose=0):
    
# #     data = None
# #     reconstruction = None
    
# #     if gvxr.isUsingParallelSource():
# #         source_shape = "PARALLELBEAM";
# #     elif gvxr.isUsingConeBeam():
# #         source_shape = "CONEBEAM";
# #     else:
# #         ValueError("Unknown beam shape.")

# #     if verbose > 0:
# #         print("Source shape:", source_shape)

# #     reader = JSON2gVXRDataReader(file_name=JSON_fname)
# #     data = reader.read()

# #     print("data.geometry", data.geometry)

# #     if has_tigre:
# #         data.reorder(order='tigre')
# #     else:
# #         data.reorder("astra")

# #     ig = data.geometry.get_ImageGeometry()

# #     ig.voxel_num_x = ref_size[0]
# #     ig.voxel_num_y = ref_size[1]
# #     ig.voxel_num_z = ref_size[2]

# #     ig.voxel_size_x = voxel_size[0]
# #     ig.voxel_size_y = voxel_size[1]
# #     ig.voxel_size_z = voxel_size[2]

# #     data_corr = TransmissionAbsorptionConverter(white_level=data.max(), min_intensity=0.000001)(data)

# #     if type(source_shape) == str:

# #         if source_shape.upper() == "PARALLELBEAM" or source_shape.upper() == "PARALLEL":
# #             reconstruction:ImageData | None = reconstructFBPWithCIL(data_corr, ig, verbose)

# #         elif source_shape.upper() == "POINTSOURCE" or source_shape.upper() == "POINT" or source_shape.upper() == "CONE" or source_shape.upper() == "CONEBEAM":
# #             reconstruction:ImageData | None = reconstructFDKWithCIL(data_corr, ig, verbose)

# #         else:
# #             raise ValueError("Unknown source shape:" + source_shape)

# #     elif type(source_shape) == type([]):
# #         if source_shape[0].upper() == "FOCALSPOT":
# #             reconstruction:ImageData | None = reconstructFDKWithCIL(data_corr, ig, verbose)

# #         else:
# #             raise ValueError("Unknown source shape:" + source_shape)

# #     else:
# #         raise ValueError("Unknown source shape:" + source_shape)    

# #     return data, reconstruction


# # def saveReconstruction(fname, reconstruction, voxel_size=None):
# #     image = sitk.GetImageFromArray(reconstruction.as_array())
# #     # image.SetOrigin((0, 0, 0))
    
# #     if voxel_size:
# #         image.SetSpacing([voxel_size[0], voxel_size[0], voxel_size[1]])

# #     writer = sitk.ImageFileWriter()
# #     writer.SetFileName(fname)
# #     writer.UseCompressionOn()
# #     writer.Execute(image)    
