
import multiprocessing


import numpy as np
import cupy as cp

import random
import pygad
import os

import cv2

from PIL import Image as im
from SSIM_PIL import compare_ssim
import albumentations as al

import time



OBJECTS = 1000
EPOCHS = 10
GEN_SIZE = 100



SCALEMIN = 128
SCALEMAX = 4096
TARGET = im.open("Target/target.png")
TARGET_FIXED = np.array(TARGET)
TARGET_GPU = cp.array(TARGET)

TARGETX = TARGET.size[0]
TARGETY = TARGET.size[1]

bestResult = im.open("Result/imgBest.png")
oldBest = im.new("RGBA",(TARGETX, TARGETY), (0,0,0,0))
bestDiff = -10000000
bestFitness = -100000000
objectsPlaced = 0

def extend_matrix(arr):
    # Get the dimensions of the input array
    x, y, _ = arr.size
    
    # Create a new array of zeros with the desired shape
    new_arr = np.zeros((TARGETX, TARGETY, 4), dtype=arr.dtype)
    
    # Compute the x and y ranges for the input array
    x_range = slice((TARGETX - x) // 2, (TARGETX + x) // 2)
    y_range = slice((TARGETY - y) // 2, (TARGETY + y) // 2)
    
    # Copy the input array into the center of the new array
    new_arr[x_range, y_range, :] = arr
    
    return new_arr

def RMSE(predictions):
    #rmse = np.sqrt(np.mean((predictions-targets)**2))
    #return rmse
    #x = predictions - TARGET_GPU
    #print(x)
    rmse = cp.linalg.norm(predictions - TARGET_GPU) / cp.sqrt(len(TARGET_GPU))
    rmse_cpu = cp.asnumpy(rmse)
    return rmse_cpu


def fitness_func(ga_instance, solution, solution_idx):
    global bestDiff
    global oldBest
   

    #denormalize genes
    #oldBest = skimage.io.imread("Result/imgBest.png")
    num = int(solution[0] * 600)
    num = abs(num)
    #imgNumber
    imgNumber = str(num)
    #Xpos
    xPos = int(solution[1] * TARGETX)
    #Ypos
    yPos = int(solution[2] * TARGETX)
    #Scale
    scale = int(solution[3] * (SCALEMAX - SCALEMIN) + SCALEMIN)
    #Rotation
    rotation = int(solution[4] * 360)
    opacity = int(solution[5] * 255)

    #Load and transform image 
    toPlace = im.open(f"spotArray/{imgNumber}.png")


    image = np.array(toPlace)

    
    transform = al.Compose([al.LongestMaxSize(max_size = scale, p=1)])
 
    alToPlace = transform(image=image)['image']

    result = oldBest

    toPlace = im.fromarray(alToPlace)
    toPlace = toPlace.rotate(rotation, expand=True)
    
    toPlace.putalpha(opacity)

    result.paste(toPlace, (xPos, yPos), toPlace)

    resultGPU = cp.asarray(result)
    
    rmse = RMSE(resultGPU)
    diff =  np.array(rmse) # diff
    #result = np.array(result)
    #diff = compare_ssim(resultGPU, TARGET_GPU, tile_size = 7)
    #
    #rmse = cp.linalg.norm(resultGPU - TARGET_GPU) / cp.sqrt(len(TARGET_GPU))
    #diff =  cp.asnumpy(rmse) # diff
    #diff = -compare_ssim(result, TARGET) 
    #rmse = np.linalg.norm(result - TARGET_FIXED) / np.sqrt(len(TARGET_FIXED))
    #diff =  np.array(rmse) # diff
    #return -rmse
    return -diff
    #diff = -(RMSE(resultGPU))

    


def on_gen(ga_instance):
    print("Generation : ", ga_instance.generations_completed)
    print("Fitness of the best solution :", ga_instance.best_solution()[1])

def main(): #we do a lil parrallel processing
    global bestDiff
    global oldBest
    global bestFitness
    global objectsPlaced

    for i in range(1000):
        print(f"TRIAL {i} START")
        start = time.time()
        ga_instance = pygad.GA(num_generations=5,
                                num_parents_mating=200,
                                parent_selection_type="sus",
                                fitness_func=fitness_func,
                                sol_per_pop=1000,
                                num_genes=6,
                                init_range_low=0,
                                init_range_high=1,
                                mutation_type="random",
                                mutation_by_replacement=True,
                                random_mutation_min_val=-0.9,
                                random_mutation_max_val=0.9,
                                mutation_num_genes=4,
                                on_generation=on_gen,
                                gene_space = [[*np.arange(0, 1, 0.05)],
                                              [*np.arange(0, 1, 0.05)],
                                              [*np.arange(0, 1, 0.05)],
                                              [*np.arange(0, 1, 0.05)],
                                              [*np.arange(0, 1, 0.05)],
                                              [*np.arange(0, 1, 0.05)]],
                                parallel_processing=("process", 4)
                                )
            
            #Plug into pygad to start optimisation, need optimization funciton
    
        ga_instance.run()
        solution, solution_fitness, solution_idx = ga_instance.best_solution()



        if True:
        #if solution_fitness > bestFitness:
            

            #denormalize genes
            #oldBest = skimage.io.imread("Result/imgBest.png")
            num = int(solution[0] * 600)
            num = abs(num)
            #imgNumber
            imgNumber = str(num)
            #Xpos
            xPos = int(solution[1] * TARGETX)
            #Ypos
            yPos = int(solution[2] * TARGETX)
            #Scale
            scale = int(solution[3] * (SCALEMAX - SCALEMIN) + SCALEMIN)
            #Rotation
            rotation = int(solution[4] * 360)
            opacity = int(solution[5] * 255)


            #Load and transform image 
            toPlace = im.open(f"spotArray/{imgNumber}.png")
            image = np.array(toPlace)
            
            transform = al.Compose([al.LongestMaxSize(max_size = scale, p=1)])
            
            alToPlace = transform(image=image)['image']
            result = oldBest
            toPlace = im.fromarray(alToPlace)
            toPlace.putalpha(opacity)
            toPlace = toPlace.rotate(rotation, expand=True, fillcolor=(0,0,0,0))

            
            result.paste(toPlace, (xPos, yPos), toPlace)
            oldBest = result

            objectsPlaced = objectsPlaced + 1

            result.save("Result/imgBest.png", "PNG")

            if solution_fitness > bestFitness:
                bestFitness = solution_fitness
                result.save(f"Result/History/{solution_fitness}.png", "PNG")

            print(f"TRIAL {i} SUCCESSFUL")
            print(f"OBJECTS PLACED:{objectsPlaced} | BEST FITNESS: {bestFitness}")
            stop = time.time()
            print(f"TIME ELAPSED:{stop - start}")
        else:
            print(f"TRIAL {i} FAILED")
            print(f"OBJECTS PLACED:{objectsPlaced} | BEST FITNESS: {bestFitness}")
            stop = time.time()
            print(f"TIME ELAPSED:{stop - start}")

if __name__ == '__main__': 
    main()

