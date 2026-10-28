## Workshop -1 Adding Sunglasses to Your Passport Photo Using OpenCV
## Name : Priyadharshini G
## Reg no : 212224230209


# Sturdy-Octo-Disco-Adding-Sunglasses-for-a-Cool-New-Look

Sturdy Octo Disco is a fun project that adds sunglasses to photos using image processing.

Welcome to Sturdy Octo Disco, a fun and creative project designed to overlay sunglasses on individual passport photos! This repository demonstrates how to use image processing techniques to create a playful transformation, making ordinary photos look extraordinary. Whether you're a beginner exploring computer vision or just looking for a quirky project to try, this is for you!

## Features:
- Detects the face in an image.
- Places a stylish sunglass overlay perfectly on the face.
- Works seamlessly with individual passport-size photos.
- Customizable for different sunglasses styles or photo types.

## Technologies Used:
- Python
- OpenCV for image processing
- Numpy for array manipulations

## How to Use:
1. Clone this repository.
2. Add your passport-sized photo to the `images` folder.
3. Run the script to see your "cool" transformation!

## Applications:
- Learning basic image processing techniques.
- Adding flair to your photos for fun.
- Practicing computer vision workflows.

Feel free to fork, contribute, or customize this project for your creative needs!


## Program

```
# Import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
# Load the Face Image
faceImage = cv2.imread('08.JPG')
plt.imshow(faceImage[:,:,::-1]);plt.title("Face")
faceImage.shape
#resized_faceImage.shape
faceImage.shape
# Load the Sunglass image with Alpha channel
# (http://pluspng.com/sunglass-png-1104.html)
glassPNG = cv2.imread('sunglass.png',-1)
plt.imshow(glassPNG[:,:,::-1]);plt.title("glassPNG")
# Resize the image to fit over the eye region
glassPNG = cv2.resize(glassPNG,(220,60))
print("image Dimension ={}".format(glassPNG.shape))
# Separate the Color and alpha channels
glassBGR = glassPNG[:,:,0:3]
glassMask1 = glassPNG[:,:,3]
# Display the images for clarity
plt.figure(figsize=[15,15])
plt.subplot(121);plt.imshow(glassBGR[:,:,::-1]);plt.title('Sunglass Color channels');
plt.subplot(122);plt.imshow(glassMask1,cmap='gray');plt.title('Sunglass Alpha channel');
# Make a copy
#faceWithGlassesNaive = resized_faceImage.copy()
faceWithGlassesNaive = faceImage.copy()

# Replace the eye region with the sunglass image
faceWithGlassesNaive[210:270, 95:315]=glassBGR

plt.imshow(faceWithGlassesNaive[...,::-1])
# Make the dimensions of the mask same as the input image.
# Since Face Image is a 3-channel image, we create a 3 channel image for the mask
glassMask = cv2.merge((glassMask1,glassMask1,glassMask1))

# Make the values [0,1] since we are using arithmetic operations
glassMask = np.uint8(glassMask/255)

# Make a copy
faceWithGlassesArithmetic = faceImage.copy()

# Get the eye region from the face image
eyeROI= faceWithGlassesArithmetic[210:270,95:315]

# Use the mask to create the masked eye region
maskedEye = cv2.multiply(eyeROI,(1-  glassMask ))

# Use the mask to create the masked sunglass region
maskedGlass = cv2.multiply(glassBGR,glassMask)

# Combine the Sunglass in the Eye Region to get the augmented image
eyeRoiFinal = cv2.add(maskedEye, maskedGlass)

# Display the intermediate results
plt.figure(figsize=[20,20])
plt.subplot(131);plt.imshow(maskedEye[...,::-1]);plt.title("Masked Eye Region")
plt.subplot(132);plt.imshow(maskedGlass[...,::-1]);plt.title("Masked Sunglass Region")
plt.subplot(133);plt.imshow(eyeRoiFinal[...,::-1]);plt.title("Augmented Eye and Sunglass")
# Replace the eye ROI with the output from the previous section
faceWithGlassesArithmetic[210:270,95:315]=eyeRoiFinal

# Display the final result
plt.figure(figsize=[20,20]);
plt.subplot(121);plt.imshow(faceImage[:,:,::-1]); plt.title("Original Image");
plt.subplot(122);plt.imshow(faceWithGlassesArithmetic[:,:,::-1]);plt.title("With Sunglasses");


```


## Output

<img width="379" height="459" alt="image" src="https://github.com/user-attachments/assets/e6da8ea2-d6e8-4d45-b050-44054e9fc02e" />

<img width="570" height="298" alt="image" src="https://github.com/user-attachments/assets/1cd88a99-0043-4087-b36d-44f37ea4c323" />

<img width="1097" height="202" alt="image" src="https://github.com/user-attachments/assets/5a2e7598-6a44-47e4-8190-0b435938380b" />

<img width="377" height="454" alt="image" src="https://github.com/user-attachments/assets/29844d1b-69c9-4030-a6b4-f95c86d49486" />

<img width="1094" height="161" alt="image" src="https://github.com/user-attachments/assets/11240d0e-cd3f-405a-a39b-28e88c44d6c8" />

<img width="1110" height="661" alt="image" src="https://github.com/user-attachments/assets/4b4122a6-7211-4f81-8132-8e9b5519e701" />



## Result 

The result of this project is an image of a person wearing sunglasses. Using Python and OpenCV, the sunglasses image is resized and placed over the person's eyes.

