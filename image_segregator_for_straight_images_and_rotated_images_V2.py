'''
TODO :

1 (Must) . Put person's side profile also in the Known faces directory. 
            Left profile and Right profile also.
            
2 (Must) . Whenever person's face is not recognizes even after rotation means
            the person might be too zoomed in therefore we would zoom out the image
            by preprocessing and then again image would be passed to the algorithm.

2 (Optional) . Implement the hard voting algo.In results list if majority of votes are True
                then only consider the image to be correctly classified.Else if the false are
                in majority then do not consider that particular face.Consider that face as
                false positive.
'''


'''
                                                             REFERENCE SECTION 
MODEL USED : inception model
LOSS FUNCTION : triplet loss function
DISTANCE (tolerance calculation) : chi squared distance.

1 . The tolerance value suggests how tough we want our model to be. The less 
    tolerance value indicates we want the images which we are 100% sure about.
    
'''

import face_recognition
import os
import cv2
from scipy import ndimage



KNOWN_FACES_DIR = 'known_faces'
UNKNOWN_FACES_DIR = 'unknown_faces'
TOLERANCE = 0.45
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = 'hog'  # default: 'hog', other one can be 'cnn' - CUDA accelerated (if available) deep-learning pretrained model

# Returns (R, G, B) from name
def name_to_color(name):
    # Take 3 first letters, tolower()
    # lowercased character ord() value rage is 97 to 122, substract 97, multiply by 8
    color = [(ord(c.lower())-97)*8 for c in name[:3]]
    return color

# Returns resized image
def resize(original_img):
    img = cv2.resize(original_img, (1000, 700))
    return img


print('Loading known faces...')
known_faces = []
known_names = []

# We oranize known faces as subfolders of KNOWN_FACES_DIR
# Each subfolder's name becomes our label (name)
for name in os.listdir(KNOWN_FACES_DIR):

    # Next we load every file of faces of known person
    for filename in os.listdir(f'{KNOWN_FACES_DIR}/{name}'):

        # Load an image
        image = face_recognition.load_image_file(f'{KNOWN_FACES_DIR}/{name}/{filename}')
        # Get 128-dimension face encoding
        # Always returns a list of found faces, for this purpose we take first face only (assuming one face per image as you can't be twice on one image)
        try:
            encoding = face_recognition.face_encodings(image)[0]
    
            # Append encodings and name
            known_faces.append(encoding)
            known_names.append(name)
            
        except:
            print("The face is not suitable for fetching the encodings!Try with different face. ")

required_images = []
print('Processing unknown faces...')
# Now let's loop over a folder of faces we want to label
for filename in os.listdir(UNKNOWN_FACES_DIR):

    # Load image
    print(f'Filename {filename}', end='')
    image = face_recognition.load_image_file(f'{UNKNOWN_FACES_DIR}/{filename}')
    image = resize(image)

    '''
    This time we first grab face locations - we'll need them to draw boxes.
    Locations would basically contains all the coordinates/locations of all
    the faces it had detected in the particular image.Because our image may 
    contains the multiple faces because of multiple people.
    '''
    locations = face_recognition.face_locations(image, model=MODEL)
    '''
    Now since we know locations, we can pass them to face_encodings as second 
    argument.Encoding would be obtained for that particular face's location.
    Without that it will search for faces once again slowing down whole process
    '''
    encodings = face_recognition.face_encodings(image, locations)

    # We passed our image through face_locations and face_encodings, so we can modify it
    # First we need to convert it from RGB to BGR as we are going to work with cv2
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    

    # But this time we assume that there might be more faces in an image - we can find faces of dirrerent people
    print(f', found {len(encodings)} face(s)')
    
    '''
    This if loop checks if the length of the encodings is 0 or not.If 0 which means 
    the image might be rotated in some manner so we would rotate it once again and
    then again we would process the rotated image to our model
    '''
    
    if len(encodings) == 0:
        print('Image might be rotated loop executed')
        
        rotated_image = ndimage.rotate(image, 90,reshape=True)
        rotated_image = cv2.resize(rotated_image, (1000,700))        
        locations = face_recognition.face_locations(rotated_image, model=MODEL)
        encodings = face_recognition.face_encodings(rotated_image, locations)

        for face_encoding, face_location in zip(encodings, locations):
    
            # We use compare_faces (but might use face_distance as well)
            # Returns array of True/False values in order of passed known_faces
            results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
            # Since order is being preserved, we check if any face was found then grab index
            # then label (name) of first matching known face within a tolerance
            match = None
            if True in results:  # If at least one is true, get a name of first of found labels
                match = known_names[results.index(True)]
                print(f' - {match} from {results}')
    
                # Each location contains positions in order: top, right, bottom, left
                top_left = (face_location[3], face_location[0])
                bottom_right = (face_location[1], face_location[2])
    
                # Get color by name using our fancy function
                color = name_to_color(match)
    
                # Paint frame
                cv2.rectangle(rotated_image, top_left, bottom_right, color, FRAME_THICKNESS)
    
                # Now we need smaller, filled grame below for a name
                # This time we use bottom in both corners - to start from bottom and move 50 pixels down
                top_left = (face_location[3], face_location[2])
                bottom_right = (face_location[1], face_location[2] + 22)
    
                # Paint frame
                cv2.rectangle(rotated_image, top_left, bottom_right, color, cv2.FILLED)
    
                # Wite a name
                cv2.putText(rotated_image, match, (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), FONT_THICKNESS)
    
                required_images.append(filename)
        # Show image
        #cv2.imshow('rotated image',rotated_image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        
        rotated_image = ndimage.rotate(image, -90,reshape=True)
        rotated_image = cv2.resize(rotated_image, (1000,1000))
        locations = face_recognition.face_locations(rotated_image, model=MODEL)
        encodings = face_recognition.face_encodings(rotated_image, locations)

        for face_encoding, face_location in zip(encodings, locations):
    
            # We use compare_faces (but might use face_distance as well)
            # Returns array of True/False values in order of passed known_faces
            results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
            # Since order is being preserved, we check if any face was found then grab index
            # then label (name) of first matching known face within a tolerance
            match = None
            if True in results:  # If at least one is true, get a name of first of found labels
                match = known_names[results.index(True)]
                print(f' - {match} from {results}')
    
                # Each location contains positions in order: top, right, bottom, left
                top_left = (face_location[3], face_location[0])
                bottom_right = (face_location[1], face_location[2])
    
                # Get color by name using our fancy function
                color = name_to_color(match)
    
                # Paint frame
                cv2.rectangle(rotated_image, top_left, bottom_right, color, FRAME_THICKNESS)
    
                # Now we need smaller, filled grame below for a name
                # This time we use bottom in both corners - to start from bottom and move 50 pixels down
                top_left = (face_location[3], face_location[2])
                bottom_right = (face_location[1], face_location[2] + 22)
    
                # Paint frame
                cv2.rectangle(rotated_image, top_left, bottom_right, color, cv2.FILLED)
    
                # Wite a name
                cv2.putText(rotated_image, match, (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), FONT_THICKNESS)
    
                required_images.append(filename)
        # Show image
        #cv2.imshow('rotated image',rotated_image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
    
    else:
        print('Image is not rotated loop executed')
        for face_encoding, face_location in zip(encodings, locations):
    
            # We use compare_faces (but might use face_distance as well)
            # Returns array of True/False values in order of passed known_faces
            results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
    
            # Since order is being preserved, we check if any face was found then grab index
            # then label (name) of first matching known face within a tolerance
            match = None
            if True in results:  # If at least one is true, get a name of first of found labels
                match = known_names[results.index(True)]
                print(f' - {match} from {results}')
    
                # Each location contains positions in order: top, right, bottom, left
                top_left = (face_location[3], face_location[0])
                bottom_right = (face_location[1], face_location[2])
    
                # Get color by name using our fancy function
                color = name_to_color(match)
    
                # Paint frame
                cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)
    
                # Now we need smaller, filled grame below for a name
                # This time we use bottom in both corners - to start from bottom and move 50 pixels down
                top_left = (face_location[3], face_location[2])
                bottom_right = (face_location[1], face_location[2] + 22)
    
                # Paint frame
                cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
    
                # Wite a name
                cv2.putText(image, match, (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), FONT_THICKNESS)
                required_images.append(filename)
                
        # Show image
        #cv2.imshow('image',image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        
count = 0
name = os.listdir(f'{KNOWN_FACES_DIR}')
dir_name = name[0] 

if not os.path.exists(dir_name):    
    os.mkdir(dir_name)  

print("Total images found are : ",len(required_images))
    
for i in required_images:
    img = cv2.imread('unknown_faces/'+str(i))
    print('Saving image ',i)
    cv2.imwrite(dir_name+'/img_'+str(count)+'.jpg', img)
    count += 1

'''
FUTURE APPLICATIONS : 

    1. Sentdex reference video - 2 of the series.
    2. Grabing the picture of individuals from video by converting the video into series of images
        and then just detecting each person's face from the images which were converted from the video
        (reference script to do conversion : video_to_image.py).Time can be specified that by how much
        time interval the pictures should be taken from the video.
    3. Use 'cnn model' rather then 'hog model'.
    
'''
