import yaml
import cv2
import torch
from torchvision import transforms
from mtcnn import MTCNN
from model import MobileFacenet
from preprocess import align_face

dict_label = {0: 'Inbar Lavi', 1: 'Alvaro Morte', 2: 'Madelaine Petsch', 3: 'Tom Hiddleston', 
              4: 'Tom Holland', 5: 'Jimmy Fallon', 6: 'Wentworth Miller', 7: 'Lindsey Morgan', 
              8: 'amber heard', 9: 'Katherine Langford', 10: 'Zac Efron', 11: 'Lionel Messi', 
              12: 'Amanda Crew', 13: 'Lili Reinhart', 14: 'Chris Hemsworth', 15: 'Neil Patrick Harris', 
              16: 'Ursula Corbero', 17: 'Anne Hathaway', 18: 'Maria Pedraza', 19: 'Penn Badgley', 
              20: 'Katharine Mcphee', 21: 'Pedro Alonso', 22: 'Jake Mcdorman', 23: 'Marie Avgeropoulos', 
              24: 'Selena Gomez', 25: 'Cristiano Ronaldo', 26: 'Emma Watson', 27: 'Megan Fox', 28: 'Nadia Hilker', 
              29: 'Irina Shayk', 30: 'Emilia Clarke', 31: 'jeff bezos', 32: 'Dominic Purcell', 33: 'barack obama', 
              34: 'Richard Harmon', 35: 'camila mendes', 36: 'Emma Stone', 37: 'Gwyneth Paltrow', 38: 'Robert De Niro',
              39: 'melissa fumero', 40: 'Chris Evans', 41: 'Johnny Depp', 42: 'Natalie Portman', 43: 'Rami Malek', 
              44: 'ellen page', 45: 'Brie Larson', 46: 'Morena Baccarin', 47: 'grant gustin', 48: 'Jeremy Renner', 
              49: 'Millie Bobby Brown', 50: 'Rihanna', 51: 'Tom Cruise', 52: 'elon musk', 53: 'Anthony Mackie',
              54: 'Bobby Morley', 55: 'gal gadot', 56: 'Shakira Isabel Mebarak', 57: 'Ben Affleck', 58: 'Zendaya',
              59: 'Rebecca Ferguson', 60: 'Keanu Reeves', 61: 'Dwayne Johnson', 62: 'elizabeth olsen', 
              63: 'Brenton Thwaites', 64: 'Jessica Barden', 65: 'Brian J. Smith', 66: 'Jennifer Lawrence', 
              67: 'Tom Hardy', 68: 'scarlett johansson', 69: 'Christian Bale', 70: 'margot robbie', 71: 'Jason Momoa', 
              72: 'Alex Lawther', 73: 'tom ellis', 74: 'Danielle Panabaker', 75: 'Eliza Taylor', 76: 'Miley Cyrus', 
              77: 'Henry Cavil', 78: 'Sarah Wayne Callies', 79: 'Stephen Amell', 80: 'Zoe Saldana', 81: 'Robert Downey Jr', 
              82: 'Bill Gates', 83: 'Taylor Swift', 84: 'Tuppence Middleton', 85: 'Chris Pratt', 86: 'Adriana Lima', 
              87: 'Andy Samberg', 88: 'Alexandra Daddario', 89: 'Avril Lavigne', 90: 'kiernen shipka', 91: 'Hugh Jackman',
              92: 'Maisie Williams', 93: 'Mark Zuckerberg', 94: 'Morgan Freeman', 95: 'Sophie Turner',
              96: 'Elizabeth Lail', 97: 'Mark Ruffalo', 98: 'Natalie Dormer', 99: 'barbara palvin', 
              100: 'Logan Lerman', 101: 'Josh Radnor', 102: 'Krysten Ritter', 103: 'alycia dabnem carey', 
              104: 'Leonardo DiCaprio'}

def inferal_video(video_path):
    with open('config.yml', 'r') as file:
        config = yaml.safe_load(file)

    checkpoint_path = config['BEST_MODEL']
    H = config['H']
    W = config['W']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = MobileFacenet()
    net = net.to(device) 

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    net.load_state_dict(ckpt['net_state_dict'], strict=True)
    net.eval()

    # Initialize MTCNN for face detection
    detector = MTCNN()

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    print("Processing video...")
    if not cap.isOpened():
        print("Error opening video file")
        return "Error opening video file"

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object
    out = cv2.VideoWriter('assests/output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces in the frame
        faces = detector.detect_faces(frame)
        for face in faces:
            box = face['box']
            landmarks = face['keypoints']

            # Crop the face from the frame
            x, y, width, height = box
            face_image = frame[y:y+height, x:x+width]

            # Adjust landmark coordinates according to the cropped region
            adjusted_landmarks = {
                'left_eye': (landmarks['left_eye'][0] - x, landmarks['left_eye'][1] - y),
                'right_eye': (landmarks['right_eye'][0] - x, landmarks['right_eye'][1] - y),
                'nose': (landmarks['nose'][0] - x, landmarks['nose'][1] - y),
                'mouth_left': (landmarks['mouth_left'][0] - x, landmarks['mouth_left'][1] - y),
                'mouth_right': (landmarks['mouth_right'][0] - x, landmarks['mouth_right'][1] - y)
            }

            aligned_face = align_face(face_image, adjusted_landmarks, H, W)
            aligned_face = (aligned_face - 127.5) / 128
            aligned_face = transforms.ToTensor()(aligned_face)
            aligned_face = aligned_face.unsqueeze(0).to(device).float()  

            logits = net(aligned_face)
            _, predicted = torch.max(logits, 1)
            text_label = dict_label[predicted.item()]
            print("Label:", text_label)

            # Draw rectangle and label on the frame
            cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 2)
            cv2.putText(frame, text_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Write the frame into the output video
        out.write(frame)

    # Release everything when job is finished
    cap.release()
    out.release()
    return "Video inference completed and saved as output.avi"