import argparse
from preprocess import preprocess_face
from train import train
from inferal import inferal  # Assuming you have an inferal function in inferal.py
from inferal_video import inferal_video  # Assuming you have an inferal_video function in inferal_video.py
def main():
    parser = argparse.ArgumentParser(description="Run different modes of the program")
    parser.add_argument('--mode', choices=['process', 'train', 'inferal', 'inferal_video'],
                        help="Mode to run: process, train, inferal")
    parser.add_argument('--path', type=str, help="Path to the image for inferal mode")

    args = parser.parse_args()

    if args.mode == 'process':
        preprocess_face()
    elif args.mode == 'train':
        train()
    elif args.mode == 'inferal':
        if not args.path:
            print("Please provide the path to the image using --path")
        else:
            inferal(args.path)
    elif args.mode == 'inferal_video':
        if not args.path:
            print("Please provide the path to the video using --path")
        else:
            inferal_video(args.path)
if __name__ == "__main__":
    main()