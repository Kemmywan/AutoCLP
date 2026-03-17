import os 

if __name__ == "__main__":
    for i in range(50):
        os.system(f"python utils/extract_dialogue.py --lengthMin={i*100} --lengthMax={(i+1)*100} --total=20 --save=T")

