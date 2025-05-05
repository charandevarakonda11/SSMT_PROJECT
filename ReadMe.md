# Stress Tranfer in SSMT

# Word Alignments
- Get the word level Alignments using whisperX .

# Feature Extraction 
- Save the features (f0+energy) and (f0+energy+sdc+mfcc) for each frame for the english data 
"/Feature_Extraction/save_data.py"

# Stress Detection Model
- Train the Models (RFC,LPA, TDNN) using the saved features 
"/Stress_Dection_Model/RFC"
"/Stress_Dection_Model/LPA"
"/Stress_Dection_Model/TDNN"

# Machine Translation and Alignment
- Translate the text obtained from ASR to hindi language  
"/Machine_Translation/translate.py"

- Obtain the mappings between the english and hindi words
"/Machine_Translation/simAlign.py"
