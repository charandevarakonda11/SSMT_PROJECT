# Stress Tranfer in SSMT

# Word Alignments
- Get the word level Alignments using whisperX .
"/whisperx/English_Word_Align.py"

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

# Scaling Factor
- Find the scaling facotors for each utterance in the english dataset and save a json file
"/Scaling-Factor/save_scaling.py"

# TTS (MAIN)
- Git Clone and Set Up vakyansh TTS from [vakyansh-tts](https://github.com/Open-Speech-EkStep/vakyansh-tts.git)
- Modify the training Scripts in "src/glow_tts" replace with
- train_modified.py
- models_modified.py
- data_utils_temp.py
- common_modified.py

# TTS PDE MODIFIER
- While inference modify the Script in "utils/tts.py" replace with 
- tts_modified.py

# Evalutaion for Stress Transter
- Get the Word Level Alignments for hindi synthetic data generated from TTS using 
"/whisperx/Hindi_Word_Align.py"
- Get the Frame Level Stress Labels as the stress marks are availble for hindi words
- Train the Stress Detection Model from
"/Stress_Dection_Model/RFC"
"/Stress_Dection_Model/LPA"
"/Stress_Dection_Model/TDNN
