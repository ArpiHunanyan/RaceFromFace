# RaceFromFace

## Clone the repository
> git clone https://github.com/ArpiHunanyan/RaceFromFace.git

## Data sets
* download train and val folder by using this link and put unziped folderds in __Data__ folder
https://drive.google.com/file/d/1Z1RqRo0_JiavaZw2yzZG6WETdZQ8qX86/view?usp=sharing

* download train_masked.zip  and val_masked.zip from this drive and put unziped folders  in __Data__ folder
https://drive.google.com/drive/folders/1IhAK3ne39pXXk1YwdjindBYm9U5uAQu9?usp=sharing

## Large model
* download 4.ResNet50.zip and put unziped folder in __Model__ folder
https://drive.google.com/drive/folders/15wFMpId_W7SYn3_b_pg_jSHAvZwatjUv?usp=sharing

## Jupiter Notbooke 
__Codes.ipynb__ file includes all codes in one Jupiter Notbook. The Notbook includes the same informaition demostratied in py files.

## How run py files 
### Install required packages
The provided requirements_UNIX.txt file is tested on MacBook (Retina, 12-inch, 2017) version 11.2.3 and it can be used to install all the required packages for UNIX operating system. Use the following command

> cd RaceFromFace \
> pip install –r requirements_UNIX.txt

The provided requirements_Linux.txt file is tested on Ubuntu 20.04 and it can be used to install all the required packages for Linux operating systems. Use the following command.

> cd RaceFromFace \
> pip install –r requirements_Linux.txt

## Codes
* For training model use
__Coding/Demo_Execution.py__

* For evaluating the existing models use 
__Coding/Demo_Evaluation.py__

* For analysing results use 
__Coding/Demo_Results.py__

* For training all Keras Applications use 
__BestKerasPreTrainedModel.py__   

### Run the codes
> cd RaceFromFace \
> python file_name
 
example:
> cd RaceFromFace \
> python Coding/Demo_Execution.py


## Notes
* Inputes in terminal should not include spaces or other characters
