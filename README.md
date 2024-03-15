# Event_Star_tracker
 Star traking Attitude Determination via event camara and NN

# To run 
$ python3 main.py -i ../data/recordings/02_02/recording_2024-02-02_17-21-24_FOV14.3_MAG6.raw --show-video 
$ python3 main.py --load-params ./parameters.yaml

# Server 
$ ssh eventjetson@10.157.95.86
$ ssh eventjetson@10.183.239.148

# Screen 

screen -S NAME_SCREEN : Create screen 
screen -ls : Avalible Screen list 
screen -r ID_SCREEN/NAME : Access screen 
screen -d ID_SCREEN/NAME : Close screen whereever it is 

key shorcuts: 
    crtl + a + d : exit 
    crtl + a + 