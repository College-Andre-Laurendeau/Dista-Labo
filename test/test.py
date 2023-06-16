import os

for root, dirs, files in os.walk("/home/dista/Documents/dista/calibration/SN2042565/stereo"):
  #if ".jpg" in files:
    #print(files)
    #time = os.path.getmtime(f"/home/dista/Documents/dista/calibration/SN2042565/stereo/{files}")
    #print(time) 
    files.sort()
    print(files[0])
   
