import os

for root, dirs, files in os.walk("/l"):
    print(files)
