# Description
This is the example how to use ArduCam IMX477 AF Module on Jetson Nano 

# Usage

## Our wrapper
We wrap the arducam provided libraries in a class that has the same interface as the rest of our cameras. The script to test this camera and the motorized focus is `src/perception/dev/focus_test.py`. It lets you manually adjust the focus with a slider.


## Library-provided test files:

`cd libraries`
* Autofocus.py Example of autofocus  
    python3 Autofocus.py -i 9  

* FocuserExample.py Example of manual focus  
    python3 FocuserExample.py -i 9  
