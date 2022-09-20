# ModernGL Moon Lander
![Moon Lander screenshot](/moon_lander.png)

## About
Moon Lander is my second moderngl.py, this time in pseudo 2D with procedurally generated moon surface.

The idea is to land on the moon, then return to the command module without crashing.
Currently there is no way to redock with the command module, but it's nigh impossible anyhow.

The game can only be operated with a pygame compatible joystick.

The physics used is fairly simple and no collision geometry is used.

To land, the first things to do is undock from the command module (button 2), then press the trigger and try to reduce your speed.

## Installation

Locate everything anywhere you like. Assuming you have python3 installed, then for the first time, run from a terminal and you'll see what dependancy packages are missing.

### Dependancies

- python3
- moderngl
- moderngl_window
- pathlib           #to locate the resources directory
- numpy             #to create vertix and indici objects
- pyrr              #for matrix translations
- pygame            #for onscreen gui
- json              #only used for diagnostic output

All standard libraries as of May 2022

## Running
Since pygame joystick doesn't work well within the moderngl_window loop, it is necessary to run the joystick program in a seperate terminal:
""python3 joystick_client3.py"" (Linux) or ""py -3 joystick_client3.py"" (Windows).
Run the main program in the second terminal:
""python3 moon_lander.py"" (Linux) or ""py -3 moon_lander.py"" (Windows)

