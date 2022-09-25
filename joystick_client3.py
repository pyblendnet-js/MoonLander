import pygame
import udp_client

pygame.init()


# Used to manage how fast the screen updates.
clock = pygame.time.Clock()

# Initialize the joysticks.
pygame.joystick.init()

client = udp_client.udpClientClass()

done = False
# Get count of joysticks.
joystick_count = pygame.joystick.get_count()
while joystick_count == 0:
  pass
print("Number of joysticks: {}".format(joystick_count))
joystick = pygame.joystick.Joystick(0)
joystick.init()
axes = joystick.get_numaxes()
buttons = joystick.get_numbuttons()

def sendbtn(state):
  st = str(state)
  for i in range(buttons):
    button = joystick.get_button(i)
    st += str(button) + ","
  #endfor  
  client.sendStr(st)
#enddef  

# -------- Main Program Loop -----------
while not done:
    #
    # EVENT PROCESSING STEP
    #
    # Possible joystick actions: JOYAXISMOTION, JOYBALLMOTION, JOYBUTTONDOWN,
    # JOYBUTTONUP, JOYHATMOTION
    for event in pygame.event.get(): # User did something.
        if event.type == pygame.QUIT: # If user clicked close.
            done = True # Flag that we are done so we exit this loop.
        elif event.type == pygame.JOYBUTTONDOWN:
            sendbtn("dn")
        elif event.type == pygame.JOYBUTTONUP:
            sendbtn("up")
        elif event.type == pygame.JOYAXISMOTION:
            st = ""
            for i in range(axes):
              axis = joystick.get_axis(i)
              st +="{:>6.3f},".format(axis)
            #endfor
            client.sendStr(st)
        #endfor
        #hats = joystick.get_numhats()
        #print( "Number of hats: {}".format(hats))
 
        # Hat position. All or nothing for direction, not a float like
        # get_axis(). Position is a tuple of int values (x, y).
        '''for i in range(hats):
            hat = joystick.get_hat(i)
            textPrint.tprint(screen, "Hat {} value: {}".format(i, str(hat)))
        '''


    # Limit to 20 frames per second.
    clock.tick(20)

# Close the window and quit.
# If you forget this line, the program will 'hang'
# on exit if running from IDLE.
pygame.quit()
