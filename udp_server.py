#from https://pythontic.com/modules/socket/udp-client-server-example

import socket

class udp_server_class:
  def __init__(self, ip = "127.0.0.1", port = 3333, buffer_size = 1024, timeout = 0):
    self.ip   = ip
    self.port = port
    self.bufferSize  = buffer_size
    self.timeout = timeout
    # Create a datagram socket
    self.socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    # Bind to address and ip
    self.socket.bind((ip,port))
    if timeout > 0:
      self.socket.settimeout(timeout)
    else:
      self.socket.setblocking(0)
    print("UDP server up and listening")

  def get(self):
    try:
      bytesAddressPair = self.socket.recvfrom(self.bufferSize)
      message = bytesAddressPair[0]
      address = bytesAddressPair[1]
      return message,address
    except socket.timeout:
      return "timeout",None
    except:
      return None,None 
    #endtry
  #enddef
#endclass

if __name__ == "__main__":
  server = udp_server_class()
  # Listen for incoming datagrams from TUIO sender
  while(True):
    message,address = server.get()
    if address:
      clientMsg = "Message from Client:{}".format(message)
      clientIP  = "Client IP Address:{}".format(address)  
      print(clientMsg)
      print(clientIP)
    #else:
    #  print(".")
    #endif
  #endwhile
#endif
   
   

