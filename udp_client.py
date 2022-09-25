import socket

class udpClientClass: 
    def __init__(self,host = "127.0.0.1",port = 3333,buffer_size = 1024,blocking=True):
        self.serverAddressPort = (host,port)
        self.bufferSize = buffer_size
        self.open_socket()
        if not blocking:
          self.socket.setblocking(0)
    #enddef

    def open_socket(self):
        self.socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    #enddef

    def sendStr(self,msg):
        self.socket.sendto(str.encode(msg),self.serverAddressPort)
    #enddef

    def send(self,data):
        self.socket.sendto(data,self.serverAddressPort)
    #enddef

    def getReply(self):
        return self.socket.recvfrom(bufferSize)
    #enddef
#endclass

if __name__ == "__main__":
    udpClient = udpClientClass()
    while(True):
        msg = input("Enter msg:")
        udpClient.sendStr(msg)
        #msg = udpClient.getReply()
        #print("Reply:",msg.decode(("utf-8"))
        
    #endwhile 

