"""Authenticated Unix-socket liveness without network access."""
import http.client
import socket
from pathlib import Path

class UnixHTTP(http.client.HTTPConnection):
    def connect(self):
        self.sock=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
        self.sock.settimeout(2)
        self.sock.connect('/ipc/retro.sock')

if __name__=='__main__':
    client=UnixHTTP('localhost')
    client.request('GET','/health/live',headers={'Authorization':'Bearer '+Path('/run/secrets/retro_operator').read_text().strip()})
    response=client.getresponse()
    assert response.status==200
    client.close()
