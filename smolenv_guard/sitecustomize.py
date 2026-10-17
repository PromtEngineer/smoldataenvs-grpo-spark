"""Loaded by every Python process in the SmolDataEnvs local sandbox (via PYTHONPATH).

Best-effort, in-process network guard: an audit hook that refuses DNS lookups and
non-Unix-socket connects. The OS backend (sandbox-exec / bwrap / unshare) is the real
boundary; this is the belt to its braces, and the only guard under backend "none".
"""
import os
import socket
import sys

if os.environ.get("SMOLENV_SANDBOX") == "1":
    _BLOCK = {"socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr",
              "socket.getnameinfo", "socket.sendto", "socket.sendmsg"}

    def _hook(event, args):
        if event in _BLOCK:
            raise PermissionError(f"network disabled in the SmolDataEnvs sandbox ({event})")
        if event == "socket.connect":
            sock = args[0]
            if getattr(sock, "family", None) != getattr(socket, "AF_UNIX", object()):
                raise PermissionError("network disabled in the SmolDataEnvs sandbox (socket.connect)")

    sys.addaudithook(_hook)
