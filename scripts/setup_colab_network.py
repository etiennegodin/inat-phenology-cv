import os
import subprocess
import time

from google.colab import userdata

ts_auth_key = userdata.get("TAILSCALE_AUTH_KEY")

subprocess.run(
    [
        "sudo",
        "tailscaled",
        "--tun=userspace-networking",
        "--socks5-server=localhost:1055",
        "--state=/tmp/tailscale.state",
    ],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)

time.sleep(2)

subprocess.run(
    ["sudo", "tailscale", "up", "--auth-key", ts_auth_key],
    check=True,
)

del ts_auth_key

os.environ["HTTP_PROXY"] = "socks5h://localhost:1055"
os.environ["HTTPS_PROXY"] = "socks5h://localhost:1055"
