import os
import subprocess
import time


def main():
    auth_key = os.environ.get("TAILSCALE_AUTH_KEY")

    if not auth_key:
        raise RuntimeError("TAILSCALE_AUTH_KEY environment variable is not set.")

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
        ["sudo", "tailscale", "up", "--auth-key", auth_key],
        check=True,
    )

    os.environ["HTTP_PROXY"] = "socks5h://localhost:1055"
    os.environ["HTTPS_PROXY"] = "socks5h://localhost:1055"


if __name__ == "__main__":
    main()
