import os
import subprocess
import time


def main():
    auth_key = os.environ.get("TAILSCALE_AUTH_KEY")

    if not auth_key:
        raise RuntimeError("TAILSCALE_AUTH_KEY environment variable is not set.")

    # Start tailscaled in the background.
    subprocess.Popen(
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

    # Give the daemon a moment to initialize.
    time.sleep(2)

    # Connect this Colab runtime to the tailnet.
    subprocess.run(
        ["sudo", "tailscale", "up", "--auth-key", auth_key],
        check=True,
    )


if __name__ == "__main__":
    main()
