import os
import subprocess
import time


def main():
    auth_key = os.environ.get("TAILSCALE_AUTH_KEY")

    if not auth_key:
        raise RuntimeError("TAILSCALE_AUTH_KEY environment variable is not set.")

    print("Launching tailscaled")
    # Start tailscaled as a detached background process.
    subprocess.run(
        [
            "sudo",
            "bash",
            "-c",
            """
            nohup tailscaled \
                --tun=userspace-networking \
                --socks5-server=localhost:1055 \
                --state=/tmp/tailscale.state \
                > /tmp/tailscaled.log 2>&1 &
            """,
        ],
        check=True,
    )

    print("Done")
    # Give the daemon a moment to initialize.
    time.sleep(2)

    print("Launching tailscale")
    # Connect this Colab runtime to the tailnet.
    subprocess.run(
        ["sudo", "tailscale", "up", "--auth-key", auth_key],
        check=True,
    )
    print("Done")


if __name__ == "__main__":
    main()
