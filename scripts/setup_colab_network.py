import subprocess
import time

from google.colab import userdata


def main():

    if not userdata.get("TAILSCALE_AUTH_KEY"):
        raise RuntimeError("TAILSCALE_AUTH_KEY is not available.")

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
        ["sudo", "tailscale", "up", "--auth-key", userdata.get("TAILSCALE_AUTH_KEY")],
        check=True,
    )
    print("Done")


if __name__ == "__main__":
    main()
