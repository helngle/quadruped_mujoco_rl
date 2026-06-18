import argparse
import socket

from tensorboard import program


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start TensorBoard for this project.")
    parser.add_argument("--logdir", default="runs", help="TensorBoard log directory.")
    parser.add_argument("--host", default="127.0.0.1", help="Host address to bind.")
    parser.add_argument("--port", type=int, default=6006, help="Port to bind.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    original_socket = socket.socket

    class TimeoutSocket(socket.socket):
        def connect_ex(self, address):  # type: ignore[no-untyped-def]
            self.settimeout(1.0)
            return super().connect_ex(address)

    socket.socket = TimeoutSocket
    try:
        tensorboard = program.TensorBoard()
        tensorboard.configure(
            argv=[
                None,
                "--logdir",
                args.logdir,
                "--host",
                args.host,
                "--port",
                str(args.port),
                "--load_fast=false",
            ]
        )
        url = tensorboard.launch()
        print(f"TensorBoard is running at {url}")
        input("Press Enter to stop TensorBoard...\n")
    finally:
        socket.socket = original_socket


if __name__ == "__main__":
    main()
