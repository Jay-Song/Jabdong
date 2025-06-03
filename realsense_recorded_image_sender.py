import argparse
import cv2
import msgpack
import pickle
from pathlib import Path
import numpy as np
import threading
import time
import zmq
from teleop.realsense_manager import RealSenseManager


FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FRAME_FPS = 30


def handle_camera_info_requests(
    rep_socket: zmq.SyncSocket, color_param: dict, depth_param: dict
):
    while True:
        message = rep_socket.recv_string()
        if message == "color":
            rep_socket.send(msgpack.packb(color_param, use_bin_type=True))
        elif message == "depth":
            rep_socket.send(msgpack.packb(depth_param, use_bin_type=True))


def load_pkl(pkl_file_path: Path):
    with open(pkl_file_path, "rb") as f:
        data = pickle.load(f)

    return data


def main():
    parser = argparse.ArgumentParser("RealSense Image Sender")
    parser.add_argument("--data-dir", type=str, required=True, help="recorded data")
    parser.add_argument(
        "--pub-ip", type=str, default="127.0.0.1", help="IP address for publishing"
    )
    parser.add_argument(
        "--pub-port", type=int, default=6001, help="port number for publishing"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    data_file = data_dir / "data.pkl"
    data = load_pkl(data_file)
    print(data)

    # get camera parameters
    width = data["cam"]["side"]["width"]
    height = data["cam"]["side"]["height"]
    color_fx = data["cam"]["side"]["color"]["fx"]
    color_fy = data["cam"]["side"]["color"]["fy"]
    color_cx = data["cam"]["side"]["color"]["cx"]
    color_cy = data["cam"]["side"]["color"]["cy"]
    depth_fx = data["cam"]["side"]["depth"]["fx"]
    depth_fy = data["cam"]["side"]["depth"]["fy"]
    depth_cx = data["cam"]["side"]["depth"]["cx"]
    depth_cy = data["cam"]["side"]["depth"]["cy"]
    depth_scale = data["cam"]["side"]["depth"]["depth_scale"]

    color_param = {
        "width": width,
        "height": height,
        "fx": color_fx,
        "fy": color_fy,
        "cx": color_cx,
        "cy": color_cy,
    }

    depth_param = {
        "width": width,
        "height": height,
        "fx": depth_fx,
        "fy": depth_fy,
        "cx": depth_cx,
        "cy": depth_cy,
        "depth_scale": depth_scale,
    }

    # Setup ZeroMQ socket
    context = zmq.Context()
    pub_socket = context.socket(zmq.PUB)
    pub_socket.bind(f"tcp://{args.pub_ip}:{args.pub_port}")

    rep_socket = context.socket(zmq.REP)
    rep_socket.bind(f"tcp://{args.pub_ip}:{int(args.pub_port) + 1}")

    # Start thread to handle camera info requests
    threading.Thread(
        target=handle_camera_info_requests,
        args=(rep_socket, color_param, depth_param),
        daemon=True,
    ).start()

    try:
        data_idx = 0
        last_time = time.time()
        while True:
            color_img_path = (
                data_dir / "side" / "color" / f"side_color_{data_idx:04d}.jpg"
            )
            depth_img_path = (
                data_dir / "side" / "depth" / f"side_depth_{data_idx:04d}.png"
            )

            color_image = cv2.imread(color_img_path, flags=cv2.IMREAD_COLOR)
            if color_image is None:
                data_idx = 0
                continue
            else:
                data_idx += 1

            depth_image = cv2.imread(depth_img_path, flags=cv2.IMREAD_ANYDEPTH)

            # JPEG compress the color image
            _, color_jpeg = cv2.imencode(
                ".jpg", color_image, [cv2.IMWRITE_JPEG_QUALITY, 100]
            )

            # Send: [JPEG color][raw depth]
            msg = msgpack.packb(
                {
                    "topic": "realsense",
                    "color": color_jpeg.tobytes(),
                    "depth": depth_image.tobytes(),
                },
                use_bin_type=True,
            )
            pub_socket.send(msg)
            print("sent")
            now = time.time()
            passed = now - last_time
            remaining = 0.033 - passed
            if remaining > 0.0001:
                time.sleep(remaining)
            last_time = time.time()

    except KeyboardInterrupt:
        pass
    finally:
        pub_socket.close()
        rep_socket.close()
        context.term()


if __name__ == "__main__":
    main()
