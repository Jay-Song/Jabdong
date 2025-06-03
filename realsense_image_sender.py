import argparse
import cv2
import msgpack
import numpy as np
import time
import zmq
from teleop.realsense_manager import RealSenseManager


SERIALS = {
    "top": "218622275992",
    "top2": "213622251190",
    "bottom": "218622274848",
    "left": "218622271500",
    "right": "218622273829",
}

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FRAME_FPS = 30


def main():
    parser = argparse.ArgumentParser("RealSense Image Sender")
    parser.add_argument(
        "--serial", type=str, default=SERIALS["top2"], help="camera serial number"
    )
    parser.add_argument(
        "--pub-ip", type=str, default="127.0.0.1", help="IP address for publishing"
    )
    parser.add_argument(
        "--pub-port", type=str, default="6001", help="port number for publishing"
    )
    args = parser.parse_args()

    # camera setup
    cam_manager = RealSenseManager(args.serial, FRAME_WIDTH, FRAME_HEIGHT, FRAME_FPS)
    # give some time for camera warm up
    time.sleep(2.0)

    # get camera parameters
    cam_color_intr = cam_manager.get_color_intrinsic()
    cam_c2d_r, cam_c2d_t = cam_manager.get_color_extrinsic()
    cam_depth_intr = cam_manager.get_depth_intrinsic()
    cam_d2c_r, cam_d2c_t = cam_manager.get_depth_extrinsic()
    cam_depth_scale = cam_manager.get_depth_scale()

    # Setup ZeroMQ socket
    context = zmq.Context()
    pub_socket = context.socket(zmq.PUB)
    pub_socket.bind(f"tcp://{args.pub_ip}:{args.pub_port}")

    try:
        while True:
            start = time.time()
            cam_manager.poll_frames()
            color_image = cam_manager.get_color_image()
            depth_image = cam_manager.get_depth_image()

            # JPEG compress the color image
            _, color_jpeg = cv2.imencode(
                ".jpg", color_image, [cv2.IMWRITE_JPEG_QUALITY, 90]
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
            print("sent ", (time.time() - start))

    except KeyboardInterrupt:
        pass
    finally:
        cam_manager.stop()


if __name__ == "__main__":
    main()
