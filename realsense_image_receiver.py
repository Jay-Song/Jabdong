import argparse
import cv2
import time
import zmq
import msgpack
import numpy as np


def main():
    parser = argparse.ArgumentParser("RealSense Image Sender")
    parser.add_argument(
        "--pub-ip", type=str, default="127.0.0.1", help="IP address for publishing"
    )
    parser.add_argument(
        "--pub-port", type=int, default=6001, help="port number for publishing"
    )
    args = parser.parse_args()

    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.CONFLATE, 1)  # to only keep the latest message
    socket.setsockopt_string(zmq.SUBSCRIBE, "")
    socket.connect(f"tcp://{args.pub_ip}:{args.pub_port}")

    # Request camera parameter
    req_socket = context.socket(zmq.REQ)
    req_socket.connect(f"tcp://{args.pub_ip}:{args.pub_port + 1}")
    req_socket.send_string("color")
    color_param_msg = req_socket.recv()
    color_param = msgpack.unpackb(color_param_msg, raw=False)
    print(color_param)

    req_socket.send_string("depth")
    depth_param_msg = req_socket.recv()
    depth_param = msgpack.unpackb(depth_param_msg, raw=False)
    print(depth_param)
    depth_width = depth_param["width"]
    depth_height = depth_param["height"]

    while True:
        try:
            # Receive both parts
            msg = socket.recv()  # flags=zmq.NOBLOCK
            data = msgpack.unpackb(msg, raw=False)

            # Decode JPEG color image
            color_image = cv2.imdecode(
                np.frombuffer(data["color"], dtype=np.uint8), cv2.IMREAD_COLOR
            )

            # Decode raw depth
            depth_image = np.frombuffer(data["depth"], dtype=np.uint16).reshape(
                (depth_height, depth_width)
            )

            # Show for debugging
            cv2.imshow("Color", color_image)
            cv2.imshow(
                "Depth", (depth_image / 1000.0).astype(np.float32)
            )  # Normalize for viewing

            """
            Do something for image processing
            
            
            
            """

            key = cv2.waitKey(1)
            if key == ord("q") or key == 27:  # ESC
                break

        except Exception as e:
            print("Error:", e)
            # no message or any other error

    cv2.destroyAllWindows()
    socket.close()
    req_socket.close()
    context.term()


if __name__ == "__main__":
    main()
