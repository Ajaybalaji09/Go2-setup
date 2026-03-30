#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import base64
import json
import subprocess
import time

from go2_interfaces.srv import Say
from unitree_api.msg import Request


class Go2TTS(Node):

    def __init__(self):
        super().__init__('go2_tts')

        # Publisher
        self.pub = self.create_publisher(Request, 'api/audiohub/request', 10)

        # Service
        self.srv = self.create_service(Say, 'say', self.handle_say)

        self.get_logger().info("TTS Python node started")

    # -------------------------------
    # Service Callback
    # -------------------------------
    def handle_say(self, request, response):

        text = request.text

        # Step 1: Generate WAV using text2wave
        try:
            command = f'echo "{text}" | text2wave -o /tmp/output.wav'
            subprocess.run(command, shell=True, check=True)
        except Exception as e:
            self.get_logger().error(f"TTS generation failed: {e}")
            response.success = False
            return response

        # Step 2: Read file
        try:
            with open("/tmp/output.wav", "rb") as f:
                audio_data = f.read()
        except:
            self.get_logger().error("Failed to read WAV file")
            response.success = False
            return response

        # Step 3: Chunking
        chunk_size = 256 * 1024
        chunks = [
            audio_data[i:i + chunk_size]
            for i in range(0, len(audio_data), chunk_size)
        ]

        total_chunks = len(chunks)

        # Step 4: Send StartAudio
        start_req = Request()
        start_req.header.identity.api_id = 4001  # replace with Audio::StartAudio

        self.pub.publish(start_req)
        time.sleep(0.1)

        # Step 5: Send chunks
        for idx, chunk in enumerate(chunks):

            encoded = base64.b64encode(chunk).decode('utf-8')

            payload = {
                "current_block_index": idx + 1,
                "total_block_number": total_chunks,
                "block_content": encoded
            }

            req = Request()
            req.header.identity.api_id = 4003  # replace with Audio::TTS
            req.parameter = json.dumps(payload)

            self.pub.publish(req)

        response.success = True
        return response


# -------------------------------
# Main
# -------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = Go2TTS()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()