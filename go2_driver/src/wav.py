#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import base64
import json
import time
import wave
import subprocess

from go2_interfaces.srv import Say
from unitree_api.msg import Request


class Go2AudioPlayer(Node):

    def __init__(self):
        super().__init__('go2_audio_player')

        self.pub = self.create_publisher(Request, 'api/audiohub/request', 10)
        self.srv = self.create_service(Say, 'play_audio', self.handle_play)

        self.get_logger().info("Audio player node started")

    def handle_play(self, request, response):

        wav_path = request.text
        converted_path = '/tmp/go2_converted.wav'

        # Step 1: Convert to 16-bit mono 24000 Hz PCM using ffmpeg
        try:
            subprocess.run([
                'ffmpeg', '-y',
                '-i', wav_path,
                '-ar', '24000',
                '-ac', '1',
                '-sample_fmt', 's16',
                converted_path
            ], check=True, capture_output=True)
            self.get_logger().info("Converted to 16-bit mono 24000 Hz")
        except subprocess.CalledProcessError as e:
            self.get_logger().error(f"ffmpeg conversion failed: {e.stderr.decode()}")
            response.success = False
            return response

        # Step 2: Read raw WAV bytes (including header — matches working TTS pattern)
        try:
            with open(converted_path, 'rb') as f:
                audio_data = f.read()
            self.get_logger().info(f"Read {len(audio_data)} bytes")
        except Exception as e:
            self.get_logger().error(f"Failed to read WAV: {e}")
            response.success = False
            return response

        # Step 3: Chunk
        chunk_size = 256 * 1024
        chunks = [
            audio_data[i:i + chunk_size]
            for i in range(0, len(audio_data), chunk_size)
        ]
        total_chunks = len(chunks)
        self.get_logger().info(f"Sending {len(audio_data)} bytes in {total_chunks} chunk(s)")

        # Step 4: StartAudio
        start_req = Request()
        start_req.header.identity.api_id = 4001
        self.pub.publish(start_req)
        time.sleep(0.1)

        # Step 5: Send chunks (no delay — matches working TTS pattern)
        for idx, chunk in enumerate(chunks):
            encoded = base64.b64encode(chunk).decode('utf-8')

            payload = {
                "current_block_index": idx + 1,
                "total_block_number": total_chunks,
                "block_content": encoded
            }

            req = Request()
            req.header.identity.api_id = 4003
            req.parameter = json.dumps(payload)

            self.pub.publish(req)
            self.get_logger().info(f"Sent chunk {idx + 1}/{total_chunks}")

        # No StopAudio, no playback wait — matches working TTS pattern

        self.get_logger().info("Done sending audio")
        response.success = True
        return response


def main(args=None):
    rclpy.init(args=args)
    node = Go2AudioPlayer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
