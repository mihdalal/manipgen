import time
from collections import deque
from multiprocessing import Process, Queue, Value

import numpy as np
import pyrealsense2 as rs
from omegaconf import DictConfig

class Rate:
    """
    Maintains constant control rate for the environment loop
    """

    def __init__(self, frequency):
        self._period = 1.0 / frequency
        self._last = time.time()

    def sleep(self):
        current_delta = time.time() - self._last
        sleep_time = max(0, self._period - current_delta)
        if sleep_time:
            time.sleep(sleep_time)
        self._last = time.time()

def add_image(
    camera_cfg: DictConfig,
    rgb_frame_queue: Queue,
    depth_frame_queue: Queue,
    hz: float,
    closed: Value,
    hole_fill: bool = True,
):
    # Configure depth and color streams
    pipe = rs.pipeline()
    config = rs.config()

    config.enable_device(camera_cfg.device_id)
    config.enable_stream(
        rs.stream.depth,
        camera_cfg.img_width,
        camera_cfg.img_height,
        rs.format.z16,
        camera_cfg.hz,
    )
    config.enable_stream(
        rs.stream.color,
        camera_cfg.img_width,
        camera_cfg.img_height,
        rs.format.rgb8,
        camera_cfg.hz,
    )

    pipe.start(config)

    if camera_cfg.warm_start.enabled:
        print(
            "[INFO] Warm start cameras (realsense auto-adjusts brightness"
            " during initial frames)"
        )
        for _ in range(camera_cfg.warm_start.frames):
            pipe.poll_for_frames()
            time.sleep(1.0 / camera_cfg.hz)

    align = rs.align(rs.stream.color)
    depth_processing = (
        rs.temporal_filter(),
    )
    if hole_fill:
        depth_processing += (rs.hole_filling_filter(),)
    
    step = 0
    rate = Rate(hz)
    try:
        while not closed.value:
            frames = pipe.poll_for_frames()
            if frames.is_frameset():
                frames = align.process(frames)
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue
                
                for img_filter in depth_processing:
                    depth_frame = img_filter.process(depth_frame)
                
                depth_image = np.asanyarray(depth_frame.get_data())
                depth_timestamp = depth_frame.get_timestamp()
                color_image = np.asanyarray(color_frame.get_data())
                color_timestamp = color_frame.get_timestamp()
                try:
                    rgb_frame_queue.put(
                        (color_image, color_timestamp), block=False
                    )
                    depth_frame_queue.put(
                        (depth_image, depth_timestamp), block=False
                    )
                except:
                    # queue is full, drop the oldest frame
                    try:
                        rgb_frame_queue.get(block=False)
                        depth_frame_queue.get(block=False)
                    except:
                        pass

            rate.sleep()
            step += 1

    except KeyboardInterrupt:
        print("[INFO] Camera stream closed")


class RealSenseCam:
    """
    A Sensor interface class for realsense camera
    that provides a gym style observation wrapper to camera images.
    """

    def __init__(self, camera_cfg: DictConfig):
        device_ls = []
        for cam in rs.context().query_devices():
            device_ls.append(cam.get_info(rs.camera_info(1)))
        device_ls.sort()
        self.hz = camera_cfg.hz
        self.device_id = camera_cfg.device_id
        self.name = camera_cfg.name
        assert self.device_id in device_ls

        # window_dur is the duration of the sliding window in seconds
        self.window_dur = camera_cfg.window_dur
        if self.window_dur is None:
            self.buffer_size = (
                1  # no sliding window, use single frame at self.hz
            )
            self.window = None
        else:
            self.buffer_size = int(self.window_dur * self.hz)
            self.window = deque(maxlen=self.buffer_size)

        self.rgb_frame_queue = Queue(self.buffer_size)
        self.depth_frame_queue = Queue(self.buffer_size)
        self.observer_proc = None
        self.camera_cfg = camera_cfg
        self.closed = Value("b", False)

        self.start()
        print(f"[INFO] Camera setup completed.")

    def start(self):
        if self.observer_proc is None:
            self.observer_proc = Process(
                target=add_image,
                args=(
                    self.camera_cfg,
                    self.rgb_frame_queue,
                    self.depth_frame_queue,
                    self.hz,
                    self.closed,
                    self.camera_cfg.hole_fill,
                ),
                daemon=True,
            )
            self.observer_proc.start()

    def close(self):
        """
        Close the image queue and the camera stream process
        """
        if self.observer_proc is not None:
            self.closed.value = True
            self.rgb_frame_queue.close()
            self.rgb_frame_queue.join_thread()
            self.rgb_frame_queue = None
            self.depth_frame_queue.close()
            self.depth_frame_queue.join_thread()
            self.depth_frame_queue = None
            self.observer_proc.join()
            self.observer_proc = None

    def reset(self):
        """
        reset sensor
        """
        obs = self.get_obs()
        info = {}
        return obs, info

    def get_obs(self):
        obs = {self.name: None}
        try:
            if self.window is None:
                obs[self.name] = self.rgb_frame_queue.get()
                obs[f"{self.name}_depth"] = self.depth_frame_queue.get()

            else:
                while not self.rgb_frame_queue.empty():
                    self.window.append(self.rgb_frame_queue.get())

                obs[self.name] = list(self.window)

        except:
            print(f"{self.name} queue is empty")

        return obs