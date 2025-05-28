import cv2
import threading
import queue
import time
import logging
import psutil

# setup logger
logging.basicConfig(level=logging.DEBUG, format="[DEBUG] %(message)s")
logger = logging.getLogger(__name__)

class ThreadingClass:
    def __init__(self, name):
        logger.debug(f"Initializing ThreadingClass with source: {name}")
        self.cap = cv2.VideoCapture(name)
        if not self.cap.isOpened():
            logger.error(f"Failed to open video source: {name}")
            raise ValueError(f"Failed to open video source: {name}")
        logger.debug(f"Video FPS: {self.cap.get(cv2.CAP_PROP_FPS)}")
        logger.debug(f"Video frame count: {self.cap.get(cv2.CAP_PROP_FRAME_COUNT)}")
        self.q = queue.Queue(maxsize=10)
        self.running = True
        self.frame_count = 0
        self.condition = threading.Condition()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()
        time.sleep(0.5)

    def _reader(self):
        # function to read frames from the video source
        retry_count = 0
        max_retries = 5
        while self.running and retry_count < max_retries:
            with self.condition:
                while self.q.full() and self.running:
                    logger.debug("Queue full, read-thread waiting")
                    self.condition.wait()
                if not self.running:
                    break
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    retry_count += 1
                    logger.warning(f"Failed to read frame from video source (attempt {retry_count}/{max_retries})")
                    time.sleep(0.5)
                    continue
                self.q.put(frame)
                self.frame_count += 1
                logger.debug(f"Frame read successfully, frame {self.frame_count}, queue size: {self.q.qsize()}")
                retry_count = 0
                time.sleep(0.005)
        if retry_count >= max_retries:
            logger.error("Max retries reached, stopping reader thread")
            self.running = False

    def read(self):
        # function to read a frame from the queue
        with self.condition:
            frame = None
            try:
                frame = self.q.get_nowait()
                self.condition.notify()  # Signal read-thread to read one frame
            except queue.Empty:
                pass
            return frame

    def release(self):
        # function to release the video capture resource
        self.running = False
        with self.condition:
            self.condition.notify()
        self.cap.release()
        logger.debug("VideoCapture released")