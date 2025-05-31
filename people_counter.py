from tracker.centroidtracker import CentroidTracker
from tracker.trackableobject import TrackableObject
from imutils.video import VideoStream
from itertools import zip_longest
from utils.mailer import Mailer
from imutils.video import FPS
from utils import thread
import numpy as np
import threading
import argparse
import datetime
import schedule
import logging
import imutils
import time
import dlib
import json
import csv
import cv2
import psutil
import signal
import sys
from picamera2 import Picamera2
from queue import Queue, Empty
from postTelemetry_mqtt_tb import MQTTThingsBoardClient

# execution start time
start_time = time.time()
# setup logger
logging.basicConfig(level=logging.DEBUG, format="[DEBUG] %(message)s")
logger = logging.getLogger(__name__)
# initiate features config.
with open("utils/config.json", "r") as file:
    config = json.load(file)

# Custom class for PiCamera threading
class PiCameraThreadingClass:
    def __init__(self):
        logger.debug("Initializing PiCameraThreadingClass")
        self.camera = Picamera2()
        config_cam = self.camera.create_video_configuration(main={"size": (320, 240), "format": "RGB888"})
        self.camera.configure(config_cam)
        self.camera.start()
        self.q = Queue(maxsize=10)
        self.running = True
        self.frame_count = 0
        self.condition = threading.Condition()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()
        time.sleep(0.5)

    def _reader(self):
        # Function to read frames from the PiCamera
        while self.running:
            with self.condition:
                while self.q.full() and self.running:
                    self.condition.wait()
                if not self.running:
                    break
                frame = self.camera.capture_array()
                if frame is not None:
                    self.q.put(frame)
                    self.frame_count += 1
                time.sleep(0.005)  # Small delay to prevent overwhelming the queue

    def read(self):
        # Function to read a frame from the queue
        with self.condition:
            frame = None
            try:
                frame = self.q.get_nowait()
                self.condition.notify()
            except Empty:
                pass
            return frame

    def release(self):
        # Function to release the camera resource
        self.running = False
        with self.condition:
            self.condition.notify()
        self.camera.stop()
        logger.debug("PiCamera released")

# Global variables for telemetry
cpu_usages = []
memory_usages = []
temperatures = []
fps_values = []
tb_client = None
running = True
lock = threading.Lock()

# Signal handler for Ctrl+C
def signal_handler(sig, frame):
    global running, tb_client, vs, writer
    print("Ctrl+C detected, cleaning up...")
    running = False
    if vs is not None:
        vs.release()
    if tb_client:
        tb_client.disconnect()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    # Print average resource usage and FPS
    if cpu_usages:
        print(f"Average CPU Usage: {sum(cpu_usages)/len(cpu_usages):.2f}%")
    if memory_usages:
        print(f"Average Memory Usage: {sum(memory_usages)/len(memory_usages):.2f}%")
    if temperatures:
        print(f"Average Temperature: {sum(temperatures)/len(temperatures):.2f}°C")
    if fps_values:
        print(f"Average FPS: {sum(fps_values)/len(fps_values):.2f}")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def parse_arguments():
    # function to parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prototxt", required=False,
        help="path to Caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model", required=True,
        help="path to Caffe pre-trained model")
    ap.add_argument("-i", "--input", type=str,
        help="path to optional input video file")
    ap.add_argument("-o", "--output", type=str,
        help="path to optional output video file")
    # confidence default 0.4
    ap.add_argument("-c", "--confidence", type=float, default=0.4,
        help="minimum probability to filter weak detections")
    ap.add_argument("-s", "--skip-frames", type=int, default=20,
        help="# of skip frames between detections")
    ap.add_argument("--server-IP", type=str, default="",
        help="ThingsBoard server domain")
    ap.add_argument("-P", "--Port", type=int, default=0,
        help="MQTT port for ThingsBoard server")
    ap.add_argument("-a", "--token", type=str, default="",
        help="Device access token for ThingsBoard authentication")
    args = vars(ap.parse_args())
    return args

def send_mail():
    # function to send the email alerts
    Mailer().send(config["Email_Receive"])

def log_data(move_in, in_time, move_out, out_time):
    # function to log the counting data
    data = [move_in, in_time, move_out, out_time]
    # transpose the data to align the columns properly
    export_data = zip_longest(*data, fillvalue='')

    with open('utils/data/logs/counting_data.csv', 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        if myfile.tell() == 0:  # check if header rows are already existing
            wr.writerow(("Move In", "In Time", "Move Out", "Out Time"))
            wr.writerows(export_data)

def monitor_resources(tb_client, server_IP, port, token):
    global cpu_usages, memory_usages, temperatures
    last_time = time.time()
    while running:
        current_time = time.time()
        if current_time - last_time >= 10:
            cpu_usage = psutil.cpu_percent(interval=None)
            memory_usage = psutil.virtual_memory().percent
            try:
                with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                    temp = int(f.read()) / 1000.0
            except:
                temp = 0.0
            with lock:
                cpu_usages.append(cpu_usage)
                memory_usages.append(memory_usage)
                temperatures.append(temp)
            tb_client.send_telemetry(server_IP, port, token, "CPU_usage", round(cpu_usage, 2))
            tb_client.send_telemetry(server_IP, port, token, "memory_usage", round(memory_usage, 2))
            tb_client.send_telemetry(server_IP, port, token, "Temperature", round(temp, 2))
            last_time = current_time
        time.sleep(1)

def people_counter():
    # main function for people_counter.py
    global tb_client, running, vs, writer
    args = parse_arguments()

    if not args["server_IP"] or not args["Port"] or not args["token"]:
        print("Error: --server-IP, --Port, and --token are required")
        sys.exit(1)

    # Initialize ThingsBoard client
    tb_client = MQTTThingsBoardClient()

    # initialize the list of class labels MobileNet SSD was trained to detect
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

    # load our serialized model from disk
    logger.debug("Loading model...")
    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

    # initialize video source
    if not args.get("input", False):
        # Use PiCamera with threading
        logger.debug("Starting the PiCamera with threading..")
        try:
            vs = PiCameraThreadingClass()
            camera = None  # No direct camera object needed
        except Exception as e:
            logger.error(f"Failed to initialize PiCamera threading: {e}")
            return
    else:
        # Use video file
        logger.debug("Starting the video..")
        if config["Thread"]:
            try:
                vs = thread.ThreadingClass(args["input"])
                camera = None  # No PiCamera object needed
            except ValueError as e:
                logger.error(f"Threading initialization failed: {e}")
                return
        else:
            vs = cv2.VideoCapture(args["input"])
            camera = None  # No PiCamera object needed
            if not vs.isOpened():
                logger.error(f"Failed to open video file: {args['input']}")
                return

    # initialize the video writer (we'll instantiate later if need be)
    writer = None

    # initialize the frame dimensions (we'll set them as soon as we read
    # the first frame from the video)
    W = None
    H = None

    # instantiate our centroid tracker, then initialize a list to store
    # each of our dlib correlation trackers, followed by a dictionary to
    # map each unique object ID to a TrackableObject
    ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
    trackers = []
    trackableObjects = {}

    # initialize the total number of frames processed thus far, along
    # with the total number of objects that have moved either up or down
    totalFrames = 0
    totalDown = 0
    totalUp = 0
    # initialize empty lists to store the counting data
    total = []
    move_out = []
    move_in = []
    out_time = []
    in_time = []

    # start the frames per second throughput estimator
    fps = FPS().start()

    # Start resource monitoring thread
    monitor_thread = threading.Thread(target=monitor_resources, args=(tb_client, args["server_IP"], args["Port"], args["token"]))
    monitor_thread.daemon = True
    monitor_thread.start()

    # Initialize FPS calculation variables
    fps_start_time = time.time()
    fps_frames = 0

    try:
        # loop over frames from the video source
        while running:
            # record start time for frame processing
            frame_start_time = time.time()

            # grab the next frame from the video source
            frame = vs.read()
            if frame is None:
                if args["input"] is not None:
                    logger.debug("End of video reached.")
                    break
                logger.warning("Failed to read frame from source, skipping...")
                continue

            # resize the frame to have a maximum width of 320 pixels
            frame = imutils.resize(frame, width=320)
            # ensure frame has 3 channels (RGB)
            if len(frame.shape) == 2 or frame.shape[2] == 1:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # if the frame dimensions are empty, set them
            if W is None or H is None:
                (H, W) = frame.shape[:2]

            # if we are supposed to be writing a video to disk, initialize
            # the writer
            if args["output"] is not None and writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(args["output"], fourcc, 30,
                                        (W, H), True)

            # initialize the current status along with our list of bounding
            # box rectangles returned by either (1) our object detector or
            # (2) the correlation trackers
            status = "Waiting"
            rects = []

            # check to see if we should run a more computationally expensive
            # object detection method to aid our tracker
            if totalFrames % args["skip_frames"] == 0:
                # set the status and initialize our new set of object trackers
                status = "Detecting"
                trackers = []

                # convert the frame to a blob and pass the blob through the
                # network and obtain the detections
                blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
                net.setInput(blob)
                detections = net.forward()

                # loop over the detections
                for i in np.arange(0, detections.shape[2]):
                    # extract the confidence (i.e., probability) associated
                    # with the prediction
                    confidence = detections[0, 0, i, 2]

                    # filter out weak detections by requiring a minimum
                    # confidence
                    if confidence > args["confidence"]:
                        # extract the index of the class label from the
                        # detections list
                        idx = int(detections[0, 0, i, 1])

                        # if the class label is not a person, ignore it
                        if CLASSES[idx] != "person":
                            continue

                        # compute the (x, y)-coordinates of the bounding box
                        # for the object
                        box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                        (startX, startY, endX, endY) = box.astype("int")

                        # construct a dlib rectangle object from the bounding
                        # box coordinates and then start the dlib correlation
                        # tracker
                        tracker = dlib.correlation_tracker()
                        rect = dlib.rectangle(startX, startY, endX, endY)
                        tracker.start_track(rgb, rect)

                        # add the tracker to our list of trackers so we can
                        # utilize it during skip frames
                        trackers.append(tracker)

            # otherwise, we should utilize our object *trackers* rather than
            # object *detectors* to obtain a higher frame processing throughput
            else:
                # loop over the trackers
                for tracker in trackers:
                    # set the status of our system to be 'tracking' rather
                    # than 'waiting' or 'detecting'
                    status = "Tracking"

                    # update the tracker and grab the updated position
                    tracker.update(rgb)
                    pos = tracker.get_position()

                    # unpack the position object
                    startX = int(pos.left())
                    startY = int(pos.top())
                    endX = int(pos.right())
                    endY = int(pos.bottom())

                    # add the bounding box coordinates to the rectangles list
                    rects.append((startX, startY, endX, endY))

            # draw a horizontal line in the center of the frame -- once an
            # object crosses this line we will determine whether they were
            # moving 'up' or 'down'
            cv2.line(frame, (0, H // 2), (W, H // 2), (0, 0, 0), 3)
            # cv2.putText(frame, "-Prediction border - Entrance-", (10, H - ((i * 20) + 200)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # use the centroid tracker to associate the (1) old object
            # centroids with (2) the newly computed object centroids
            objects = ct.update(rects)

            # loop over the tracked objects
            with lock:
                for (objectID, centroid) in objects.items():
                    # check to see if a trackable object exists for the current
                    # object ID
                    to = trackableObjects.get(objectID, None)

                    # if there is no existing trackable object, create one
                    if to is None:
                        to = TrackableObject(objectID, centroid)

                    # otherwise, there is a trackable object so we can utilize it
                    # to determine direction
                    else:
                        # the difference between the y-coordinate of the *current*
                        # centroid and the mean of *previous* centroids will tell
                        # us in which direction the object is moving (negative for
                        # 'up' and positive for 'down')
                        y = [c[1] for c in to.centroids]
                        direction = centroid[1] - np.mean(y)
                        to.centroids.append(centroid)

                        # check to see if the object has been counted or not
                        if not to.counted:
                            # if the direction is negative (indicating the object
                            # is moving up) AND the centroid is above the center
                            # line, count the object
                            if direction < 0 and centroid[1] < H // 2:
                                totalUp += 1
                                date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                                move_out.append(totalUp)
                                out_time.append(date_time)
                                to.counted = True
                                tb_client.send_telemetry(args["server_IP"], args["Port"], args["token"], "exited_people", totalUp)
                                tb_client.send_telemetry(args["server_IP"], args["Port"], args["token"], "entered_people", totalDown)
                                tb_client.send_telemetry(args["server_IP"], args["Port"], args["token"], "people_inside", totalDown - totalUp)
                                logger.debug(f"Counted OUT: Total OUT = {totalUp}")

                            # if the direction is positive (indicating the object
                            # is moving down) AND the centroid is below the center
                            # line, count the object
                            elif direction > 0 and centroid[1] > H // 2:
                                totalDown += 1
                                date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                                move_in.append(totalDown)
                                in_time.append(date_time)
                                # if the people limit exceeds over threshold, send an email alert
                                if sum(total) >= config["Threshold"]:
                                    cv2.putText(frame, "-ALERT: People limit exceeded-", (10, frame.shape[0] - 80),
                                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)
                                    if config["ALERT"]:
                                        logger.debug("Sending email alert..")
                                        email_thread = threading.Thread(target=send_mail)
                                        email_thread.daemon = True
                                        email_thread.start()
                                        logger.debug("Alert sent!")
                                to.counted = True
                                tb_client.send_telemetry(args["server_IP"], args["Port"], args["token"], "entered_people", totalDown)
                                tb_client.send_telemetry(args["server_IP"], args["Port"], args["token"], "exited_people", totalUp)
                                tb_client.send_telemetry(args["server_IP"], args["Port"], args["token"], "people_inside", totalDown - totalUp)
                                logger.debug(f"Counted IN: Total IN = {totalDown}")

                            # compute the sum of total people inside
                            # total = []
                            # total.append(len(move_in) - len(move_out))

                    # store the trackable object in our dictionary
                    trackableObjects[objectID] = to

                    # draw both the ID of the object and the centroid of the
                    # object on the output frame
                    text = "ID {}".format(objectID)
                    cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)

            # construct a tuple of information we will be displaying on the frame
            info_status = [
                ("Exit", totalUp),
                ("Enter", totalDown),
                ("Status", status),
            ]

            # info_total = [
            #     ("Total people inside", ', '.join(map(str, total))),
            #     ("FPS", "N/A" if totalFrames < 1 else f"{totalFrames / (time.time() - start_time):.2f}"),
            # ]

            # display the output
            for (i, (k, v)) in enumerate(info_status):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (10, H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # for (i, (k, v)) in enumerate(info_total):
            #     text = "{}: {}".format(k, v)
            #     cv2.putText(frame, text, (165, H - ((i * 20) + 60)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # initiate a simple log to save the counting data
            if config["Log"]:
                log_data(move_in, in_time, move_out, out_time)

            # check to see if we should write the frame to disk
            if writer is not None:
                writer.write(frame)

            # show the output frame
            cv2.imshow("Real-Time Monitoring/Analysis Window", frame)
            cv2.waitKey(100)  # Increased for GUI stability
            key = cv2.waitKey(1) & 0xFF
            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                running = False
                break
            # increment the total number of frames processed thus far and
            # then update the FPS counter
            totalFrames += 1
            fps_frames += 1
            fps.update()

            # Send FPS telemetry every 10 seconds
            with lock:
                elapsed_time = time.time() - fps_start_time
                if elapsed_time >= 10:
                    current_fps = fps_frames / elapsed_time if elapsed_time > 0 else 0
                    fps_values.append(current_fps)
                    tb_client.send_telemetry(args["server_IP"], args["Port"], args["token"], "FPS", round(current_fps, 2))
                    fps_start_time = time.time()
                    fps_frames = 0

            # initiate the timer
            if config["Timer"]:
                # automatic timer to stop the live stream (set to 8 hours/28800s)
                end_time = time.time()
                num_seconds = (end_time - start_time)
                if num_seconds > 28800:
                    running = False
                    break

    finally:
        # log final elapsed time
        elapsed_time = time.time() - start_time
        logger.debug("Elapsed time: {:.2f}".format(elapsed_time))
        logger.debug("Approx. FPS: {:.2f}".format(totalFrames / elapsed_time if elapsed_time > 0 else 0))

        # release the video source
        if vs is not None:
            vs.release()

        # keep the last frame visible for a few seconds
        logger.debug("Displaying last frame for 3 seconds...")
        time.sleep(3)

        # close any open windows
        cv2.destroyAllWindows()
        if tb_client:
            tb_client.disconnect()
        if writer is not None:
            writer.release()
        # Print average resource usage and FPS
        if cpu_usages:
            print(f"Average CPU Usage: {sum(cpu_usages)/len(cpu_usages):.2f}%")
        if memory_usages:
            print(f"Average Memory Usage: {sum(memory_usages)/len(memory_usages):.2f}%")
        if temperatures:
            print(f"Average Temperature: {sum(temperatures)/len(temperatures):.2f}°C")
        if fps_values:
            print(f"Average FPS: {sum(fps_values)/len(fps_values):.2f}")

# initiate the scheduler
if config["Scheduler"]:
    # runs every day at 09:00
    schedule.every().day.at("09:00").do(people_counter)
    while True:
        schedule.run_pending()
else:
    people_counter()