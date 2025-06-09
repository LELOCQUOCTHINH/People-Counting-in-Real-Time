# People-Counting-in-Real-Time

<div align="center">
<img src=https://imgur.com/SaF1kk3.gif" width=550>
<p>Live demo</p>
</div>

## Overview
This project implements a people counting system for buses using a Raspberry Pi Zero 2W as the edge computing device. It utilizes the MobileNet SSD model for person detection, a 5MP Pi Camera for capturing video frames, and MQTT to send processed data to a ThingsBoard dashboard for real-time monitoring. The system tracks people entering and exiting the bus, calculates the number of people inside, logs counting data, sends email alerts when a threshold is exceeded, and monitors system resources (CPU, memory, and temperature).

> NOTE: This is an improvement/modification to https://www.pyimagesearch.com/2018/08/13/opencv-people-counter/

## Repository Contents
- **people_counter.py**: Main script for processing video from the Pi Camera or a video file. It performs person detection, tracking, counting, logging, email alerts, and sends telemetry data to ThingsBoard.
- **postTelemetry_mqtt_tb.py**: Utility script for handling MQTT communication with the ThingsBoard server to send telemetry data.
- **mailer.py**: Utility script for sending email alerts when the number of people inside exceeds a configured threshold.
- **thread.py**: Utility script for threaded video file reading to improve performance when using video file input.
- **trackableobject.py**: Defines the `TrackableObject` class for storing object IDs, centroids, and counting status.
- **centroidtracker.py**: Implements the `CentroidTracker` class for tracking objects based on centroid distances.
- **utils/config.json**: Configuration file for email settings, alert thresholds, logging, and scheduling options.

## Simple Theory

### SSD detector

- We are using a SSD ```Single Shot Detector``` with a MobileNet architecture. In general, it only takes a single shot to detect whatever is in an image. That is, one for generating region proposals, one for detecting the object of each proposal. 
- Compared to other two shot detectors like R-CNN, SSD is quite fast.
- ```MobileNet```, as the name implies, is a DNN designed to run on resource constrained devices. For e.g., mobiles, ip cameras, scanners etc.
- Thus, SSD seasoned with a MobileNet should theoretically result in a faster, more efficient object detector.

### Centroid tracker

- Centroid tracker is one of the most reliable trackers out there.
- To be straightforward, the centroid tracker computes the ```centroid``` of the bounding boxes.
- That is, the bounding boxes are ```(x, y)``` co-ordinates of the objects in an image. 
- Once the co-ordinates are obtained by our SSD, the tracker computes the centroid (center) of the box. In other words, the center of an object.
- Then an ```unique ID``` is assigned to every particular object deteced, for tracking over the sequence of frames.

---

## Features
- **Person Detection**: Uses MobileNet SSD for lightweight and efficient person detection on the Raspberry Pi Zero 2W.
- **Tracking**: Employs DLib’s correlation tracker and a custom centroid tracker to follow detected persons across frames.
- **Counting Logic**: Counts people crossing a horizontal virtual line to determine entries ("Enter") and exits ("Exit") from the bus.
- **Telemetry**: Sends real-time data (entry/exit counts, people inside, CPU usage, memory usage, temperature, and FPS) to a ThingsBoard dashboard.
- **Email Alerts**: Sends email notifications when the number of people inside exceeds a threshold defined in `config.json`.
- **Data Logging**: Logs entry and exit events with timestamps to a CSV file for record-keeping.
- **Resource Monitoring**: Tracks CPU, memory, and temperature usage on the Raspberry Pi Zero 2W for performance optimization.
- **Threaded Camera Input**: Uses a custom `PiCameraThreadingClass` for efficient frame capture from the Pi Camera.
- **Frame Optimization**: Processes frames at a reduced resolution (320px width) and skips frames to optimize performance on the Raspberry Pi Zero 2W.
- **Scheduling**: Optional daily scheduling to run the system at specific times (e.g., 09:00 daily) via `config.json`.

## Hardware Requirements
- **Raspberry Pi Zero 2W**: Acts as the edge computing device.
- **Pi Camera Module (5MP)**: Captures video frames for person detection.
- **Internet Connection**: Required for sending telemetry data to ThingsBoard and email alerts.

## Software Requirements
- **Python 3.7+**
- **OpenCV (with contrib)**: For computer vision tasks and DLib tracking.
  ```bash
  pip install opencv-contrib-python
  ```
- **Picamera2**: For interfacing with the Pi Camera.
  ```bash
  pip install picamera2
  ```
- **Paho MQTT**: For communication with ThingsBoard.
  ```bash
  pip install paho-mqtt
  ```
- **Psutil**: For resource monitoring.
  ```bash
  pip install psutil
  ```
- **Imutils**: For image processing utilities.
  ```bash
  pip install imutils
  ```
- **DLib**: For correlation tracking.
  ```bash
  pip install dlib
  ```
- **Schedule**: For scheduling daily runs.
  ```bash
  pip install schedule
  ```
- **MobileNet SSD Model Files**:
  - Download `deploy.prototxt` and the pre-trained `MobileNetSSD_deploy.caffemodel` from a trusted source (e.g., OpenCV repository).
  - Or you can use availabel `deploy.prototxt` and `MobileNetSSD_deploy.caffemodel` in this repo.
  - Place them in the project directory.
- **Configuration File**:
  - Create or modify the `utils/config.json` file with the following structure:
    ```json
    {
      "Email_Send": "your_email@gmail.com",
      "Email_Password": "your_app_password",
      "Email_Receive": "recipient_email@example.com",
      "Threshold": 10,
      "ALERT": true,
      "Log": true,
      "Scheduler": false,
      "Timer": true,
      "Thread": true
    }
    ```
  - Replace `Email_Send` and `Email_Password` with your Gmail credentials (use an App Password for Gmail).
  - Set `Email_Receive` to the recipient’s email for alerts.
  - Adjust `Threshold` for the maximum number of people before an alert is triggered.
  - Enable/disable features like `ALERT`, `Log`, `Scheduler`, and `Timer` as needed.

## Setup Instructions
1. **Install Dependencies**:
   Ensure all required Python packages are installed using the commands above.
2. **Configure ThingsBoard**:
   - Set up a ThingsBoard server (e.g., `app.coreiot.io` or a local instance).
   - Create a device in ThingsBoard and obtain the **device access token**.
   - Note the server IP and MQTT port (e.g., 1883).
3. **Prepare MobileNet SSD Model**:
   - Download `deploy.prototxt` and `MobileNetSSD_deploy.caffemodel`.
   - Place them in the project directory.
4. **Configure Email**:
   - Update `utils/config.json` with your email credentials and settings.
   - Ensure Gmail’s “Less secure app access” is off and use an App Password for `Email_Password`.
5. **Connect Pi Camera**:
   - Ensure the 5MP Pi Camera is properly connected to the Raspberry Pi Zero 2W.
   - Enable the camera interface in `raspi-config`.
6. **Run the Script**:
   - For Pi Camera operation:
     ```bash
     python3 people_counter.py --model MobileNetSSD_deploy.caffemodel --prototxt deploy.prototxt --server-IP <THINGSBOARD_IP> --Port <MQTT_PORT> --token <DEVICE_TOKEN>
     ```
   - For testing with a video file:
     ```bash
     python3 people_counter.py --model MobileNetSSD_deploy.caffemodel --prototxt deploy.prototxt --input <VIDEO_FILE_PATH> --server-IP <THINGSBOARD_IP> --Port <MQTT_PORT> --token <DEVICE_TOKEN>
     ```
   - To save output video (optional):
     ```bash
     python3 people_counter.py --model MobileNetSSD_deploy.caffemodel --prototxt deploy.prototxt --output output.mp4 --server-IP <THINGSBOARD_IP> --Port <MQTT_PORT> --token <DEVICE_TOKEN>
     ```
7. **View Dashboard**:
   - Access the ThingsBoard dashboard to monitor:
     - Number of people entering (`entered_people`)
     - Number of people exiting (`exited_people`)
     - Current people inside (`people_inside`)
     - System metrics (CPU usage, memory usage, temperature, FPS)
     - Example:
        ![Screenshot 2025-06-03 211056](https://github.com/user-attachments/assets/cbb0fb58-f331-4a4d-9ebc-9c0204fe77c5)
        ![Screenshot 2025-06-03 211103](https://github.com/user-attachments/assets/ce96f62d-67c0-4cc1-a9c0-4bd79099c53a)
    
        - You can view the example dashboard via this link: [My Dashboard](https://app.coreiot.io/dashboard/5eef5c50-3ca9-11f0-aae0-0f85903b3644?publicId=00e331c0-f1ec-11ef-87b5-21bccf7d29d5).
        - if you want to manipulate with [My Dashboard](https://app.coreiot.io/dashboard/5eef5c50-3ca9-11f0-aae0-0f85903b3644?publicId=00e331c0-f1ec-11ef-87b5-21bccf7d29d5), you can run this command:
          
        ```bash
         python3 people_counter.py --model MobileNetSSD_deploy.caffemodel --prototxt deploy.prototxt --input test1.mp4 --server-IP app.coreiot.io --Port 1883 --token I1WYm7V1FMBsKgBLMJVL
        ```
        
        (For testing with test1.mp4).
8. **Check Logs**:
   - Counting data is saved to `utils/data/logs/counting_data.csv` if `Log` is enabled in `config.json`.

## Usage Notes
- **Single Camera**: Optimized for the Raspberry Pi Zero 2W with low-resolution processing (320px width) and frame skipping (default: 20 frames).
- **Counting Logic**: A horizontal line is drawn at the frame’s vertical center. People moving upward past the line are counted as "Exit," and downward as "Enter."
- **Email Alerts**: Triggered when the number of people inside exceeds the `Threshold` in `config.json`. Emails are sent asynchronously to avoid blocking.
- **Data Logging**: Entry and exit events with timestamps are saved to a CSV file for analysis.
- **Scheduling**: If `Scheduler` is enabled in `config.json`, the system runs daily at 09:00. Otherwise, it runs immediately.
- **Timer**: If `Timer` is enabled, the system stops after 8 hours (28,800 seconds) to prevent indefinite running.
- **Performance**: Frame skipping and low resolution optimize performance on the Raspberry Pi Zero 2W. Adjust `--skip-frames` or resolution in `people_counter.py` for tuning.
- **Exit**: Press `q` to quit or use `Ctrl+C` to gracefully exit, displaying average resource usage and FPS.


## Features

### Real-Time alert

If selected, we send an email alert in real-time. Example use case: If the total number of people (say 10 or 30) are exceeded in a store/building, we simply alert the staff. 

- You can set the max. people limit in config, e.g., ```"Threshold": 10```.
- This is quite useful considering scenarios similar to COVID-19. Below is an example:
<img src="https://imgur.com/35Yf1SR.png" width=350>

> ***1. Setup your emails:***

In the config, setup your sender email ```"Email_Send": ""``` to send the alerts and your receiver email ```"Email_Receive": ""``` to receive the alerts.

> ***2. Setup your password:***

Similarly, setup the sender email password ```"Email_Password": ""```.

Note that the password varies if you have secured 2 step verification turned on, so refer the links below and create an application specific password:

- Google mail has a guide here: https://myaccount.google.com/lesssecureapps
- For 2 step verified accounts: https://support.google.com/accounts/answer/185833

### Threading

- Multi-Threading is implemented in ```utils/thread.py```. If you ever see a lag/delay in your real-time stream, consider using it.
- Threading removes ```OpenCV's internal buffer``` (which basically stores the new frames yet to be processed until your system processes the old frames) and thus reduces the lag/increases fps.
- If your system is not capable of simultaneously processing and outputting the result, you might see a delay in the stream. This is where threading comes into action.
- It is most suitable to get solid performance on complex real-time applications. To use threading: set ```"Thread": true,``` in config.

### Scheduler

- Automatic scheduler to start the software. Configure to run at every second, minute, day, or workdays e.g., Monday to Friday.
- This is extremely useful in a business scenario, for instance, you could run the people counter only at your desired time (maybe 9-5?).
- Variables and any cache/memory would be reset, thus, less load on your machine.

```python
# runs at every day (09:00 am)
schedule.every().day.at("9:00").do(run)
```

### Timer

- Configure stopping the software execution after a certain time, e.g., 30 min or 8 hours (currently set) from now.
- All you have to do is set your desired time and run the script.

```python
# automatic timer to stop the live stream (set to 8 hours/28800s)
end_time = time.time()
num_seconds = (end_time - start_time)
if num_seconds > 28800:
    break
```

### Simple log

- Logs the counting data at end of the day.
- Useful for footfall analysis. Below is an example:
<img src="https://imgur.com/CV2nCjx.png" width=400>

---

## Limitations
- **Single Camera**: Assumes a single entry/exit point. Multiple entry points may require additional cameras or logic.
- **MobileNet SSD**: May miss detections in low-light or crowded scenes. Adjust `--confidence` (default: 0.4) for better accuracy.
- **Raspberry Pi Zero 2W**: Limited processing power may lead to lower FPS. Tune frame skip and resolution for performance.
- **Email Alerts**: Requires a stable internet connection and proper Gmail App Password configuration.
- **CSV Logging**: Overwrites `counting_data.csv` on each run. Modify `log_data` in `people_counter.py` to append instead if needed.

## Future Improvements
- Support multiple entry/exit points with additional cameras.
- Implement appending to CSV logs instead of overwriting.
- Enhance MobileNet SSD detection with custom training for bus-specific scenarios.
- Add support for more robust tracking algorithms (e.g., DeepSORT).
- Improve error handling for network issues in MQTT and email communication.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For questions or contributions, please contact [thinhle.hardware@gmail.com](mailto:thinhle.hardware@gmail.com) or
[My Linkedin](https://www.linkedin.com/in/lelocquocthinh/) or open an issue on the repository.

## References

***Main:***

- SSD paper: https://arxiv.org/abs/1512.02325
- MobileNets paper: https://arxiv.org/abs/1704.04861
- Centroid tracker: https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/

***Optional:***

- Object detection with SSD/MobileNets: https://pyimagesearch.com/2017/09/11/object-detection-with-deep-learning-and-opencv/
- Schedule: https://pypi.org/project/schedule/

---

*saimj7/ 19-08-2020 - © <a href="http://saimj7.github.io" target="_blank">Sai_Mj</a>.*

*LeLocQuocThinh/ 06-2025 - © <a href="https://github.com/LELOCQUOCTHINH" target="_blank">LLQT</a>.*
