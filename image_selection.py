import os
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
import numpy as np
import cv2
import config.credentials as credentials

API_KEY = credentials.API_KEY

DEFAULT_CAMERA_TOPICS = [
    "/camera_array/bottom/image_raw",
    "/camera_array/front/image_raw",
    "/zed/zed_node/left/image_rect_color",
    "/zed/zed_node/rgb/image_rect_color",
    "/zed/zed_node/right/image_rect_color",
    "/proc_simulation/bottom",
    "/proc_simulation/front",
]


class ImageSelector():
    def __init__(self, bag_paths, preselection_coeff=.1, topic_list=DEFAULT_CAMERA_TOPICS):
        """
        Initializes the ImageSelector with a directory path and image size.
        :param bag_path_list: List of paths to ros2 bags.
        :param preselection_coeff: Coefficient for image preselection.
        :param topic_list: List of topic names to extract images from."""

        self.TEMP_DIR = './temp/'
        # Ensure TEMP_DIR exists
        temp_dir_path = os.path.expanduser(self.TEMP_DIR)
        if not os.path.exists(temp_dir_path):
            os.makedirs(temp_dir_path)

        self.bag_path_list = bag_paths
        self.preselection_coeff = preselection_coeff
        self.topic_list = topic_list
        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))



    def manage_all_bags(self):
        """
        Manage all the ros2 bags.
        :return: path to the images.
        """
        if not self.bag_path_list:
            raise ValueError("No bag paths provided.")

        for bag_path in self.bag_path_list:
            self.current_bag_path = bag_path
            self._manage_bag()

        cv2.destroyAllWindows()


    def _manage_bag(self):
        """
        Manage the ros2 bag.
        """
        if not os.path.exists(self.current_bag_path):
            raise FileNotFoundError(f"Bag path {self.current_bag_path} does not exist.")

        for topic in self.topic_list:
            self.current_topic = topic
            self._deserialize_ros2_bag()


    def _display_image(self, img, index, num_images):
        display_img = img.copy()
        scale = 2
        width = int(display_img.shape[1] * scale)
        height = int(display_img.shape[0] * scale)
        display_img = cv2.resize(display_img, (width, height), interpolation=cv2.INTER_LINEAR)

        num_saved = len([f for f in os.listdir(self.TEMP_DIR) if f.endswith('.png')])
        cv2.putText(display_img, f"{num_saved} saved => {index}/{num_images}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 0), 2)

        cv2.putText(display_img, os.path.basename(self.current_bag_path),(10, 30), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 255), 2)
        cv2.putText(display_img, self.current_topic,(10, 60), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 255), 2)
        # Create a named window and set its size
        cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Image', width, height)
        cv2.imshow('Image', display_img)


    def _simplified_topic(self):
        match self.current_topic:
            case '/camera_array/bottom/image_raw':
                return 'bottom'
            case '/camera_array/front/image_raw':
                return 'front'
            case '/zed/zed_node/left/image_rect_color':
                return 'zed_left'
            case '/zed/zed_node/rgb/image_rect_color':
                return 'zed_rgb'
            case '/zed/zed_node/right/image_rect_color':
                return 'zed_right'
            case '/proc_simulation/bottom':
                return 'sim_bottom'
            case '/proc_simulation/front':
                return 'sim_front'


    def preprocess_image(self, img):
        ycrcb_img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb_img)
        y_clahe = self.clahe.apply(y)
        clahe_ycrcb = cv2.merge([y_clahe, cr, cb])
        return cv2.cvtColor(clahe_ycrcb, cv2.COLOR_YCrCb2BGR)


    def _deserialize_ros2_bag(self):
        typestore = get_typestore(Stores.LATEST)

        # Create reader instance and open for reading.
        with Reader(self.current_bag_path) as reader:
            messages = list(reader.messages())
            i = 0
            while i < len(messages):
                connection, timestamp, rawdata = messages[i]
                if connection.topic == self.current_topic:
                    msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                    img = np.frombuffer(msg.data,dtype=np.uint8)
                    if msg.encoding == "rgb8":
                        img = img.reshape((msg.height, msg.width, 3))
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    elif msg.encoding == "bgr8":
                        img = img.reshape((msg.height, msg.width, 3))
                    elif msg.encoding == "rgba8":
                        img = img.reshape((msg.height, msg.width, 4))
                        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                    elif msg.encoding == "bgra8":
                        img = img.reshape((msg.height, msg.width, 4))
                        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                    self._display_image(img, int(i*self.preselection_coeff), int(len(messages)*self.preselection_coeff))
                    key = cv2.waitKey(0)
                    filename = str(timestamp) + '_' + self._simplified_topic() + '.png'
                    path = os.path.join(os.path.expanduser(self.TEMP_DIR), filename)
                    if key == ord('y'):
                        print(f"Saving image to {path}")
                        cv2.imwrite(path, img)
                        i += round(1/self.preselection_coeff)
                    elif key == ord('q'):
                        print("Exiting image selection.")
                        i = len(messages)
                    elif key == ord('p'):
                        print("Going to previous image.")
                        i -= round(1/self.preselection_coeff)
                        if i < 0:
                            i = 0
                    elif key == ord('n'):
                        if os.path.exists(path):
                            print(f"Removing image {path}")
                            os.remove(path)
                        i += round(1/self.preselection_coeff)
                    elif key == 27:  # ESC key
                        exit()
                    else:
                        self._print_help()
                else:
                    i = len(messages)


    def _print_help(self):
        """
        Print the help message for the user.
        """
        print("Press 'y' to save the image, 'n' to skip, 'p' to go back, 'q' to quit.")
        print("Press any other key to see this message again.")


    def __del__(self):
        """
        Destructor to clean up resources.
        """
        cv2.destroyAllWindows()


