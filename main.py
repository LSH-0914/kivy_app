from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.image import Image as KivyImage
from kivy.uix.button import Button
from kivy.uix.videoplayer import VideoPlayer
from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout
from kivy.uix.popup import Popup
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.core.window import Window
from kivy.lang import Builder
import cv2
import os
from PIL import Image
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense,Dropout
from tensorflow.keras.callbacks import TensorBoard
from keras.optimizers import Adam
from kivy.uix.screenmanager import Screen, ScreenManager


mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False  # Image is no longer writeable
    results = model.process(image)  # Make prediction
    image.flags.writeable = True  # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                              mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                              )
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                              )
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                              )
    # Draw right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                              )
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('Sign_Language_Data_v2')

# Actions that we try to detect
actions = np.array(['Apa Khabar', 'Selamat Jalan','Terima Kasih','Satu','Dua','Tiga','Sepuluh','Seratus','Tahniah','Maaf'])

# Thirty videos worth of data
no_sequences = 30

# Videos are going to be 30 frames in length
sequence_length = 30

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='tanh', input_shape=(30, 1662)))
model.add(Dropout(0.2))  # Adding dropout for regularization
model.add(LSTM(128, return_sequences=True, activation='tanh'))
model.add(Dropout(0.2))
model.add(LSTM(64, return_sequences=False, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(actions.shape[0], activation='softmax'))

model.load_weights('sign_language_model_v2.h5')

# colors = [(245,117,16), (117,245,16), (16,117,245),(174, 65, 232),(208, 229, 29)]
colors = (245, 117, 16)
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        # cv2.rectangle(output_frame, (0,60+num*40), (int(prob*200), 90+num*40), colors[num], -1)
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 200), 90 + num * 40), colors, -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                    cv2.LINE_AA)

    return output_frame
# Initialize mediapipe
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Define a video capture object
cap = cv2.VideoCapture(0)

# New detection variables
sequence = []
sentence = []
predictions = []
threshold = 0.7

class CameraApp(App):
    def build(self):
        # Set the initial dimensions of the Kivy window
        Window.size = (385, 650)
        # Window.clearcolor = (1, 1, 1, 1)  # Set background color to white

        # Create the main layout
        layout = BoxLayout(orientation='vertical')

        # Create a widget for displaying the camera feed
        self.image_widget = KivyImage()
        layout.add_widget(self.image_widget)

        # Create a label to display text under the camera
        self.text_label = Label(text="", font_size=20, size_hint_y=0.2)
        layout.add_widget(self.text_label)

        # Create a navigation bar
        nav_bar = BoxLayout(size_hint=(1, None), height=50, padding=[10, 5, 10, 5])

        # Create a button for the main tab
        main_tab_button = Button(text="Recognition", on_press=self.switch_to_main_tab, background_color=(0.6, 0.6, 0.6, 1))
        nav_bar.add_widget(main_tab_button)

        # Create a button for the actions tab
        actions_tab_button = Button(text="Support Words", on_press=self.switch_to_actions_tab,
                                    background_color=(0.6, 0.6, 0.6, 1))
        nav_bar.add_widget(actions_tab_button)

        # Add the navigation bar to the main layout
        layout.add_widget(nav_bar)

        # Create a grid layout for the actions tab
        self.actions_grid = GridLayout(cols=2, spacing=10, size_hint=(1, None), height=400)

        # Add buttons for each action in the actions grid
        for action in actions:
            action_button = Button(text=action)
            self.actions_grid.add_widget(action_button)

        # Create a scroll view for the actions grid
        self.actions_scrollview = ScrollView(size_hint=(1, None), height=400)
        self.actions_scrollview.add_widget(self.actions_grid)

        # Create a popup for the actions tab
        self.actions_popup = Popup(title='Support Words', content=self.actions_scrollview, size_hint=(None, None), size=(400, 500))

        self.action_videos = {
            'Apa Khabar': 'video\\Apa Khabar.mp4',
            'Selamat Jalan': 'video\\Selamat Jalan.mp4',
            'Terima Kasih': 'video\\Terima Kasih.mp4',
            'Satu': 'video\\video2.mp4',
            'Dua': 'video\\video2.mp4',
            'Tiga': 'video\\video2.mp4',
            'Sepuluh': 'video\\video2.mp4',
            'Seratus': 'video\\video2.mp4',
            'Tahniah': 'video\\Tahniah.mp4',
            'Maaf': 'video\\Maaf.mp4',
        }
        # Schedule the open_camera function to be called every 0.01 seconds (10 milliseconds)
        Clock.schedule_interval(self.open_camera, 1 / 10.0)

        return layout

    def open_camera(self, dt):
        global sequence, sentence, predictions

        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)

        # Draw landmarks
        draw_styled_landmarks(image, results)

        # Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        if len(sequence) == 30:
            # Assume you have a model variable defined somewhere
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            # print(actions[np.argmax(res)])
            predictions.append(np.argmax(res))

            # Viz logic
            if np.unique(predictions[-10:])[0] == np.argmax(res):
                if res[np.argmax(res)] > threshold:
                    if len(sentence) > 0:
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])

            if len(sentence) > 3:
                sentence = sentence[-3:]

            # Viz probabilities
            image = prob_viz(res, actions, image, colors)

        # Show the sentence under the camera
        self.text_label.text = ' '.join(sentence)

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Update the Kivy Image widget with the camera feed
        buf1 = cv2.flip(image, 0)
        buf = buf1.tostring()
        texture1 = Texture.create(size=(image.shape[1], image.shape[0]), colorfmt='rgb')
        texture1.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
        self.image_widget.texture = texture1

    def switch_to_main_tab(self, instance):
        # Implement logic for switching to the main tab
        pass

    def switch_to_actions_tab(self, instance):
        # Implement logic for switching to the actions tab
        self.actions_popup.open()

        # Add the following lines to set up the action buttons
        for action_button in self.actions_grid.children:
            action_button.bind(on_press=lambda btn: self.play_video_for_action(btn.text))

    def play_video_for_action(self, action):
        # Play the video for the selected action
        video_file = self.action_videos.get(action, '')
        if video_file:
            video_player = VideoPlayer(source=video_file, state='play', size_hint=(1, 1))
            video_popup = Popup(title=action, content=video_player, size_hint=(None, None), size=(600, 400))
            video_popup.open()
        else:
            print(f"No video found for action: {action}")

# Run the Kivy application
if __name__ == '__main__':
    CameraApp().run()
