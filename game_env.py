import base64
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import WebDriverException
import time
import cv2
import numpy as np
from io import BytesIO
from PIL import Image

Image_width, Image_height = 80, 80
Image_num = 4

class GameEnv(object):
    def __init__(self, speed = 0):
        init_script = "document.getElementsByClassName('runner-canvas')[0].id = 'runner-canvas'"
        options = Options()
        options.add_argument("--mute-audio")
        options.add_argument("disable-infobars")
        self.driver = webdriver.Chrome(executable_path=r'C:\Users\sajan poudel\webdriver\chromedriver.exe', options=options)
        self.driver.set_network_conditions(offline = True, latency = 5, 
        download_throughput = 100, upload_throughput = 10)
        try:
            self.driver.get("chrome:\\dino")
        except WebDriverException:
            self.driver.execute_script("Runner.config.ACCELERATION = %d"%speed)
            self.driver.execute_script(init_script)
            self.jump()
            print("jump function called")

    def jump(self):
        self.driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_UP)
    
    def crouch(self):
        self.driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_DOWN)
    
    def is_crashed(self):
        return self.driver.execute_script("return Runner.instance_.crashed;")

    def restart_game(self):
        self.driver.execute_script("Runner.instance_.restart();")

    def pause_game(self):
        self.driver.execute_script("Runner.instance_.stop();")

    def resume_game(self):
        self.driver.execute_script("Runner.instance_.play();")
    
    def end_game(self):
        self.driver.close()
    
    def get_score(self):
        score = self.driver.execute_script("return Runner.instance_.distanceMeter.digits;")
        score = "".join(score)
        return score

    def capture_screen(self):
        img_script = "return document.getElementById('runner-canvas').toDataURL().substring(22);"
        image = self.driver.execute_script(img_script)
        # print(image)
        screen = np.array(Image.open(BytesIO(base64.b64decode(image))))
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        screen = screen[:300, :450]
        screen = cv2.resize(screen, (80, 80))
        return screen

def show_image():
    while True:
        screen = (yield)
        title = "image shown"
        image = cv2.resize(screen, (800, 400))
        cv2.imshow(title, image)
        cv2.waitKey(1)