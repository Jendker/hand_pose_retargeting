#!/usr/bin/env python

import time


class FPSCounter:
    def __init__(self):
        self.start_time = time.time()
        self.x = 5 # displays the frame rate every X second
        self.counter = 0

    def printFPS(self):
            self.counter += 1
            current_time = time.time()
            if (current_time - self.start_time) > self.x:
                print("Frequency: {}Hz".format(int(self.counter / (current_time - self.start_time))))
                self.counter = 0
                self.start_time = current_time
