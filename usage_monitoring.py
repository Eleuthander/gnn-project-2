import psutil
import torch
from tqdm import tqdm
import time
from threading import Thread
import GPUtil
import gc

class ResourceMonitor:
    def __init__(self, progress_bar, postfix_dict, interval=5):
        self.interval = interval
        self.running = False
        self.thread = None
        self.progress_bar = progress_bar
        self.postfix = postfix_dict  # Shared dictionary
        
    def get_gpu_usage(self):
        try:
            gpu = GPUtil.getGPUs()[0]
            return f"GPU: {gpu.load*100:.1f}% | Mem: {gpu.memoryUtil*100:.1f}%"
        except:
            return "GPU stats N/A"
    
    def get_cpu_usage(self):
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        return f"CPU: {cpu_percent:.1f}% | RAM: {memory.percent:.1f}%"
    
    def monitor(self):
        while self.running:
            self.postfix['resources'] = f"{self.get_gpu_usage()} | {self.get_cpu_usage()}"
            self.progress_bar.set_postfix(self.postfix)
            time.sleep(self.interval)
    
    def start(self):
        self.running = True
        self.thread = Thread(target=self.monitor, daemon=True)
        self.thread.start()
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()