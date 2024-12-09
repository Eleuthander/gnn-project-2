import psutil
import torch
from tqdm import tqdm
import time
from threading import Thread
import GPUtil
import gc

class ResourceMonitor:
    def __init__(self, progress_bar, interval=5):
        """
        Initialize resource monitor
        Args:
            progress_bar (tqdm): tqdm progress bar to update
            interval (int): Monitoring interval in seconds
        """
        self.interval = interval
        self.running = False
        self.thread = None
        self.progress_bar = progress_bar
        
    def get_gpu_usage(self):
        """Get GPU utilization if available"""
        try:
            gpu = GPUtil.getGPUs()[0]  # Assuming first GPU
            return f"GPU: {gpu.load*100:.1f}% | Mem: {gpu.memoryUtil*100:.1f}%"
        except:
            return "GPU stats N/A"
    
    def get_cpu_usage(self):
        """Get CPU utilization"""
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        return f"CPU: {cpu_percent:.1f}% | RAM: {memory.percent:.1f}%"
    
    def monitor(self):
        """Main monitoring loop"""
        while self.running:
            gpu_stats = self.get_gpu_usage()
            cpu_stats = self.get_cpu_usage()
            self.progress_bar.set_postfix_str(f"{gpu_stats} | {cpu_stats}")
            time.sleep(self.interval)
    
    def start(self):
        """Start monitoring in background thread"""
        self.running = True
        self.thread = Thread(target=self.monitor, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop monitoring"""
        self.running = False
        if self.thread:
            self.thread.join()