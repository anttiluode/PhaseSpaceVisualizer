import numpy as np
import pyaudio
import pygame
import threading
import time
from pygame.locals import QUIT, KEYDOWN, K_ESCAPE
from tkinter import Tk, ttk, StringVar, messagebox, IntVar, DoubleVar
from collections import deque

class AudioVisualizer:
    def __init__(self, sample_rate=44100, buffer_size=1024):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.audio = pyaudio.PyAudio()
        self.input_device = None
        self.output_device = None
        self.running = False
        self.audio_stream = None
        self.phase_data = np.zeros((self.buffer_size,), dtype=np.float32)
        self.prev_phase_data = np.zeros((self.buffer_size,), dtype=np.float32)

    def list_devices(self):
        devices = {}
        for i in range(self.audio.get_device_count()):
            try:
                dev_info = self.audio.get_device_info_by_index(i)
                devices[dev_info['name']] = i
            except Exception as e:
                print(f"Error accessing device index {i}: {e}")
        return devices

    def setup_audio(self):
        if not self.input_device:
            raise ValueError("Input device not set.")
        self.audio_stream = self.audio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.input_device,
            frames_per_buffer=self.buffer_size
        )

    def read_audio(self):
        if not self.audio_stream:
            return np.zeros(self.buffer_size)
        try:
            data = np.frombuffer(self.audio_stream.read(self.buffer_size, exception_on_overflow=False), dtype=np.float32)
            return data
        except Exception as e:
            print(f"Audio read error: {e}")
            return np.zeros(self.buffer_size)

    def close_audio(self):
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
        self.audio.terminate()

    def start(self):
        self.running = True
        self.audio_thread = threading.Thread(target=self.audio_loop, daemon=True)
        self.audio_thread.start()

    def stop(self):
        self.running = False
        if self.audio_thread.is_alive():
            self.audio_thread.join()
        self.close_audio()

    def audio_loop(self):
        while self.running:
            self.prev_phase_data = self.phase_data.copy()
            self.phase_data = self.read_audio()
            time.sleep(self.buffer_size / self.sample_rate)

class PhaseSpaceVisualizer:
    def __init__(self, visualizer, trail_length=100, color_scheme='Rainbow', dot_size=3, sample_rate=10):
        self.visualizer = visualizer
        self.trail_length = trail_length
        self.color_scheme = color_scheme
        self.dot_size = dot_size
        self.sample_rate = sample_rate  # Sampling rate for plotting
        self.width, self.height = 800, 800
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Audio Phase Space Visualizer")
        self.clock = pygame.time.Clock()
        self.trail = deque(maxlen=self.trail_length)  # Store (prev_sample, current_sample) tuples
        self.color_schemes = {
            'Rainbow': self.generate_rainbow_colors(self.trail_length),
            'Monochrome': self.generate_monochrome_colors(),
            'Fire': self.generate_fire_colors(self.trail_length),
            'Ocean': self.generate_ocean_colors(self.trail_length),
            'Green Gradient': self.generate_green_gradient_colors(self.trail_length)  # Correctly named
        }

    def generate_rainbow_colors(self, length):
        colors = []
        for i in range(length):
            color = pygame.Color(0, 0, 0)
            hue = (i * 360 / length) % 360  # Ensure hue is within 0-360
            color.hsva = (hue, 100, 100, 100)  # Hue, Saturation, Value, Alpha
            colors.append(color)
        return colors

    def generate_monochrome_colors(self):
        # All dots will be white
        return [(255, 255, 255) for _ in range(self.trail_length)]

    def generate_fire_colors(self, length):
        colors = []
        for i in range(length):
            r = min(255, int(255 * (i / length) * 2))
            g = min(255, int(255 * (i / length)))
            b = 0
            colors.append((r, g, b))
        return colors

    def generate_ocean_colors(self, length):
        colors = []
        for i in range(length):
            r = 0
            g = min(255, int(255 * (i / length)))
            b = min(255, int(255 * (1 - i / length)))
            colors.append((r, g, b))
        return colors

    def generate_green_gradient_colors(self, length):
        colors = []
        for i in range(length):
            r = 0
            g = min(255, int(255 * (i / length)))
            b = 0
            colors.append((r, g, b))
        return colors

    def update_color_scheme(self):
        if self.color_scheme not in self.color_schemes:
            self.color_scheme = 'Rainbow'
        # Update the colors list based on the current color scheme and trail length
        if self.color_scheme == 'Rainbow':
            self.color_schemes['Rainbow'] = self.generate_rainbow_colors(self.trail_length)
        elif self.color_scheme in ['Fire', 'Ocean', 'Green Gradient']:
            method_name = f"generate_{self.color_scheme.lower().replace(' ', '_')}_colors"
            if hasattr(self, method_name):
                self.color_schemes[self.color_scheme] = getattr(self, method_name)(self.trail_length)
            else:
                print(f"Color scheme method {method_name} not found. Falling back to Rainbow.")
                self.color_scheme = 'Rainbow'
                self.color_schemes['Rainbow'] = self.generate_rainbow_colors(self.trail_length)

    def draw_phase_space(self):
        self.screen.fill((0, 0, 0))
        current = self.visualizer.phase_data
        previous = self.visualizer.prev_phase_data

        # Sample every 'self.sample_rate' samples to limit the number of points
        for i in range(0, len(current), self.sample_rate):
            prev_sample = previous[i]
            curr_sample = current[i]
            self.trail.append((prev_sample, curr_sample))

        # Update color scheme
        self.update_color_scheme()
        colors = self.color_schemes.get(self.color_scheme, self.color_schemes['Rainbow'])

        # Draw the trail
        for idx, (prev, curr) in enumerate(self.trail):
            # Normalize data to fit the screen (-1 to 1 assumed)
            x = int(self.width / 2 + prev * self.width / 2)
            y = int(self.height / 2 + curr * self.height / 2)
            # Clamp values to screen boundaries
            x = max(0, min(self.width - 1, x))
            y = max(0, min(self.height - 1, y))
            # Get color based on the age of the point
            color = colors[idx % len(colors)]
            pygame.draw.circle(self.screen, color, (x, y), self.dot_size)

        pygame.display.flip()

    def run(self):
        self.visualizer.start()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                    running = False
            self.draw_phase_space()
            self.clock.tick(60)  # Limit to 60 FPS
        self.visualizer.stop()
        pygame.quit()

class ConfigUI:
    def __init__(self, visualizer):
        self.visualizer = visualizer
        self.root = Tk()
        self.root.title("Audio Visualizer Config")
        self.input_var = StringVar()
        self.output_var = StringVar()
        self.trail_length_var = IntVar(value=100)
        self.color_scheme_var = StringVar(value='Rainbow')
        self.dot_size_var = IntVar(value=3)

        devices = self.visualizer.list_devices()
        self.input_devices = {name: idx for name, idx in devices.items() if self.visualizer.audio.get_device_info_by_index(idx)['maxInputChannels'] > 0}
        self.output_devices = {name: idx for name, idx in devices.items() if self.visualizer.audio.get_device_info_by_index(idx)['maxOutputChannels'] > 0}

        self.create_widgets()

    def create_widgets(self):
        # Input Device Selection
        ttk.Label(self.root, text="Input Device:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        input_combobox = ttk.Combobox(self.root, textvariable=self.input_var, values=list(self.input_devices.keys()), state="readonly", width=50)
        input_combobox.grid(row=0, column=1, padx=5, pady=5)
        if self.input_devices:
            input_combobox.current(0)

        # Output Device Selection
        ttk.Label(self.root, text="Output Device:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        output_combobox = ttk.Combobox(self.root, textvariable=self.output_var, values=list(self.output_devices.keys()), state="readonly", width=50)
        output_combobox.grid(row=1, column=1, padx=5, pady=5)
        if self.output_devices:
            output_combobox.current(0)

        # Trail Length
        ttk.Label(self.root, text="Trail Length:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        trail_slider = ttk.Scale(self.root, from_=10, to=500, variable=self.trail_length_var, orient='horizontal')
        trail_slider.grid(row=2, column=1, padx=5, pady=5, sticky="we")
        trail_value = ttk.Label(self.root, textvariable=self.trail_length_var)
        trail_value.grid(row=2, column=2, padx=5, pady=5, sticky="w")

        # Color Scheme
        ttk.Label(self.root, text="Color Scheme:").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        color_combobox = ttk.Combobox(self.root, textvariable=self.color_scheme_var, values=list(
            ['Rainbow', 'Monochrome', 'Fire', 'Ocean', 'Green Gradient']), state="readonly", width=48)
        color_combobox.grid(row=3, column=1, padx=5, pady=5)
        color_combobox.current(0)

        # Dot Size
        ttk.Label(self.root, text="Dot Size:").grid(row=4, column=0, padx=5, pady=5, sticky="w")
        dot_slider = ttk.Scale(self.root, from_=1, to=10, variable=self.dot_size_var, orient='horizontal')
        dot_slider.grid(row=4, column=1, padx=5, pady=5, sticky="we")
        dot_value = ttk.Label(self.root, textvariable=self.dot_size_var)
        dot_value.grid(row=4, column=2, padx=5, pady=5, sticky="w")

        # Start Button
        start_button = ttk.Button(self.root, text="Start Visualizer", command=self.start_visualizer)
        start_button.grid(row=5, column=0, columnspan=3, pady=10)

        # Configure grid to make widgets expand properly
        self.root.columnconfigure(1, weight=1)

    def start_visualizer(self):
        input_device_name = self.input_var.get()
        output_device_name = self.output_var.get()
        trail_length = self.trail_length_var.get()
        color_scheme = self.color_scheme_var.get()
        dot_size = self.dot_size_var.get()

        if not input_device_name or not output_device_name:
            messagebox.showerror("Error", "Please select both input and output devices.")
            return

        self.visualizer.input_device = self.input_devices[input_device_name]
        self.visualizer.output_device = self.output_devices[output_device_name]
        try:
            self.visualizer.setup_audio()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize audio: {e}")
            return

        # Destroy the config window and start the visualizer
        self.root.destroy()

        # Initialize and run the phase space visualizer with selected settings
        phase_space = PhaseSpaceVisualizer(
            visualizer=self.visualizer,
            trail_length=trail_length,
            color_scheme=color_scheme,
            dot_size=dot_size
        )
        phase_space.run()

def main():
    visualizer = AudioVisualizer()
    config_ui = ConfigUI(visualizer)
    config_ui.root.mainloop()

if __name__ == "__main__":
    main()
