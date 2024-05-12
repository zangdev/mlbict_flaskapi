import time
import threading
import tkinter as tk

from Logic.ModelPredict import ModelPredict, check_consecutive_signals


class TextUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Realtime Signal Display")

        # Thêm thanh trượt để chọn thời gian sleep
        self.sleep_scale = tk.Scale(self.root, from_=25, to=600, orient="horizontal", label="Time Re'Predict")
        self.sleep_scale.pack()

        self.signal_label = tk.Label(self.root, text="Signal: Hold")
        self.signal_label.pack()

        self.signal_display_frame = tk.Frame(self.root)
        self.signal_display_frame.pack()

        self.signal_display_text = tk.Text(self.signal_display_frame, wrap="word", height=10, width=50)
        self.signal_display_text.pack(side="left", fill="both", expand=True)

        self.scrollbar = tk.Scrollbar(self.signal_display_frame, orient="vertical", command=self.signal_display_text.yview)
        self.scrollbar.pack(side="right", fill="y")
        self.signal_display_text.configure(yscrollcommand=self.scrollbar.set)

        self.model = ModelPredict()
        self.global_signal = None

        threading.Thread(target=self.update_signal_loop).start()

    def update_signal_loop(self):
        while True:
            signal = self.model.run()
            if self.global_signal is None:
                self.global_signal = signal
                print(time.strftime('%Y-%m-%d %H:%M:%S'))
                print("Signal: ", signal)
                print()
                self.signal_display_text.delete(1.0, "end")
                self.signal_display_text.insert("end",
                                                f"{time.strftime('%Y-%m-%d %H:%M:%S')} Signal: {signal}\n")
                self.signal_display_text.see("end")
            elif signal != "Hold" and signal != self.global_signal:
                if check_consecutive_signals(signal):
                    self.global_signal = signal
                    signal_history = []
                    print(time.strftime('%Y-%m-%d %H:%M:%S'))
                    print("Signal: ", signal)
                    print()
                    self.signal_display_text.delete(1.0, "end")
                    self.signal_display_text.insert("end",
                                                    f"{time.strftime('%Y-%m-%d %H:%M:%S')} Signal: {signal}\n")
                    self.signal_display_text.see("end")
                else:
                    print("Warning of minor fluctuations in trend: ", signal)
                    self.signal_display_text.delete(1.0, "end")
                    self.signal_display_text.insert("end",
                                                    f"{time.strftime('%Y-%m-%d %H:%M:%S')} Warning of minor fluctuations in trend: {signal}\n")
                    self.signal_display_text.see("end")
            else:
                signal_history = []
                print(time.strftime('%Y-%m-%d %H:%M:%S'))
                print("Signal: ", signal)
                print()
                self.signal_display_text.delete(1.0, "end")
                self.signal_display_text.insert("end",
                                                f"{time.strftime('%Y-%m-%d %H:%M:%S')} Signal: {signal}\n")
                self.signal_display_text.see("end")
            self.update_signal()
            # Sử dụng giá trị từ thanh trượt cho thời gian sleep
            time.sleep(self.sleep_scale.get())

    def update_signal(self):
        if self.global_signal is not None:
            self.signal_label.config(text=f"Signal: {self.global_signal}")

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    text_ui = TextUI()
    text_ui.run()
