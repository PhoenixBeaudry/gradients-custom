import time
import math
from collections import deque
from transformers import TrainerCallback, TrainerState, TrainerControl

class TimeStopCallback(TrainerCallback):
    """
    Callback to dynamically adjust max_steps based on elapsed time and average step duration.
    init_args:
      max_time_s: float  # total allowed training time in seconds

    Updates `state.max_steps` every step using a sliding window of the last 5 step durations.
    """
    def __init__(self, max_time_s: float):
        super().__init__()
        self.max_time_s = max_time_s
        self.window = 5
        self.start_time = None
        self.step_times = deque(maxlen=self.window)
        self._last_step_start = None

    def on_train_begin(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        # Record training start time
        self.start_time = time.time()
        return control

    def on_step_begin(
        self,
        args,
        state,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        # Mark the start of the current step
        self._last_step_start = time.time()
        return control

    def on_step_end(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        # Compute duration of this step
        now = time.time()
        if self._last_step_start is not None:
            self.step_times.append(now - self._last_step_start)

        # Calculate average over sliding window
        if self.step_times:
            avg_step = sum(self.step_times) / len(self.step_times)
        else:
            avg_step = 0

        # Calculate remaining time
        elapsed = now - self.start_time
        remaining_s = self.max_time_s - elapsed

        # Update max_steps based on average step time
        if avg_step > 0:
            new_max = math.floor(remaining_s / avg_step)
            state.max_steps = new_max

        return control
