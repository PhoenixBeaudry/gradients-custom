import time
import math
import logging
from collections import deque
from pydantic import BaseModel
from transformers.trainer_callback import TrainerCallback
from ..base import BasePlugin

LOGGER = logging.getLogger("axolotl.integrations.time_stop")

class TimeStopArgs(BaseModel):
    """
    Input args for TimeStop plugin.
    """
    max_time_s: float  # total allowed training time in seconds

class TimeStopCallbackHandler(TrainerCallback):
    """
    Transformer TrainerCallback to dynamically limit max_steps based on elapsed time.
    Uses a sliding window over the last 5 step durations.
    """
    def __init__(self, max_time_s: float):
        super().__init__()
        self.max_time_s = max_time_s
        self.window = 5
        self.start_time = None
        self.step_times = deque(maxlen=self.window)
        self._last_step_start = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        LOGGER.info(f"[TimeStop] Training started with budget {self.max_time_s}s")
        return control

    def on_step_begin(self, args, state, control, **kwargs):
        self._last_step_start = time.time()
        return control

    def on_step_end(self, args, state, control, **kwargs):
        now = time.time()
        if self._last_step_start is not None:
            self.step_times.append(now - self._last_step_start)

        # compute average step time
        if self.step_times:
            avg_step = sum(self.step_times) / len(self.step_times)
        else:
            avg_step = 0

        # compute remaining time
        elapsed = now - self.start_time
        remaining = self.max_time_s - elapsed

        # update max_steps
        if avg_step > 0:
            new_max = math.floor(remaining / avg_step)
            state.max_steps = new_max
            LOGGER.info(
                f"[TimeStop] step={state.global_step} avg_step={avg_step:.2f}s "
                f"elapsed={elapsed:.2f}s remaining={remaining:.2f}s -> max_steps={new_max}"
            )
        return control

class TimeStopPlugin(BasePlugin):
    """
    Plugin for integrating TimeStopCallback into Axolotl training.
    """
    def get_input_args(self):
        # return the Pydantic model path for parsing config
        return "axolotl.integrations.time_stop.TimeStopArgs"

    def add_callbacks_post_trainer(self, cfg, trainer):
        LOGGER.info("Adding TimeStop callback to the trainer")
        callback = TimeStopCallbackHandler(cfg.max_time_s)
        return [callback]
