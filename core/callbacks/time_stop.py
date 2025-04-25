import time
import math
import logging
from collections import deque
from transformers import TrainerCallback, TrainerState, TrainerControl

logger = logging.getLogger("TimeStopCallback")
logger.setLevel(logging.INFO)

class TimeStopCallback(TrainerCallback):
    def __init__(self, max_time_s: float):
        super().__init__()
        self.max_time_s = max_time_s
        self.window = 5
        self.start_time = None
        self.step_times = deque(maxlen=self.window)
        self._last_step_start = None

    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        self.start_time = time.time()
        logger.info(f"[TimeStop] Training started with budget {self.max_time_s}s")
        return control

    def on_step_begin(self, args, state, control, **kwargs):
        self._last_step_start = time.time()
        return control

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        now = time.time()
        if self._last_step_start is not None:
            self.step_times.append(now - self._last_step_start)

        avg_step = sum(self.step_times) / len(self.step_times) if self.step_times else 0
        elapsed = now - self.start_time
        remaining_s = self.max_time_s - elapsed

        if avg_step > 0:
            new_max = math.floor(remaining_s / avg_step)
            logger.info(
                f"[TimeStop] step={state.global_step} "
                f"avg_step={avg_step:.2f}s elapsed={elapsed:.1f}s "
                f"remaining={remaining_s:.1f}s → setting max_steps={new_max}"
            )
            state.max_steps = new_max

        return control
