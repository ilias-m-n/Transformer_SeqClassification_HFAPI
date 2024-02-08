from transformers import TrainerCallback#, TrainerState, Trainer
import copy

class CustomCallback(TrainerCallback):
    def __init__(self, trainer):
        self._trainer = trainer
        self.all_log_history = []
        #self.all_args_history = []

    def on_train_end(self, args, state, control, **kwargs):
        # Save a copy of the log history at the end of each training
        self.all_log_history.append(copy.deepcopy(self._trainer.state.log_history))
        #self.all_args_history.append(copy.deepcopy(self._trainer.args))