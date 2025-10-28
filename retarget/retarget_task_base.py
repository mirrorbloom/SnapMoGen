class RetargetTaskBase:
    def __init__(self, task_config, *args, **kwargs):
        self.task_config = task_config

    def run(self, *args, **kwargs):
        pass