from retarget.retarget_task_base import RetargetTaskBase


class FKRetargetTask(RetargetTaskBase):
    def __init__(self, task_config, *args, **kwargs):
        super().__init__(task_config, *args, **kwargs)
        # Initialize FK retargeting specific parameters here

    def run(self, *args, **kwargs):
        # Implement the FK retargeting logic here
        super().run(*args, **kwargs)