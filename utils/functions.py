class fn():
    def __init__(self, name, u):
        self.name = name
        self.u = u

    def run(self):
        pass


def step_fn():
    name = 'Step function'

    def run(u):
        if u > 0:
            return 1
        else:
            return 0

    return [name, run]
