def step_fn():
    name = 'Step function'

    def fn(u):
        if u > 0:
            return 1
        else:
            return 0

    return [name, fn]
