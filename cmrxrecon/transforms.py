
class normalize(object):
    def __init__(self, mean_input, mean_target, std_input, std_target):
        self.mean_input = mean_input.unsqueeze(1).unsqueeze(2)
        self.mean_target = mean_target.unsqueeze(1).unsqueeze(2)
        self.std_input = std_input.unsqueeze(1).unsqueeze(2)
        self.std_target = std_target.unsqueeze(1).unsqueeze(2)

    def __call__(self, sample):
        input, target = sample

        input = (input - self.mean_input)/self.std_input
        target = (target - self.mean_target)/self.std_target

        return (input, target)

class normalize_sample(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        input, target = sample

        input_mean = input.mean((2, 3)).unsqueeze(2).unsqueeze(2)
        input_std = input.std((2, 3)).unsqueeze(2).unsqueeze(2)
        target_mean = target.mean((2, 3)).unsqueeze(2).unsqueeze(2)
        target_std = target.std((2, 3)).unsqueeze(2).unsqueeze(2)

        input = (input - input_mean)/input_std
        target = (target - target_mean)/target_std

        return (input, target)

class log_transform(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        input, target = sample

        return (input.log(), target.log())