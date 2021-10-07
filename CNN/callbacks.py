import pickle


class Callback:
    def __init__(self):
        pass

    @staticmethod
    def get_name():
        raise NotImplementedError("abstract name")

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("abstract callback")


# TODO: when implementing strategy, could create abstract dumper parent with implemented name
class DumpModelPickleCallback(Callback):
    def __init__(self, save_path):
        super(DumpModelPickleCallback, self).__init__()
        self.save_path = save_path

    @staticmethod
    def get_name():
        return "SaveModel"

    def __call__(self, *args, **kwargs):
        # TODO: implement dumper strategy with variable format
        with open(self.save_path, 'wb') as file:
            pickle.dump(args[0], file)
