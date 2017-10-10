import json

from pascal_voc_loader import PascalVOCLoader
from camvid_loader import CamvidLoader


def get_loader(name):
    """get_loader
    :param name:
    """
    return {
        'pascal': PascalVOCLoader,
        'camvid': CamvidLoader
    }[name]