import hjson
import os

__PATH = os.path.dirname(os.path.abspath(__file__))

user = hjson.load(open(f"{__PATH}/user.json", 'r'))
version_to_train = hjson.load(open(f"{__PATH}/version_to_train.json", 'r'))

