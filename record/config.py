import json
import os


class Config(dict):
    def __init__(self, filename):
        self.filename = filename
        super().__init__()

    def __setitem__(self, key, value):
        # Load the existing data
        data = self.load()
        # Update the data
        data[key] = value
        # Save the updated data
        self.save(data)

    def __getitem__(self, key):
        # Load the existing data
        data = self.load()
        # Get the requested value
        return data.get(key)

    def load(self):
        if os.path.exists(self.filename):
            with open(self.filename, "r") as f:
                return json.load(f)
        else:
            return {}

    def save(self, data):
        with open(self.filename, "w") as f:
            json.dump(data, f, indent=4)
