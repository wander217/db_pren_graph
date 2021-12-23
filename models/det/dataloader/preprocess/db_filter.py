
class DBFilter:
    def __init__(self, key:list):
        self.key: list = key

    def build(self, data: dict) -> dict:
        for item in self.key:
            if item in data:
                del data[item]
        return data
