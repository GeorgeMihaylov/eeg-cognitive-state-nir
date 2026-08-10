from pathlib import Path

class PersonalizedModelStore:
    def __init__(self, directory): self.directory = Path(directory)
    def path_for(self, user_id): return self.directory / f"{user_id}.model"
    def save(self, user_id, model): model.save(self.path_for(user_id))
    def load(self, user_id, loader): return loader(self.path_for(user_id))
