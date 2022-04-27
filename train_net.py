from paz.models.detection.ssd300 import SSD300

if __name__ == "__main__":
    model = SSD300()
    model.fit()
    model.evaluate()
