from copy import deepcopy

class FewShotCalibrator:
    def calibrate(self, model, user_id, calibration_features, calibration_labels):
        adapted = deepcopy(model)
        adapted.fit(calibration_features, calibration_labels)
        adapted.version = f"{model.version}-user-{user_id}"
        return adapted
