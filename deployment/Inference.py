from io import BytesIO
import PIL.Image as Image
import base64

import numpy as np
import torch
from model import LR

class Inference(object):
    def __init__(self):
        self.device = 'cpu'
        self.model = LR().to(self.device)
        self.model.load_state_dict(torch.load("./model.pt"))

    def predict(self, X, feature_names):
        X = X[0].encode()
        im_bytes = base64.b64decode(X)
        im_file = BytesIO(im_bytes)
        img = Image.open(im_file)
        img = np.asarray(img.resize((128, 128))).astype(np.float32) / 255.0
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        img = np.transpose(img, (-1, 0, 1))
        img = np.expand_dims(img, axis=0)
        img = torch.from_numpy(img).to(self.device)
        op = torch.nn.Sigmoid()(self.model(img))
        op = torch.where(op > 0.5, 1, 0).item()
        return [op]


if __name__ == "__main__":
    img = Image.open("test.jpg")
    im_file = BytesIO()
    img.save(im_file, format="JPEG")
    im_bytes = im_file.getvalue()  # im_bytes: image in binary format.
    im_b64 = base64.b64encode(im_bytes)
    im_b64 = im_b64.decode()
    op = Inference().predict(im_b64, ["Image"])
    print(op)

