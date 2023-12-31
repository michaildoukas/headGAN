import cv2
import numpy as np
import mxnet as mx
from skimage import transform as trans

import insightface
from . import img_helper


arcface_src = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041] ], dtype=np.float32 )

def estimate_trans(landmark, input_size, s = 0.8):
  assert landmark.shape==(5,2)
  if s>0.0:
    default_input_size = 224
    S = s*2.0
    D = (2.0-S)/4
    src = arcface_src*S
    src += default_input_size*D
    src[:,1] -= 20
    scale = float(input_size) / default_input_size
    src *= scale

  tform = trans.SimilarityTransform()
  tform.estimate(landmark, src)
  M = tform.params[0:2,:]
  return M

class Handler:
  def __init__(self, prefix, epoch, im_size=128, ctx_id=0):
    if ctx_id>=0:
      ctx = mx.gpu(ctx_id)
    else:
      ctx = mx.cpu()
    if not isinstance(prefix, list):
        prefix = [prefix]
    image_size = (im_size, im_size)
    self.models = []
    for pref in prefix:
        sym, arg_params, aux_params = mx.model.load_checkpoint(pref, epoch)
        all_layers = sym.get_internals()
        sym = all_layers['fc1_output']
        self.image_size = image_size
        model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
        model.bind(for_training=False, data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
        model.set_params(arg_params, aux_params)
        self.models.append(model)
    self.detector = insightface.model_zoo.get_model('retinaface_r50_v1')
    self.detector.prepare(ctx_id=ctx_id)
    self.aug = 1
    self.aug_value = 5

  def get(self, img):
    out = []
    out_lands = []
    limit = 512
    det_scale = 1.0
    if min(img.shape[0:2])>limit:
      det_scale = float(limit)/min(img.shape[0:2])
    bboxes, landmarks = self.detector.detect(img, scale=det_scale)
    if bboxes.shape[0]==0:
        return out
    for fi in range(bboxes.shape[0]):
      bbox = bboxes[fi]
      landmark = landmarks[fi]
      input_blob = np.zeros( (self.aug, 3)+self.image_size,dtype=np.uint8 )
      M_list = []
      for retry in range(self.aug):
          w, h = (bbox[2]-bbox[0]), (bbox[3]-bbox[1])
          center = (bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2
          rotate = 0
          _scale = 128.0/max(w,h)
          rimg, M = img_helper.transform(img, center, self.image_size[0], _scale, rotate)
          rimg = cv2.cvtColor(rimg, cv2.COLOR_BGR2RGB)
          rimg = np.transpose(rimg, (2,0,1))
          input_blob[retry] = rimg
          M_list.append(M)
      data = mx.nd.array(input_blob)
      db = mx.io.DataBatch(data=(data,))
      for model in self.models:
          model.forward(db, is_train=False)
      X = None
      for model in self.models:
          x = model.get_outputs()[-1].asnumpy()
          if X is None:
              X = x
          else:
              X += x
      X /= len(self.models)
      if X.shape[1]>=3000:
        X = X.reshape( (X.shape[0], -1, 3) )
      else:
        X = X.reshape( (X.shape[0], -1, 2) )
      X[:,:,0:2] += 1
      X[:,:,0:2] *= (self.image_size[0]//2)
      if X.shape[2]==3:
        X[:,:,2] *= (self.image_size[0]//2)
      for i in range(X.shape[0]):
        M = M_list[i]
        IM = cv2.invertAffineTransform(M)
        x = X[i]
        x = img_helper.trans_points(x, IM)
        X[i] = x
      ret = np.mean(X, axis=0)
      out.append(ret)
      out_lands.append(landmark)
    return out, out_lands
