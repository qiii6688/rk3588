import torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.nn.modules.head import Detect

# 替换 forward
def new_forward(self, x):
    outputs = []
    for i in range(self.nl):
        feat = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        bs, _, h, w = feat.shape
        box = feat[:, :self.reg_max * 4, :, :]
        cls = feat[:, self.reg_max * 4:, :, :]
        # DFL
        box = box.reshape(bs, 4, self.reg_max, h, w)
        box = F.softmax(box, dim=2)
        weight = torch.arange(self.reg_max, dtype=box.dtype, device=box.device)
        weight = weight.reshape(1, 1, self.reg_max, 1, 1)
        box = (box * weight).sum(dim=2)  # [bs, 4, h, w]
        out = torch.cat([box, cls], dim=1)  # [bs, 4+nc, h, w]
        outputs.append(out)
    return tuple(outputs)

Detect.forward = new_forward

# 改成你的权重路径
model = YOLO(r'C:\Users\123\desktop\best.pt')
model.export(format='onnx', opset=12, simplify=True, imgsz=640)
print("done!")
