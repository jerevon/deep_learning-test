from saliency import SaliencyMap
from utils import OpencvIo
oi = OpencvIo()
src = oi.imread('..\IMG_7842.jpg')

sm = SaliencyMap(src)
oi.imshow_array([sm.map])