from typing import List
import os
from PIL import Image, ImageSequence
import decord
import re

VALID_DATA_FORMAT_STRING = "Input data must be {'.jpg', '.jpeg', '.png', '.tif'} for image; or {'.mp4', '.avi', '.webm', '.mov', '.mkv', '.wmv', '.gif'}  for videos!"

# 均匀抽帧，必采样首尾帧。
def sample_frame_indices(start_frame, total_frames: int, n_frames: int):
    if n_frames == 1:
        return [0]  # sample first frame in default
    sample_ids = [round(i * (total_frames - 1) / (n_frames - 1)) for i in range(n_frames)]
    sample_ids = [i + start_frame for i in sample_ids]
    return sample_ids

def sample_video(
    video_path: str, 
    n_frames: int = None,
    start_time: int = 0,
    end_time: int = -1
    ) -> List[Image.Image]:

    assert os.path.exists(video_path), f"File not found: {video_path}"
    vr = decord.VideoReader(video_path, num_threads=1, ctx=decord.cpu(0))
    vr.seek(0)
    total_frames = len(vr)
    fps = vr.get_avg_fps()

    start_frame = 0
    # end_frame = total_frames - 1
    end_frame = total_frames * 1 // 2 - 1
    if start_time > 0:
        start_frame = min((total_frames-1), int(fps*start_time))
    if end_time > 0:
        end_frame = max(start_frame, int(fps*end_time))
        end_frame = min(end_frame, (total_frames-1))
    frame_indices = sample_frame_indices(
        start_frame=start_frame,
        total_frames=end_frame - start_frame + 1,
        n_frames=n_frames,
    )

    frames = vr.get_batch(frame_indices).asnumpy()
    frames = [Image.fromarray(f).convert('RGB') for f in frames]
    return frames

def sample_gif(
        gif_path: str,
        n_frames:int = None,
        start_time: int = 0,
        end_time: int = -1
    ) -> List[Image.Image]:

    assert os.path.exists(gif_path), f"File not found: {gif_path}"
    
    gif_frames = Image.open(gif_path)

    start_frame = 0
    end_frame = gif_frames.n_frames - 1
    frame_indices = sample_frame_indices(
        start_frame=start_frame,
        total_frames=end_frame - start_frame + 1,
        n_frames=n_frames,
    )
        
    frames = []
    i = 0
    for frame in ImageSequence.Iterator(gif_frames):
        if i in frame_indices:
            frames.append(frame.convert('RGB'))
        i += 1
    return frames

def sample_image(
    image_path: str, 
    n_frames: int = None,
    start_time: int = 0,
    end_time: int = -1
    ):
    assert os.path.exists(image_path), f"File not found: {image_path}"
    image = Image.open(image_path).convert('RGB')
    return [image]

def get_visual_type(input_file):
    ext = os.path.splitext(input_file)[-1]
    if ext in {'.gif'}:
        return 'gif'
    elif ext in {'.mp4', '.avi', '.webm', '.mov', '.mkv', '.wmv'}:
        return 'video'
    elif ext in {'.jpg', '.jpeg', '.png', '.tif'}:
        return 'image'
    else:
        print(f"{VALID_DATA_FORMAT_STRING} But found {ext}!")
        return 'unk'

def get_benchmarks(benchmarks):
    final_benchmarks = []
    type2bm = {
        'dream': ['dream'],
        'caption': ['msvd-caption', 'msr-vtt-caption', 'vatex-caption'],
        'mc_qa': ['next-qa', 'egoschema', 'mvbench', 'video-mme'],
        'oe_qa': ['msvd-qa', 'msr-vtt-qa', 'tgif-qa', 'anet-qa'],
    }
    for bm in benchmarks:
        bm = bm.lower()
        if bm in final_benchmarks:
            continue
        if bm == 'all':
            for v in type2bm.values():
                final_benchmarks.extend(v)
            return final_benchmarks
        if bm in type2bm:
            final_benchmarks.extend(type2bm[bm])
        else:
            final_benchmarks.append(bm)
    return final_benchmarks

class EasyDict(dict):
    """
    Get attributes

    >>> d = EasyDict({'foo':3})
    >>> d['foo']
    3
    >>> d.foo
    3
    >>> d.bar
    Traceback (most recent call last):
    ...
    AttributeError: 'EasyDict' object has no attribute 'bar'

    Works recursively

    >>> d = EasyDict({'foo':3, 'bar':{'x':1, 'y':2}})
    >>> isinstance(d.bar, dict)
    True
    >>> d.bar.x
    1
    """

    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {}
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)
        # Class attributes
        for k in self.__class__.__dict__.keys():
            if not (k.startswith("__") and k.endswith("__")) and not k in ("update", "pop"):
                setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x) if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
        super(EasyDict, self).__setattr__(name, value)
        super(EasyDict, self).__setitem__(name, value)

    __setitem__ = __setattr__

    def update(self, e=None, **f):
        d = e or dict()
        d.update(f)
        for k in d:
            setattr(self, k, d[k])

    def pop(self, k, d=None):
        if hasattr(self, k):
            delattr(self, k)
        return super(EasyDict, self).pop(k, d)
    
conv_tarsier = EasyDict({
    "system": "",
    "roles": ("USER", "ASSISTANT"),
    "messages": [],
    "sep1": " ",
    "sep2": "</s>",
}
)

IMAGE_TOKEN = "<image>"
VIDEO_TOKEN = "<video>"

def get_prompt(conv):
    ret = ""
    if conv.system:
        ret = conv.system + conv.sep1
    for i, (role, message) in enumerate(conv.messages):
        if message:
            # In current version, the image should be add at the first conversation round.
            # So we need to remove the special image tokens in following user input.
            if i > 0:
                message = re.sub(f"({IMAGE_TOKEN}|{VIDEO_TOKEN})\n*", "", message)
            ret += role + ": " + message
            if i % 2:
                ret += conv.sep2
            else:
                ret += conv.sep1
        else:
            ret += role + ":"
    return ret