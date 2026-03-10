样本（注意 <video><audio> 这两个占位符必须在文本里出现且顺序满足“video 后紧跟 audio”）：
{
  "messages": [
    {"role": "user", "content": "<video><audio>结合画面与声音回答：..."},
    {"role": "assistant", "content": "..."}
  ],
  "videos": ["data/0001.mp4"],
  "audios": ["data/0001.wav"]
}

训练 yaml 里加一行（顶层即可）：
use_audio_in_video: true
video_max_pixels: 307200   # 例如 512*512
video_min_pixels: 65536    # 例如 256*256（按需）
video_fps: 2.0
video_maxlen: 512


