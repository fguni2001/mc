import os
import argparse
from tqdm import tqdm 

import result_renderer  

def encode_video(input_path, output_path, ffmpeg_path="ffmpeg"):
    """Encode AVI to H264 MP4 using ffmpeg."""
    import subprocess
    cmd = [
        ffmpeg_path, "-y", "-i", input_path, "-c:v", "libx264",
        "-preset", "slow", "-crf", "21", output_path
    ]

    subprocess.call(cmd)

def cli_args():
    parser = argparse.ArgumentParser(
        description="VideoMaker: Compose and (optionally) encode tracking results to video."
    )
    parser.add_argument("--seq_root", required=True, help="Path to dataset sequences (e.g. MOT16/test)")
    parser.add_argument("--tracks_dir", required=True, help="Folder with tracking results (.txt or .npy files)")
    parser.add_argument("--video_out", required=True, help="Folder to save output videos")
    parser.add_argument("--h264", action="store_true", help="If set, also create .mp4 encoded videos (needs ffmpeg)")
    parser.add_argument("--frame_interval", default=None, help="Milliseconds between frames (optional)")
    return parser.parse_args()

def main():
    args = cli_args()
    os.makedirs(args.video_out, exist_ok=True)
    results = [f for f in os.listdir(args.tracks_dir) if f.endswith('.txt') and not f.startswith('.')]

    print(f"VideoMaker started!\nProcessing {len(results)} result files...")

    # Pass 1: Create .avi videos
    for res_file in tqdm(results, desc="Creating AVI videos"):
        seq_name = os.path.splitext(res_file)[0]
        seq_path = os.path.join(args.seq_root, seq_name)
        if not os.path.isdir(seq_path):
            print(f"Warning: Skipping {seq_name} (sequence folder not found)")
            continue
        result_path = os.path.join(args.tracks_dir, res_file)
        avi_output = os.path.join(args.video_out, f"{seq_name}.avi")

        print(f"\nRendering: {res_file} → {avi_output}")
        result_renderer.run(
            sequence_path=seq_path,
            tracking_results_file=result_path,
            highlight_false_alarms=False,
            detection_data_file=None,
            frame_interval_ms=args.frame_interval,
            output_video_file=avi_output
        )

    if args.h264:
        print("\nEncoding to MP4...")
        for res_file in tqdm(results, desc="Encoding MP4"):
            seq_name = os.path.splitext(res_file)[0]
            avi_in = os.path.join(args.video_out, f"{seq_name}.avi")
            mp4_out = os.path.join(args.video_out, f"{seq_name}.mp4")
            if os.path.exists(avi_in):
                encode_video(avi_in, mp4_out)
            else:
                print(f"Missing AVI for {seq_name}, skipping MP4 encoding.")

    print("\nAll videos processed and saved to:", args.video_out)

if __name__ == "__main__":
    main()
